# -*- coding: utf-8 -*-
"""
事件处理器注册和管理
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..monitoring.exception_logger import ExceptionLogger
from .event_bus import Event, EventBus, EventFilter, get_event_bus
from .event_types import *


@dataclass
class HandlerInfo:
    """处理器信息"""

    name: str
    handler: Callable
    event_types: List[str]
    description: str = ""
    enabled: bool = True
    priority: int = 0  # 数字越小优先级越高
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class EventHandlerRegistry:
    """事件处理器注册表"""

    def __init__(self, event_bus: EventBus = None):
        self.event_bus = event_bus or get_event_bus()
        self.logger = logging.getLogger(__name__)
        self.exception_logger = ExceptionLogger()

        # 处理器注册表
        self._handlers: Dict[str, HandlerInfo] = {}
        self._subscriptions: Dict[str, str] = {}  # handler_name -> subscription_id

        # 内置处理器
        self._register_builtin_handlers()

        self.logger.info("事件处理器注册表初始化完成")

    def register_handler(
        self,
        name: str,
        handler: Callable,
        event_types: List[str],
        description: str = "",
        priority: int = 0,
        auto_subscribe: bool = True,
    ) -> bool:
        """
        注册事件处理器

        Args:
            name: 处理器名称
            handler: 处理器函数
            event_types: 支持的事件类型列表
            description: 处理器描述
            priority: 优先级
            auto_subscribe: 是否自动订阅

        Returns:
            bool: 是否注册成功
        """
        try:
            if name in self._handlers:
                self.logger.warning(f"处理器 {name} 已存在，将被覆盖")
                self.unregister_handler(name)

            handler_info = HandlerInfo(
                name=name,
                handler=handler,
                event_types=event_types,
                description=description,
                priority=priority,
            )

            self._handlers[name] = handler_info

            if auto_subscribe:
                asyncio.create_task(self._subscribe_handler(name))

            self.logger.info(f"处理器 {name} 注册成功, 事件类型: {event_types}")
            return True

        except Exception as e:
            self.exception_logger.log_exception(
                e, {"operation": "register_handler", "handler_name": name}
            )
            return False

    def unregister_handler(self, name: str) -> bool:
        """
        注销事件处理器

        Args:
            name: 处理器名称

        Returns:
            bool: 是否注销成功
        """
        try:
            if name not in self._handlers:
                self.logger.warning(f"处理器 {name} 不存在")
                return False

            # 取消订阅
            asyncio.create_task(self._unsubscribe_handler(name))

            # 移除处理器
            del self._handlers[name]

            self.logger.info(f"处理器 {name} 注销成功")
            return True

        except Exception as e:
            self.exception_logger.log_exception(
                e, {"operation": "unregister_handler", "handler_name": name}
            )
            return False

    def enable_handler(self, name: str) -> bool:
        """启用处理器"""
        if name in self._handlers:
            self._handlers[name].enabled = True
            asyncio.create_task(self._subscribe_handler(name))
            self.logger.info(f"处理器 {name} 已启用")
            return True
        return False

    def disable_handler(self, name: str) -> bool:
        """禁用处理器"""
        if name in self._handlers:
            self._handlers[name].enabled = False
            asyncio.create_task(self._unsubscribe_handler(name))
            self.logger.info(f"处理器 {name} 已禁用")
            return True
        return False

    async def _subscribe_handler(self, name: str):
        """订阅处理器"""
        if name not in self._handlers:
            return

        handler_info = self._handlers[name]
        if not handler_info.enabled:
            return

        # 取消已有订阅
        await self._unsubscribe_handler(name)

        # 创建包装处理器
        async def wrapped_handler(event: Event):
            try:
                if handler_info.enabled:
                    if asyncio.iscoroutinefunction(handler_info.handler):
                        await handler_info.handler(event)
                    else:
                        handler_info.handler(event)
            except Exception as e:
                self.exception_logger.log_exception(
                    e,
                    {
                        "operation": "execute_handler",
                        "handler_name": name,
                        "event_type": event.type,
                    },
                )

        # 为每个事件类型订阅
        for event_type in handler_info.event_types:
            try:
                subscription_id = await self.event_bus.subscribe(
                    event_type, wrapped_handler
                )
                self._subscriptions[f"{name}_{event_type}"] = subscription_id
            except Exception as e:
                self.exception_logger.log_exception(
                    e,
                    {
                        "operation": "subscribe_handler",
                        "handler_name": name,
                        "event_type": event_type,
                    },
                )

    async def _unsubscribe_handler(self, name: str):
        """取消订阅处理器"""
        if name not in self._handlers:
            return

        handler_info = self._handlers[name]

        # 取消所有相关订阅
        for event_type in handler_info.event_types:
            subscription_key = f"{name}_{event_type}"
            if subscription_key in self._subscriptions:
                try:
                    await self.event_bus.unsubscribe(event_type, handler_info.handler)
                    del self._subscriptions[subscription_key]
                except Exception as e:
                    self.exception_logger.log_exception(
                        e,
                        {
                            "operation": "unsubscribe_handler",
                            "handler_name": name,
                            "event_type": event_type,
                        },
                    )

    def get_handlers(self) -> Dict[str, HandlerInfo]:
        """获取所有处理器"""
        return self._handlers.copy()

    def get_handler(self, name: str) -> Optional[HandlerInfo]:
        """获取特定处理器"""
        return self._handlers.get(name)

    def list_handlers_by_event_type(self, event_type: str) -> List[HandlerInfo]:
        """按事件类型列出处理器"""
        return [
            handler_info
            for handler_info in self._handlers.values()
            if event_type in handler_info.event_types and handler_info.enabled
        ]

    def _register_builtin_handlers(self):
        """注册内置处理器"""

        # 系统日志处理器
        def system_log_handler(event: Event):
            """系统事件日志处理器"""
            if event.type == "system":
                action = event.data.get("action", "unknown")
                system_data = event.data.get("system_data", {})

                if action == "error":
                    self.logger.error(f"系统错误: {system_data}")
                elif action == "startup":
                    self.logger.info(f"系统启动: {system_data}")
                elif action == "shutdown":
                    self.logger.info(f"系统关闭: {system_data}")
                else:
                    self.logger.info(f"系统事件 {action}: {system_data}")

        # 风险告警处理器
        def risk_alert_handler(event: Event):
            """风险事件告警处理器"""
            if event.type == "risk":
                risk_type = event.data.get("risk_type", "unknown")
                severity = event.data.get("severity", "warning")
                risk_data = event.data.get("risk_data", {})

                if severity == "critical":
                    self.logger.critical(f"严重风险告警 {risk_type}: {risk_data}")
                elif severity == "warning":
                    self.logger.warning(f"风险告警 {risk_type}: {risk_data}")
                else:
                    self.logger.info(f"风险信息 {risk_type}: {risk_data}")

        # 交易日志处理器
        def trade_log_handler(event: Event):
            """交易事件日志处理器"""
            if event.type == "trade":
                trade_id = event.data.get("trade_id", "unknown")
                order_id = event.data.get("order_id", "unknown")
                trade_data = event.data.get("trade_data", {})

                self.logger.info(
                    f"交易执行 - Trade ID: {trade_id}, Order ID: {order_id}, Data: {trade_data}"
                )

        # 订单状态处理器
        def order_status_handler(event: Event):
            """订单状态处理器"""
            if event.type == "order":
                order_id = event.data.get("order_id", "unknown")
                action = event.data.get("action", "unknown")
                order_data = event.data.get("order_data", {})

                self.logger.info(
                    f"订单状态变更 - Order ID: {order_id}, Action: {action}, Data: {order_data}"
                )

        # 注册内置处理器
        self.register_handler(
            "system_logger",
            system_log_handler,
            ["system"],
            "系统事件日志记录器",
            priority=10,
            auto_subscribe=False,
        )

        self.register_handler(
            "risk_alerter",
            risk_alert_handler,
            ["risk"],
            "风险事件告警器",
            priority=1,
            auto_subscribe=False,
        )

        self.register_handler(
            "trade_logger",
            trade_log_handler,
            ["trade"],
            "交易事件日志记录器",
            priority=5,
            auto_subscribe=False,
        )

        self.register_handler(
            "order_tracker",
            order_status_handler,
            ["order"],
            "订单状态跟踪器",
            priority=5,
            auto_subscribe=False,
        )

    async def start_all_handlers(self):
        """启动所有处理器"""
        for name, handler_info in self._handlers.items():
            if handler_info.enabled:
                await self._subscribe_handler(name)

        self.logger.info(f"已启动 {len(self._handlers)} 个事件处理器")

    async def stop_all_handlers(self):
        """停止所有处理器"""
        for name in list(self._handlers.keys()):
            await self._unsubscribe_handler(name)

        self.logger.info("所有事件处理器已停止")

    def get_stats(self) -> Dict[str, Any]:
        """获取处理器统计信息"""
        total_handlers = len(self._handlers)
        enabled_handlers = sum(1 for h in self._handlers.values() if h.enabled)
        active_subscriptions = len(self._subscriptions)

        event_type_coverage = {}
        for handler_info in self._handlers.values():
            for event_type in handler_info.event_types:
                if event_type not in event_type_coverage:
                    event_type_coverage[event_type] = 0
                if handler_info.enabled:
                    event_type_coverage[event_type] += 1

        return {
            "total_handlers": total_handlers,
            "enabled_handlers": enabled_handlers,
            "disabled_handlers": total_handlers - enabled_handlers,
            "active_subscriptions": active_subscriptions,
            "event_type_coverage": event_type_coverage,
            "handler_names": list(self._handlers.keys()),
        }


# 全局处理器注册表实例
_global_handler_registry: Optional[EventHandlerRegistry] = None


def get_handler_registry() -> EventHandlerRegistry:
    """获取全局处理器注册表实例"""
    global _global_handler_registry
    if _global_handler_registry is None:
        _global_handler_registry = EventHandlerRegistry()
    return _global_handler_registry


# 装饰器函数
def event_handler(
    event_types: List[str], name: str = None, description: str = "", priority: int = 0
):
    """
    事件处理器装饰器

    Args:
        event_types: 事件类型列表
        name: 处理器名称
        description: 处理器描述
        priority: 优先级
    """

    def decorator(func):
        handler_name = name or func.__name__
        registry = get_handler_registry()
        registry.register_handler(
            handler_name, func, event_types, description, priority
        )
        return func

    return decorator


async def start_event_system():
    """启动事件系统"""
    # 启动事件总线
    bus = get_event_bus()
    await bus.start()

    # 启动处理器注册表
    registry = get_handler_registry()
    await registry.start_all_handlers()


async def stop_event_system():
    """停止事件系统"""
    # 停止处理器注册表
    global _global_handler_registry
    if _global_handler_registry:
        await _global_handler_registry.stop_all_handlers()
        _global_handler_registry = None

    # 停止事件总线
    await stop_global_event_bus()
