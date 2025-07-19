# -*- coding: utf-8 -*-
"""
事件总线 - 事件驱动架构的核心组件
"""

import asyncio
import logging
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from ..exceptions import MonitoringException
from ..monitoring.exception_logger import ExceptionLogger


class EventPriority(Enum):
    """事件优先级"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:
    """基础事件类"""

    type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: EventPriority = EventPriority.NORMAL
    source: str = None
    correlation_id: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.correlation_id is None:
            self.correlation_id = f"{self.type}_{int(time.time() * 1000000)}"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.name,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }


class EventFilter:
    """事件过滤器"""

    def __init__(
        self,
        event_types: List[str] = None,
        sources: List[str] = None,
        priority_min: EventPriority = None,
    ):
        self.event_types = set(event_types) if event_types else None
        self.sources = set(sources) if sources else None
        self.priority_min = priority_min

    def matches(self, event: Event) -> bool:
        """检查事件是否匹配过滤条件"""
        if self.event_types and event.type not in self.event_types:
            return False

        if self.sources and event.source not in self.sources:
            return False

        if self.priority_min and event.priority.value < self.priority_min.value:
            return False

        return True


class EventSubscription:
    """事件订阅"""

    def __init__(
        self,
        handler: Callable,
        event_filter: EventFilter = None,
        async_handler: bool = None,
    ):
        self.handler = handler
        self.filter = event_filter
        self.async_handler = (
            async_handler
            if async_handler is not None
            else asyncio.iscoroutinefunction(handler)
        )
        self.created_at = datetime.now()
        self.call_count = 0
        self.last_called = None
        self.error_count = 0

    def matches(self, event: Event) -> bool:
        """检查事件是否匹配订阅条件"""
        return self.filter.matches(event) if self.filter else True

    def get_stats(self) -> Dict[str, Any]:
        """获取订阅统计信息"""
        return {
            "handler": str(self.handler),
            "async_handler": self.async_handler,
            "created_at": self.created_at.isoformat(),
            "call_count": self.call_count,
            "last_called": self.last_called.isoformat() if self.last_called else None,
            "error_count": self.error_count,
        }


class EventBus:
    """事件总线 - 支持异步事件发布和订阅"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.exception_logger = ExceptionLogger()

        # 配置参数
        self.max_queue_size = self.config.get("max_queue_size", 10000)
        self.worker_count = self.config.get("worker_count", 4)
        self.batch_size = self.config.get("batch_size", 10)
        self.max_retry_attempts = self.config.get("max_retry_attempts", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)

        # 订阅管理
        self._subscriptions: Dict[str, List[EventSubscription]] = {}
        self._subscription_lock = None

        # 事件队列
        self._event_queue = None
        self._priority_queue = None

        # 运行状态
        self._running = False
        self._workers = []
        self._initialized = False
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "handlers_executed": 0,
            "handlers_failed": 0,
            "average_processing_time": 0.0,
        }

        # 线程池用于同步处理器
        self._thread_pool = ThreadPoolExecutor(max_workers=self.worker_count)

        # 事件历史记录（用于调试）
        self._event_history: List[Event] = []
        self._max_history_size = self.config.get("max_history_size", 1000)

        self.logger.info("事件总线初始化完成")

    async def _init_async_components(self):
        """初始化异步组件"""
        if not self._initialized:
            self._subscription_lock = asyncio.Lock()
            self._event_queue = asyncio.Queue(maxsize=self.max_queue_size)
            self._priority_queue = asyncio.PriorityQueue()
            self._initialized = True

    def _ensure_sync_mode(self):
        """确保同步模式下的订阅管理"""
        if self._subscription_lock is None:
            # 在同步模式下，不需要锁
            pass

    async def start(self):
        """启动事件总线"""
        if self._running:
            return

        # 初始化异步组件
        await self._init_async_components()

        self._running = True

        # 启动工作线程
        for i in range(self.worker_count):
            worker = asyncio.create_task(self._event_worker(f"worker-{i}"))
            self._workers.append(worker)

        self.logger.info(f"事件总线已启动，工作线程数: {self.worker_count}")

    async def stop(self):
        """停止事件总线"""
        if not self._running:
            return

        self._running = False

        # 停止工作线程
        for worker in self._workers:
            if not worker.done():
                worker.cancel()

        # 等待工作线程结束，设置较短的超时
        if self._workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=0.5
                )
            except asyncio.TimeoutError:
                self.logger.warning("工作线程停止超时，强制终止")
            
            # 确保所有任务都被取消
            for worker in self._workers:
                if not worker.done():
                    worker.cancel()
                    try:
                        await worker
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass

        self._workers.clear()

        # 清空队列并标记任务完成
        if self._event_queue:
            while not self._event_queue.empty():
                try:
                    self._event_queue.get_nowait()
                    try:
                        self._event_queue.task_done()
                    except ValueError:
                        # task_done called more times than items were put
                        pass
                except asyncio.QueueEmpty:
                    break
                except Exception:
                    break
                    
        if self._priority_queue:
            while not self._priority_queue.empty():
                try:
                    self._priority_queue.get_nowait()
                    try:
                        self._priority_queue.task_done()
                    except ValueError:
                        # task_done called more times than items were put
                        pass
                except asyncio.QueueEmpty:
                    break
                except Exception:
                    break

        # 关闭线程池
        self._thread_pool.shutdown(wait=False)

        # 重置异步组件
        self._event_queue = None
        self._priority_queue = None

        # 给短暂时间让所有异步操作完成
        await asyncio.sleep(0.01)

        self.logger.info("事件总线已停止")

    def subscribe_sync(self, event_type, handler, event_filter=None):
        """
        同步订阅事件（用于测试兼容性）
        """
        # 处理EventType枚举
        if hasattr(event_type, 'value'):
            event_type = event_type.value
        
        if event_type not in self._subscriptions:
            self._subscriptions[event_type] = []

        subscription = EventSubscription(handler, event_filter)
        self._subscriptions[event_type].append(subscription)

        subscription_id = f"{event_type}_{len(self._subscriptions[event_type])}"
        self.logger.debug(f"新订阅: {subscription_id}, 处理器: {handler}")
        return subscription_id
    
    def subscribe(self, event_type, handler, event_filter=None):
        """
        订阅事件（支持同步和异步调用）
        """
        # 处理EventType枚举
        if hasattr(event_type, 'value'):
            event_type = event_type.value
            
        return self.subscribe_sync(event_type, handler, event_filter)

    async def subscribe_async(
        self, event_type: str, handler: Callable, event_filter: EventFilter = None
    ) -> str:
        """
        异步订阅事件

        Args:
            event_type: 事件类型
            handler: 事件处理器
            event_filter: 事件过滤器

        Returns:
            str: 订阅ID
        """
        # 处理EventType枚举
        if hasattr(event_type, 'value'):
            event_type = event_type.value
            
        # 确保异步组件已初始化
        if self._subscription_lock is None:
            await self._init_async_components()

        async with self._subscription_lock:
            if event_type not in self._subscriptions:
                self._subscriptions[event_type] = []

            subscription = EventSubscription(handler, event_filter)
            self._subscriptions[event_type].append(subscription)

            subscription_id = f"{event_type}_{len(self._subscriptions[event_type])}"

            self.logger.debug(f"新订阅: {subscription_id}, 处理器: {handler}")
            return subscription_id

    async def unsubscribe(self, event_type: str, handler: Callable) -> bool:
        """
        取消订阅

        Args:
            event_type: 事件类型
            handler: 事件处理器

        Returns:
            bool: 是否成功取消订阅
        """
        # 确保异步组件已初始化
        if self._subscription_lock is None:
            await self._init_async_components()
            
        async with self._subscription_lock:
            if event_type not in self._subscriptions:
                return False

            subscriptions = self._subscriptions[event_type]
            original_count = len(subscriptions)

            # 移除匹配的订阅
            self._subscriptions[event_type] = [
                sub for sub in subscriptions if sub.handler != handler
            ]

            removed_count = original_count - len(self._subscriptions[event_type])

            if removed_count > 0:
                self.logger.debug(
                    f"取消订阅: {event_type}, 移除 {removed_count} 个处理器"
                )
                return True

            return False

    def publish(self, event_type_or_event, data=None, sync=True):
        """
        发布事件（支持多种调用方式）
        """
        try:
            # 支持多种调用方式
            if isinstance(event_type_or_event, Event):
                event = event_type_or_event
            else:
                # 处理EventType枚举
                event_type = event_type_or_event
                if hasattr(event_type, 'value'):
                    event_type = event_type.value
                
                event = Event(
                    type=event_type,
                    data=data or {},
                    priority=EventPriority.NORMAL
                )

            # 添加到历史记录
            self._add_to_history(event)

            # 更新统计
            self._stats["events_published"] += 1

            # 同步处理（适用于测试）
            return self._process_event_sync_blocking(event)

        except Exception as e:
            self.logger.error(f"发布事件失败: {e}")
            return False
    
    def _process_event_sync_blocking(self, event):
        """
        同步阻塞处理事件（用于测试）
        """
        try:
            # 获取订阅者
            subscriptions = self._subscriptions.get(event.type, [])
            matching_subscriptions = [
                sub for sub in subscriptions if sub.matches(event)
            ]

            if not matching_subscriptions:
                self.logger.debug(f"事件 {event.type} 没有订阅者")
                return True

            # 执行处理器
            for subscription in matching_subscriptions:
                try:
                    subscription.call_count += 1
                    subscription.last_called = datetime.now()
                    subscription.handler(event)
                except Exception as e:
                    subscription.error_count += 1
                    self.logger.error(f"事件处理器执行失败: {e}")

            self._stats["events_processed"] += 1
            return True
            
        except Exception as e:
            self._stats["events_failed"] += 1
            self.logger.error(f"事件处理失败: {e}")
            return False

    async def publish_async(self, event: Event, sync: bool = False) -> bool:
        """
        异步发布事件

        Args:
            event: 事件对象
            sync: 是否同步处理

        Returns:
            bool: 是否成功发布
        """
        try:
            if not self._running:
                await self.start()

            # 添加到历史记录
            self._add_to_history(event)

            # 更新统计
            self._stats["events_published"] += 1

            if sync:
                # 同步处理
                return await self._process_event_sync(event)
            else:
                # 异步处理：添加到队列
                if event.priority == EventPriority.CRITICAL:
                    # 高优先级事件使用优先级队列
                    await self._priority_queue.put((event.priority.value, event))
                else:
                    # 普通事件使用FIFO队列
                    await self._event_queue.put(event)

                return True

        except Exception as e:
            # 直接使用logger避免coroutine泄漏
            self.logger.error(
                f"异步发布事件失败 {event.type}: {str(e)}", 
                extra={
                    "operation": "publish_event",
                    "event_type": event.type,
                    "event_id": event.correlation_id,
                }
            )
            return False

    async def publish_batch(self, events: List[Event]) -> Dict[str, bool]:
        """
        批量发布事件

        Args:
            events: 事件列表

        Returns:
            Dict: 发布结果字典
        """
        results = {}

        for event in events:
            try:
                success = await self.publish(event)
                results[event.correlation_id] = success
            except Exception as e:
                # 直接使用logger避免coroutine泄漏
                self.logger.error(
                    f"批量发布事件失败 {event.correlation_id}: {str(e)}", 
                    extra={"operation": "publish_batch", "event_id": event.correlation_id}
                )
                results[event.correlation_id] = False

        return results

    async def _event_worker(self, worker_name: str):
        """事件工作线程"""
        self.logger.debug(f"事件工作线程 {worker_name} 启动")

        try:
            while self._running:
                try:
                    event = None

                    # 优先处理高优先级事件  
                    if self._priority_queue and not self._priority_queue.empty():
                        try:
                            priority, event = await asyncio.wait_for(
                                self._priority_queue.get(), timeout=0.05
                            )
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            pass
                    
                    # 处理普通事件
                    if not event and self._event_queue and not self._event_queue.empty():
                        try:
                            event = await asyncio.wait_for(
                                self._event_queue.get(), timeout=0.05
                            )
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            pass

                    if event:
                        await self._process_event(event, worker_name)
                    else:
                        # 没有事件时短暂等待
                        await asyncio.sleep(0.01)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    # 直接使用logger避免coroutine泄漏
                    self.logger.error(f"工作线程 {worker_name} 异常: {str(e)}")

        except asyncio.CancelledError:
            pass
        finally:
            self.logger.debug(f"事件工作线程 {worker_name} 结束")

    async def _process_event(self, event: Event, worker_name: str = None):
        """处理单个事件"""
        start_time = time.time()

        try:
            # 获取订阅者
            async with self._subscription_lock:
                subscriptions = self._subscriptions.get(event.type, [])
                matching_subscriptions = [
                    sub for sub in subscriptions if sub.matches(event)
                ]

            if not matching_subscriptions:
                self.logger.debug(f"事件 {event.type} 没有订阅者")
                return

            # 执行处理器
            tasks = []
            for subscription in matching_subscriptions:
                if subscription.async_handler:
                    # 异步处理器
                    task = asyncio.create_task(
                        self._execute_async_handler(subscription, event)
                    )
                    tasks.append(task)
                else:
                    # 同步处理器，使用线程池
                    task = asyncio.create_task(
                        self._execute_sync_handler(subscription, event)
                    )
                    tasks.append(task)

            # 等待所有处理器完成
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 统计结果
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            error_count = len(results) - success_count

            self._stats["events_processed"] += 1
            self._stats["handlers_executed"] += len(results)
            self._stats["handlers_failed"] += error_count

            if error_count > 0:
                self._stats["events_failed"] += 1
                self.logger.warning(
                    f"事件 {event.type} 处理完成，{success_count} 成功，{error_count} 失败"
                )

            # 更新平均处理时间
            processing_time = time.time() - start_time
            total_events = self._stats["events_processed"]
            current_avg = self._stats["average_processing_time"]
            self._stats["average_processing_time"] = (
                current_avg * (total_events - 1) + processing_time
            ) / total_events

        except Exception as e:
            self._stats["events_failed"] += 1
            # 直接使用logger避免coroutine泄漏
            self.logger.error(
                f"事件处理失败 {event.type}: {str(e)}", 
                extra={
                    "operation": "process_event",
                    "event_type": event.type,
                    "event_id": event.correlation_id,
                    "worker": worker_name,
                }
            )

    async def _process_event_sync(self, event: Event) -> bool:
        """同步处理事件"""
        try:
            await self._process_event(event, "sync")
            return True
        except Exception as e:
            # 直接使用logger避免coroutine泄漏
            self.logger.error(
                f"同步事件处理失败 {event.type}: {str(e)}", 
                extra={"operation": "process_event_sync", "event_type": event.type}
            )
            return False

    async def _execute_async_handler(
        self, subscription: EventSubscription, event: Event
    ):
        """执行异步处理器"""
        try:
            subscription.call_count += 1
            subscription.last_called = datetime.now()

            await subscription.handler(event)

        except Exception as e:
            subscription.error_count += 1
            # 直接使用logger避免coroutine泄漏
            self.logger.error(
                f"异步处理器执行失败 {event.type}: {str(e)}", 
                extra={
                    "operation": "execute_async_handler",
                    "handler": str(subscription.handler),
                    "event_type": event.type,
                    "event_id": event.correlation_id,
                }
            )
            raise

    async def _execute_sync_handler(
        self, subscription: EventSubscription, event: Event
    ):
        """执行同步处理器"""
        try:
            subscription.call_count += 1
            subscription.last_called = datetime.now()

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._thread_pool, subscription.handler, event)

        except Exception as e:
            subscription.error_count += 1
            # 直接使用logger避免coroutine泄漏
            self.logger.error(
                f"同步处理器执行失败 {event.type}: {str(e)}", 
                extra={
                    "operation": "execute_sync_handler",
                    "handler": str(subscription.handler),
                    "event_type": event.type,
                    "event_id": event.correlation_id,
                }
            )
            raise

    def _add_to_history(self, event: Event):
        """添加事件到历史记录"""
        self._event_history.append(event)

        # 限制历史记录大小
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size :]

    def get_stats(self) -> Dict[str, Any]:
        """获取事件总线统计信息"""
        stats = self._stats.copy()
        stats.update(
            {
                "queue_size": self._event_queue.qsize() if self._event_queue else 0,
                "priority_queue_size": self._priority_queue.qsize() if self._priority_queue else 0,
                "subscription_count": sum(
                    len(subs) for subs in self._subscriptions.values()
                ),
                "event_types": list(self._subscriptions.keys()),
                "running": self._running,
                "worker_count": len(self._workers),
            }
        )
        return stats

    def get_subscription_stats(self) -> Dict[str, List[Dict[str, Any]]]:
        """获取订阅统计信息"""
        stats = {}
        for event_type, subscriptions in self._subscriptions.items():
            stats[event_type] = [sub.get_stats() for sub in subscriptions]
        return stats

    def get_event_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取事件历史记录"""
        recent_events = self._event_history[-limit:] if limit else self._event_history
        return [event.to_dict() for event in recent_events]

    def clear_history(self):
        """清理事件历史记录"""
        self._event_history.clear()
        self.logger.info("事件历史记录已清理")

    async def wait_for_event(
        self, event_type: str, timeout: float = None, event_filter: EventFilter = None
    ) -> Optional[Event]:
        """
        等待特定事件

        Args:
            event_type: 事件类型
            timeout: 超时时间
            event_filter: 事件过滤器

        Returns:
            Event: 匹配的事件，超时返回None
        """
        event_received = asyncio.Event()
        received_event = None

        async def event_waiter(event: Event):
            nonlocal received_event
            if event_filter is None or event_filter.matches(event):
                received_event = event
                event_received.set()

        # 订阅事件
        await self.subscribe_async(event_type, event_waiter)

        try:
            # 等待事件
            await asyncio.wait_for(event_received.wait(), timeout=timeout)
            return received_event
        except asyncio.TimeoutError:
            return None
        finally:
            # 取消订阅
            await self.unsubscribe(event_type, event_waiter)


# 全局事件总线实例
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """获取全局事件总线实例"""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


async def start_global_event_bus():
    """启动全局事件总线"""
    bus = get_event_bus()
    await bus.start()


async def stop_global_event_bus():
    """停止全局事件总线"""
    global _global_event_bus
    if _global_event_bus:
        await _global_event_bus.stop()
        _global_event_bus = None
