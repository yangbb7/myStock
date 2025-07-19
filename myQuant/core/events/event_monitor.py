# -*- coding: utf-8 -*-
"""
事件监控和调试工具
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from ..monitoring.exception_logger import ExceptionLogger
from .event_bus import Event, EventBus, get_event_bus
from .event_handlers import EventHandlerRegistry, get_handler_registry


@dataclass
class EventMetrics:
    """事件指标"""

    event_type: str
    total_count: int = 0
    success_count: int = 0
    error_count: int = 0
    avg_processing_time: float = 0.0
    min_processing_time: float = float("inf")
    max_processing_time: float = 0.0
    last_event_time: datetime = None
    first_event_time: datetime = None

    def update(self, processing_time: float, success: bool = True):
        """更新指标"""
        self.total_count += 1

        if success:
            self.success_count += 1
        else:
            self.error_count += 1

        # 更新处理时间统计
        if processing_time < self.min_processing_time:
            self.min_processing_time = processing_time
        if processing_time > self.max_processing_time:
            self.max_processing_time = processing_time

        # 更新平均处理时间
        current_avg = self.avg_processing_time
        self.avg_processing_time = (
            current_avg * (self.total_count - 1) + processing_time
        ) / self.total_count

        self.last_event_time = datetime.now()
        if self.first_event_time is None:
            self.first_event_time = self.last_event_time

    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_type": self.event_type,
            "total_count": self.total_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.get_success_rate(),
            "avg_processing_time": self.avg_processing_time,
            "min_processing_time": (
                self.min_processing_time
                if self.min_processing_time != float("inf")
                else 0
            ),
            "max_processing_time": self.max_processing_time,
            "first_event_time": (
                self.first_event_time.isoformat() if self.first_event_time else None
            ),
            "last_event_time": (
                self.last_event_time.isoformat() if self.last_event_time else None
            ),
        }


@dataclass
class EventTrace:
    """事件追踪记录"""

    event: Event
    processing_start: datetime
    processing_end: datetime = None
    processing_time: float = None
    handlers_executed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    success: bool = True

    def complete(self, success: bool = True, errors: List[str] = None):
        """完成处理"""
        self.processing_end = datetime.now()
        self.processing_time = (
            self.processing_end - self.processing_start
        ).total_seconds()
        self.success = success
        if errors:
            self.errors.extend(errors)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event": self.event.to_dict(),
            "processing_start": self.processing_start.isoformat(),
            "processing_end": (
                self.processing_end.isoformat() if self.processing_end else None
            ),
            "processing_time": self.processing_time,
            "handlers_executed": self.handlers_executed,
            "errors": self.errors,
            "success": self.success,
        }


class EventMonitor:
    """事件监控器"""

    def __init__(
        self,
        event_bus: EventBus = None,
        handler_registry: EventHandlerRegistry = None,
        config: Dict[str, Any] = None,
    ):
        self.event_bus = event_bus or get_event_bus()
        self.handler_registry = handler_registry or get_handler_registry()
        self.config = config or {}

        self.logger = logging.getLogger(__name__)
        self.exception_logger = ExceptionLogger()

        # 配置参数
        self.max_trace_history = self.config.get("max_trace_history", 10000)
        self.metrics_window = self.config.get("metrics_window", 3600)  # 1小时
        self.enable_detailed_tracing = self.config.get("enable_detailed_tracing", True)

        # 监控数据
        self._metrics: Dict[str, EventMetrics] = defaultdict(lambda: EventMetrics(""))
        self._trace_history: deque = deque(maxlen=self.max_trace_history)
        self._active_traces: Dict[str, EventTrace] = {}

        # 统计信息
        self._monitor_stats = {
            "start_time": datetime.now(),
            "total_events_monitored": 0,
            "events_per_second": 0.0,
            "last_activity": None,
        }

        # 告警阈值
        self._alert_thresholds = {
            "error_rate": self.config.get("error_rate_threshold", 0.1),  # 10%
            "avg_processing_time": self.config.get(
                "processing_time_threshold", 5.0
            ),  # 5秒
            "queue_size": self.config.get("queue_size_threshold", 1000),
        }

        # 告警回调
        self._alert_callbacks: List[Callable] = []

        # 启动监控
        self._monitoring = False
        self._monitor_task = None

        self.logger.info("事件监控器初始化完成")

    async def start(self):
        """启动监控"""
        if self._monitoring:
            return

        self._monitoring = True

        # 注册事件监听器
        await self._register_event_listeners()

        # 启动监控任务
        self._monitor_task = asyncio.create_task(self._monitor_loop())

        self.logger.info("事件监控器已启动")

    async def stop(self):
        """停止监控"""
        if not self._monitoring:
            return

        self._monitoring = False

        # 停止监控任务
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # 取消事件监听器
        await self._unregister_event_listeners()

        self.logger.info("事件监控器已停止")

    async def _register_event_listeners(self):
        """注册事件监听器"""
        # 监听所有事件类型
        from .event_types import get_supported_event_types

        for event_type in get_supported_event_types():
            await self.event_bus.subscribe(event_type, self._on_event)

    async def _unregister_event_listeners(self):
        """取消事件监听器"""
        from .event_types import get_supported_event_types

        for event_type in get_supported_event_types():
            await self.event_bus.unsubscribe(event_type, self._on_event)

    async def _on_event(self, event: Event):
        """事件监听器"""
        try:
            # 更新统计
            self._monitor_stats["total_events_monitored"] += 1
            self._monitor_stats["last_activity"] = datetime.now()

            # 创建追踪记录
            if self.enable_detailed_tracing:
                trace = EventTrace(event=event, processing_start=datetime.now())
                self._active_traces[event.correlation_id] = trace

            # 记录事件开始处理
            start_time = time.time()

            # 模拟处理完成（实际应该在处理器执行完成后调用）
            asyncio.create_task(self._complete_event_processing(event, start_time))

        except Exception as e:
            self.exception_logger.log_exception(
                e,
                {
                    "operation": "monitor_event",
                    "event_type": event.type,
                    "event_id": event.correlation_id,
                },
            )

    async def _complete_event_processing(self, event: Event, start_time: float):
        """完成事件处理监控"""
        try:
            # 等待一小段时间模拟处理
            await asyncio.sleep(0.001)

            processing_time = time.time() - start_time
            success = True  # 在实际实现中应该根据处理结果确定

            # 更新指标
            if event.type not in self._metrics:
                self._metrics[event.type] = EventMetrics(event.type)

            self._metrics[event.type].update(processing_time, success)

            # 完成追踪记录
            if event.correlation_id in self._active_traces:
                trace = self._active_traces[event.correlation_id]
                trace.complete(success)

                # 添加到历史记录
                self._trace_history.append(trace)

                # 移除活跃追踪
                del self._active_traces[event.correlation_id]

            # 检查告警条件
            await self._check_alerts(event.type)

        except Exception as e:
            self.exception_logger.log_exception(
                e,
                {
                    "operation": "complete_event_processing",
                    "event_type": event.type,
                    "event_id": event.correlation_id,
                },
            )

    async def _monitor_loop(self):
        """监控循环"""
        while self._monitoring:
            try:
                await asyncio.sleep(60)  # 每分钟检查一次

                # 更新每秒事件数
                self._update_events_per_second()

                # 清理过期数据
                self._cleanup_old_data()

                # 生成监控报告
                if self.logger.isEnabledFor(logging.DEBUG):
                    report = self.generate_monitoring_report()
                    self.logger.debug(f"监控报告: {report}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.exception_logger.log_exception(e, {"operation": "monitor_loop"})

    def _update_events_per_second(self):
        """更新每秒事件数"""
        now = datetime.now()
        start_time = self._monitor_stats["start_time"]
        duration = (now - start_time).total_seconds()

        if duration > 0:
            self._monitor_stats["events_per_second"] = (
                self._monitor_stats["total_events_monitored"] / duration
            )

    def _cleanup_old_data(self):
        """清理过期数据"""
        # 清理过期的活跃追踪
        now = datetime.now()
        expired_traces = []

        for correlation_id, trace in self._active_traces.items():
            if (now - trace.processing_start).total_seconds() > 300:  # 5分钟超时
                expired_traces.append(correlation_id)

        for correlation_id in expired_traces:
            trace = self._active_traces[correlation_id]
            trace.complete(False, ["Processing timeout"])
            self._trace_history.append(trace)
            del self._active_traces[correlation_id]

    async def _check_alerts(self, event_type: str):
        """检查告警条件"""
        try:
            if event_type not in self._metrics:
                return

            metrics = self._metrics[event_type]
            alerts = []

            # 检查错误率
            error_rate = 1 - metrics.get_success_rate()
            if error_rate > self._alert_thresholds["error_rate"]:
                alerts.append(
                    {
                        "type": "high_error_rate",
                        "event_type": event_type,
                        "current_value": error_rate,
                        "threshold": self._alert_thresholds["error_rate"],
                        "message": f"事件类型 {event_type} 错误率过高: {error_rate:.2%}",
                    }
                )

            # 检查平均处理时间
            if (
                metrics.avg_processing_time
                > self._alert_thresholds["avg_processing_time"]
            ):
                alerts.append(
                    {
                        "type": "high_processing_time",
                        "event_type": event_type,
                        "current_value": metrics.avg_processing_time,
                        "threshold": self._alert_thresholds["avg_processing_time"],
                        "message": f"事件类型 {event_type} 平均处理时间过长: {metrics.avg_processing_time:.3f}s",
                    }
                )

            # 触发告警
            for alert in alerts:
                await self._trigger_alert(alert)

        except Exception as e:
            self.exception_logger.log_exception(
                e, {"operation": "check_alerts", "event_type": event_type}
            )

    async def _trigger_alert(self, alert: Dict[str, Any]):
        """触发告警"""
        alert["timestamp"] = datetime.now().isoformat()

        self.logger.warning(f"事件监控告警: {alert['message']}")

        # 调用告警回调
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                self.exception_logger.log_exception(
                    e,
                    {
                        "operation": "trigger_alert_callback",
                        "alert_type": alert["type"],
                    },
                )

    def add_alert_callback(self, callback: Callable):
        """添加告警回调"""
        self._alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable):
        """移除告警回调"""
        if callback in self._alert_callbacks:
            self._alert_callbacks.remove(callback)

    def get_metrics(self, event_type: str = None) -> Dict[str, Any]:
        """获取指标"""
        if event_type:
            return self._metrics.get(event_type, EventMetrics(event_type)).to_dict()
        else:
            return {
                event_type: metrics.to_dict()
                for event_type, metrics in self._metrics.items()
            }

    def get_trace_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取追踪历史"""
        recent_traces = (
            list(self._trace_history)[-limit:] if limit else list(self._trace_history)
        )
        return [trace.to_dict() for trace in recent_traces]

    def get_active_traces(self) -> List[Dict[str, Any]]:
        """获取活跃追踪"""
        return [trace.to_dict() for trace in self._active_traces.values()]

    def generate_monitoring_report(self) -> Dict[str, Any]:
        """生成监控报告"""
        now = datetime.now()

        # 基础统计
        report = {
            "timestamp": now.isoformat(),
            "monitor_stats": self._monitor_stats.copy(),
            "active_traces_count": len(self._active_traces),
            "trace_history_count": len(self._trace_history),
            "monitored_event_types": list(self._metrics.keys()),
        }

        # 事件类型统计
        event_type_stats = {}
        for event_type, metrics in self._metrics.items():
            event_type_stats[event_type] = {
                "total_count": metrics.total_count,
                "success_rate": metrics.get_success_rate(),
                "avg_processing_time": metrics.avg_processing_time,
                "last_event": (
                    metrics.last_event_time.isoformat()
                    if metrics.last_event_time
                    else None
                ),
            }

        report["event_type_stats"] = event_type_stats

        # 系统健康状态
        total_events = sum(m.total_count for m in self._metrics.values())
        total_errors = sum(m.error_count for m in self._metrics.values())
        overall_error_rate = total_errors / total_events if total_events > 0 else 0

        report["system_health"] = {
            "overall_error_rate": overall_error_rate,
            "total_events_processed": total_events,
            "total_errors": total_errors,
            "events_per_second": self._monitor_stats["events_per_second"],
            "queue_health": (
                "good"
                if overall_error_rate < 0.05
                else "warning" if overall_error_rate < 0.1 else "critical"
            ),
        }

        return report

    def reset_metrics(self):
        """重置指标"""
        self._metrics.clear()
        self._trace_history.clear()
        self._active_traces.clear()
        self._monitor_stats = {
            "start_time": datetime.now(),
            "total_events_monitored": 0,
            "events_per_second": 0.0,
            "last_activity": None,
        }
        self.logger.info("事件监控指标已重置")

    def export_metrics(self, format: str = "json") -> str:
        """导出指标"""
        data = {
            "export_time": datetime.now().isoformat(),
            "monitor_stats": self._monitor_stats,
            "metrics": {k: v.to_dict() for k, v in self._metrics.items()},
            "recent_traces": self.get_trace_history(1000),
        }

        if format.lower() == "json":
            return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的导出格式: {format}")


# 全局事件监控器实例
_global_event_monitor: Optional[EventMonitor] = None


def get_event_monitor() -> EventMonitor:
    """获取全局事件监控器实例"""
    global _global_event_monitor
    if _global_event_monitor is None:
        _global_event_monitor = EventMonitor()
    return _global_event_monitor


async def start_event_monitoring():
    """启动事件监控"""
    monitor = get_event_monitor()
    await monitor.start()


async def stop_event_monitoring():
    """停止事件监控"""
    global _global_event_monitor
    if _global_event_monitor:
        await _global_event_monitor.stop()
        _global_event_monitor = None
