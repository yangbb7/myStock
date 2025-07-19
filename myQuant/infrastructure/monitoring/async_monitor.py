# -*- coding: utf-8 -*-
"""
异步操作监控器 - 提供异步操作的性能监控和错误处理
"""

import asyncio
import json
import logging
import threading
import time
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from core.exceptions import MonitoringException, NetworkException
from core.monitoring.exception_logger import ExceptionLogger


class MonitorLevel(Enum):
    """监控级别"""

    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5


@dataclass
class AsyncOperationMetrics:
    """异步操作指标"""

    operation_id: str
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: str = "running"  # running, completed, failed, timeout
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.end_time and self.start_time:
            self.duration = self.end_time - self.start_time

    def complete(self, error: Exception = None):
        """标记操作完成"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

        if error:
            self.status = "failed"
            self.error_message = str(error)
        else:
            self.status = "completed"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "operation_id": self.operation_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "status": self.status,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class AsyncTaskMonitor:
    """异步任务监控器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.exception_logger = ExceptionLogger()

        # 监控配置
        self.max_metrics_history = self.config.get("max_metrics_history", 10000)
        self.cleanup_interval = self.config.get("cleanup_interval", 300)  # 5分钟
        self.alert_thresholds = self.config.get(
            "alert_thresholds",
            {
                "error_rate": 0.1,  # 10%错误率阈值
                "avg_duration": 30.0,  # 30秒平均耗时阈值
                "timeout_rate": 0.05,  # 5%超时率阈值
            },
        )

        # 存储
        self._metrics: Dict[str, AsyncOperationMetrics] = {}
        self._metrics_history: List[AsyncOperationMetrics] = []
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._task_refs: weakref.WeakSet = weakref.WeakSet()

        # 统计信息
        self._stats = {
            "total_operations": 0,
            "completed_operations": 0,
            "failed_operations": 0,
            "timeout_operations": 0,
            "avg_duration": 0.0,
            "current_active_tasks": 0,
        }

        # 监控状态
        self._monitoring = False
        self._cleanup_task = None
        self._alert_callbacks: List[Callable] = []

        self.logger.info("异步任务监控器初始化完成")

    async def start(self):
        """启动监控器"""
        if self._monitoring:
            return

        self._monitoring = True

        # 启动清理任务
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        self.logger.info("异步任务监控器已启动")

    async def stop(self):
        """停止监控器"""
        if not self._monitoring:
            return

        self._monitoring = False

        # 停止清理任务
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # 取消所有活跃任务
        for task in list(self._active_tasks.values()):
            if not task.done():
                task.cancel()

        self.logger.info("异步任务监控器已停止")

    def monitor_task(self, operation_name: str, metadata: Dict[str, Any] = None):
        """任务监控装饰器"""

        def decorator(func):
            async def wrapper(*args, **kwargs):
                operation_id = f"{operation_name}_{int(time.time() * 1000000)}"

                # 创建监控指标
                metrics = AsyncOperationMetrics(
                    operation_id=operation_id,
                    operation_name=operation_name,
                    start_time=time.time(),
                    metadata=metadata or {},
                )

                # 注册指标
                self._register_metrics(metrics)

                try:
                    # 执行任务
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)

                    # 标记完成
                    metrics.complete()

                    return result

                except Exception as e:
                    # 标记失败
                    metrics.complete(error=e)

                    # 记录异常
                    self.exception_logger.log_exception(
                        e,
                        {
                            "operation_id": operation_id,
                            "operation_name": operation_name,
                            "monitoring": "async_task_monitor",
                        },
                    )

                    raise
                finally:
                    # 更新统计信息
                    self._update_stats(metrics)

                    # 移除活跃任务记录
                    self._active_tasks.pop(operation_id, None)

            return wrapper

        return decorator

    async def monitor_async_generator(
        self,
        generator: AsyncGenerator,
        operation_name: str,
        metadata: Dict[str, Any] = None,
    ) -> AsyncGenerator:
        """监控异步生成器"""
        operation_id = f"{operation_name}_gen_{int(time.time() * 1000000)}"

        metrics = AsyncOperationMetrics(
            operation_id=operation_id,
            operation_name=operation_name,
            start_time=time.time(),
            metadata=metadata or {},
        )

        self._register_metrics(metrics)

        try:
            async for item in generator:
                yield item

            # 生成器正常完成
            metrics.complete()

        except Exception as e:
            # 生成器出错
            metrics.complete(error=e)

            self.exception_logger.log_exception(
                e,
                {
                    "operation_id": operation_id,
                    "operation_name": operation_name,
                    "monitoring": "async_generator_monitor",
                },
            )

            raise
        finally:
            self._update_stats(metrics)

    async def monitor_concurrent_operations(
        self, tasks: List[asyncio.Task], operation_name: str, timeout: float = None
    ) -> List[Any]:
        """监控并发操作"""
        operation_id = f"{operation_name}_concurrent_{int(time.time() * 1000000)}"

        metrics = AsyncOperationMetrics(
            operation_id=operation_id,
            operation_name=operation_name,
            start_time=time.time(),
            metadata={"task_count": len(tasks)},
        )

        self._register_metrics(metrics)

        try:
            # 执行并发任务
            if timeout:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=timeout
                )
            else:
                results = await asyncio.gather(*tasks, return_exceptions=True)

            # 统计结果
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            error_count = len(results) - success_count

            metrics.metadata.update(
                {
                    "success_count": success_count,
                    "error_count": error_count,
                    "success_rate": success_count / len(results) if results else 0,
                }
            )

            if error_count == 0:
                metrics.complete()
            else:
                # 如果有错误，记录但不标记为失败（部分成功）
                metrics.status = "partial_success"
                metrics.end_time = time.time()
                metrics.duration = metrics.end_time - metrics.start_time

            return results

        except asyncio.TimeoutError as e:
            metrics.status = "timeout"
            metrics.complete(error=e)
            raise

        except Exception as e:
            metrics.complete(error=e)

            self.exception_logger.log_exception(
                e,
                {
                    "operation_id": operation_id,
                    "operation_name": operation_name,
                    "task_count": len(tasks),
                    "monitoring": "concurrent_operations_monitor",
                },
            )

            raise
        finally:
            self._update_stats(metrics)

    def _register_metrics(self, metrics: AsyncOperationMetrics):
        """注册监控指标"""
        self._metrics[metrics.operation_id] = metrics
        self._stats["total_operations"] += 1
        self._stats["current_active_tasks"] = len(self._metrics)

    def _update_stats(self, metrics: AsyncOperationMetrics):
        """更新统计信息"""
        # 移动到历史记录
        self._metrics_history.append(metrics)
        self._metrics.pop(metrics.operation_id, None)

        # 限制历史记录大小
        if len(self._metrics_history) > self.max_metrics_history:
            self._metrics_history = self._metrics_history[-self.max_metrics_history :]

        # 更新统计
        if metrics.status == "completed":
            self._stats["completed_operations"] += 1
        elif metrics.status == "failed":
            self._stats["failed_operations"] += 1
        elif metrics.status == "timeout":
            self._stats["timeout_operations"] += 1

        # 更新平均时长
        completed_ops = [m for m in self._metrics_history if m.duration is not None]
        if completed_ops:
            total_duration = sum(m.duration for m in completed_ops)
            self._stats["avg_duration"] = total_duration / len(completed_ops)

        self._stats["current_active_tasks"] = len(self._metrics)

        # 检查告警阈值
        self._check_alerts()

    def _check_alerts(self):
        """检查告警阈值"""
        try:
            total_ops = self._stats["total_operations"]
            if total_ops < 10:  # 样本太少，不进行告警检查
                return

            # 错误率检查
            error_rate = self._stats["failed_operations"] / total_ops
            if error_rate > self.alert_thresholds["error_rate"]:
                self._trigger_alert(
                    "high_error_rate",
                    {
                        "current_rate": error_rate,
                        "threshold": self.alert_thresholds["error_rate"],
                        "failed_operations": self._stats["failed_operations"],
                        "total_operations": total_ops,
                    },
                )

            # 平均耗时检查
            avg_duration = self._stats["avg_duration"]
            if avg_duration > self.alert_thresholds["avg_duration"]:
                self._trigger_alert(
                    "high_avg_duration",
                    {
                        "current_duration": avg_duration,
                        "threshold": self.alert_thresholds["avg_duration"],
                    },
                )

            # 超时率检查
            timeout_rate = self._stats["timeout_operations"] / total_ops
            if timeout_rate > self.alert_thresholds["timeout_rate"]:
                self._trigger_alert(
                    "high_timeout_rate",
                    {
                        "current_rate": timeout_rate,
                        "threshold": self.alert_thresholds["timeout_rate"],
                        "timeout_operations": self._stats["timeout_operations"],
                    },
                )

        except Exception as e:
            self.logger.error(f"告警检查失败: {e}")

    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """触发告警"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "alert_type": alert_type,
            "severity": "warning",
            "data": data,
            "stats": self._stats.copy(),
        }

        self.logger.warning(f"异步监控告警: {alert_type} - {data}")

        # 调用注册的告警回调
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(alert))
                else:
                    callback(alert)
            except Exception as e:
                self.logger.error(f"告警回调执行失败: {e}")

    async def _cleanup_loop(self):
        """清理循环"""
        while self._monitoring:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_old_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"清理循环出错: {e}")

    async def _cleanup_old_metrics(self):
        """清理旧的监控指标"""
        try:
            current_time = time.time()
            cleanup_threshold = current_time - (self.cleanup_interval * 2)

            # 清理长时间未完成的操作
            stale_operations = []
            for op_id, metrics in list(self._metrics.items()):
                if metrics.start_time < cleanup_threshold:
                    metrics.status = "timeout"
                    metrics.end_time = current_time
                    metrics.duration = current_time - metrics.start_time
                    stale_operations.append(op_id)

            # 移除超时操作
            for op_id in stale_operations:
                metrics = self._metrics.pop(op_id, None)
                if metrics:
                    self._metrics_history.append(metrics)
                    self._stats["timeout_operations"] += 1

            if stale_operations:
                self.logger.warning(f"清理了 {len(stale_operations)} 个超时操作")

            # 限制历史记录大小
            if len(self._metrics_history) > self.max_metrics_history:
                removed_count = len(self._metrics_history) - self.max_metrics_history
                self._metrics_history = self._metrics_history[
                    -self.max_metrics_history :
                ]
                self.logger.debug(f"清理了 {removed_count} 条历史记录")

        except Exception as e:
            self.logger.error(f"清理旧指标失败: {e}")

    def add_alert_callback(self, callback: Callable):
        """添加告警回调"""
        self._alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable):
        """移除告警回调"""
        if callback in self._alert_callbacks:
            self._alert_callbacks.remove(callback)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取监控指标摘要"""
        return {
            "stats": self._stats.copy(),
            "active_operations": len(self._metrics),
            "history_size": len(self._metrics_history),
            "alert_thresholds": self.alert_thresholds.copy(),
            "monitoring_active": self._monitoring,
        }

    def get_recent_operations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取最近的操作记录"""
        recent = self._metrics_history[-limit:] if limit else self._metrics_history
        return [metrics.to_dict() for metrics in recent]

    def get_operation_details(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """获取特定操作的详细信息"""
        # 先查找活跃操作
        if operation_id in self._metrics:
            return self._metrics[operation_id].to_dict()

        # 再查找历史记录
        for metrics in reversed(self._metrics_history):
            if metrics.operation_id == operation_id:
                return metrics.to_dict()

        return None

    def export_metrics(self, format: str = "json") -> str:
        """导出监控指标"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_metrics_summary(),
            "recent_operations": self.get_recent_operations(),
            "active_operations": [m.to_dict() for m in self._metrics.values()],
        }

        if format.lower() == "json":
            return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的导出格式: {format}")


# 全局监控器实例
_global_monitor: Optional[AsyncTaskMonitor] = None


def get_async_monitor() -> AsyncTaskMonitor:
    """获取全局异步监控器实例"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = AsyncTaskMonitor()
    return _global_monitor


def monitor_async_operation(operation_name: str, metadata: Dict[str, Any] = None):
    """异步操作监控装饰器"""
    monitor = get_async_monitor()
    return monitor.monitor_task(operation_name, metadata)


async def start_monitoring():
    """启动全局监控"""
    monitor = get_async_monitor()
    await monitor.start()


async def stop_monitoring():
    """停止全局监控"""
    global _global_monitor
    if _global_monitor:
        await _global_monitor.stop()
        _global_monitor = None
