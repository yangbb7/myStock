# -*- coding: utf-8 -*-
"""
Metrics - 系统性能监控和指标收集模块
"""

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import psutil


@dataclass
class MetricPoint:
    """指标数据点"""

    timestamp: datetime
    value: float
    tags: Dict[str, str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class TimeSeries:
    """时间序列数据"""

    def __init__(self, max_points: int = 1000):
        self.max_points = max_points
        self.points: deque = deque(maxlen=max_points)
        self._lock = threading.Lock()

    def add_point(
        self, value: float, timestamp: datetime = None, tags: Dict[str, str] = None
    ):
        """添加数据点"""
        if timestamp is None:
            timestamp = datetime.now()

        point = MetricPoint(timestamp=timestamp, value=value, tags=tags or {})

        with self._lock:
            self.points.append(point)

    def get_points(
        self, since: datetime = None, until: datetime = None
    ) -> List[MetricPoint]:
        """获取数据点"""
        with self._lock:
            points = list(self.points)

        if since or until:
            filtered_points = []
            for point in points:
                if since and point.timestamp < since:
                    continue
                if until and point.timestamp > until:
                    continue
                filtered_points.append(point)
            return filtered_points

        return points

    def get_latest(self) -> Optional[MetricPoint]:
        """获取最新数据点"""
        with self._lock:
            return self.points[-1] if self.points else None

    def get_average(self, since: datetime = None) -> float:
        """获取平均值"""
        points = self.get_points(since=since)
        if not points:
            return 0.0
        return sum(p.value for p in points) / len(points)

    def get_max(self, since: datetime = None) -> float:
        """获取最大值"""
        points = self.get_points(since=since)
        if not points:
            return 0.0
        return max(p.value for p in points)

    def get_min(self, since: datetime = None) -> float:
        """获取最小值"""
        points = self.get_points(since=since)
        if not points:
            return 0.0
        return min(p.value for p in points)


class Counter:
    """计数器"""

    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()

    def increment(self, amount: float = 1.0):
        """增加计数"""
        with self._lock:
            self.value += amount

    def decrement(self, amount: float = 1.0):
        """减少计数"""
        with self._lock:
            self.value -= amount

    def reset(self):
        """重置计数"""
        with self._lock:
            self.value = 0

    def get_value(self) -> float:
        """获取当前值"""
        with self._lock:
            return self.value


class Gauge:
    """仪表指标"""

    def __init__(self):
        self.value = 0.0
        self._lock = threading.Lock()

    def set_value(self, value: float):
        """设置值"""
        with self._lock:
            self.value = value

    def get_value(self) -> float:
        """获取值"""
        with self._lock:
            return self.value


class Histogram:
    """直方图"""

    def __init__(self, buckets: List[float] = None):
        if buckets is None:
            buckets = [
                0.005,
                0.01,
                0.025,
                0.05,
                0.075,
                0.1,
                0.25,
                0.5,
                0.75,
                1.0,
                2.5,
                5.0,
                7.5,
                10.0,
            ]

        self.buckets = sorted(buckets)
        self.bucket_counts = {bucket: 0 for bucket in self.buckets}
        self.bucket_counts[float("inf")] = 0  # 添加无穷大桶

        self.count = 0
        self.sum = 0.0
        self._lock = threading.Lock()

    def observe(self, value: float):
        """观察一个值"""
        with self._lock:
            self.count += 1
            self.sum += value

            # 增加对应桶的计数
            for bucket in self.buckets:
                if value <= bucket:
                    self.bucket_counts[bucket] += 1

            # 处理超出最大桶的情况
            if value > max(self.buckets):
                self.bucket_counts[float("inf")] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "count": self.count,
                "sum": self.sum,
                "average": self.sum / max(self.count, 1),
                "buckets": dict(self.bucket_counts),
            }


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}
        self.time_series: Dict[str, TimeSeries] = {}

        self._lock = threading.Lock()

        # 系统指标收集
        self.system_metrics_enabled = False
        self.system_metrics_thread = None
        self.system_metrics_interval = 10  # 秒

    def get_counter(self, name: str) -> Counter:
        """获取或创建计数器"""
        with self._lock:
            if name not in self.counters:
                self.counters[name] = Counter()
            return self.counters[name]

    def get_gauge(self, name: str) -> Gauge:
        """获取或创建仪表"""
        with self._lock:
            if name not in self.gauges:
                self.gauges[name] = Gauge()
            return self.gauges[name]

    def get_histogram(self, name: str, buckets: List[float] = None) -> Histogram:
        """获取或创建直方图"""
        with self._lock:
            if name not in self.histograms:
                self.histograms[name] = Histogram(buckets)
            return self.histograms[name]

    def get_time_series(self, name: str, max_points: int = 1000) -> TimeSeries:
        """获取或创建时间序列"""
        with self._lock:
            if name not in self.time_series:
                self.time_series[name] = TimeSeries(max_points)
            return self.time_series[name]

    def increment_counter(
        self, name: str, amount: float = 1.0, tags: Dict[str, str] = None
    ):
        """增加计数器"""
        counter = self.get_counter(name)
        counter.increment(amount)

        # 同时记录到时间序列
        ts = self.get_time_series(f"{name}_ts")
        ts.add_point(counter.get_value(), tags=tags)

    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """设置仪表值"""
        gauge = self.get_gauge(name)
        gauge.set_value(value)

        # 同时记录到时间序列
        ts = self.get_time_series(f"{name}_ts")
        ts.add_point(value, tags=tags)

    def observe_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """观察直方图值"""
        histogram = self.get_histogram(name)
        histogram.observe(value)

        # 记录到时间序列
        ts = self.get_time_series(f"{name}_ts")
        ts.add_point(value, tags=tags)

    def record_time_series(self, name: str, value: float, tags: Dict[str, str] = None):
        """记录时间序列数据"""
        ts = self.get_time_series(name)
        ts.add_point(value, tags=tags)

    def start_system_metrics_collection(self, interval: int = 10):
        """开始系统指标收集"""
        if self.system_metrics_enabled:
            return

        self.system_metrics_enabled = True
        self.system_metrics_interval = interval
        self.system_metrics_thread = threading.Thread(
            target=self._collect_system_metrics
        )
        self.system_metrics_thread.daemon = True
        self.system_metrics_thread.start()

    def stop_system_metrics_collection(self):
        """停止系统指标收集"""
        self.system_metrics_enabled = False
        if self.system_metrics_thread:
            self.system_metrics_thread.join(timeout=5.0)

    def _collect_system_metrics(self):
        """收集系统指标"""
        while self.system_metrics_enabled:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                self.set_gauge("system.cpu.usage_percent", cpu_percent)

                # 内存使用
                memory = psutil.virtual_memory()
                self.set_gauge("system.memory.usage_percent", memory.percent)
                self.set_gauge(
                    "system.memory.available_mb", memory.available / 1024 / 1024
                )
                self.set_gauge("system.memory.used_mb", memory.used / 1024 / 1024)

                # 磁盘使用
                disk = psutil.disk_usage("/")
                self.set_gauge("system.disk.usage_percent", disk.percent)
                self.set_gauge("system.disk.free_gb", disk.free / 1024 / 1024 / 1024)

                # 网络IO
                net_io = psutil.net_io_counters()
                self.set_gauge("system.network.bytes_sent", net_io.bytes_sent)
                self.set_gauge("system.network.bytes_recv", net_io.bytes_recv)

                time.sleep(self.system_metrics_interval)

            except Exception as e:
                print(f"Error collecting system metrics: {e}")
                time.sleep(self.system_metrics_interval)

    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        with self._lock:
            metrics = {
                "counters": {
                    name: counter.get_value() for name, counter in self.counters.items()
                },
                "gauges": {
                    name: gauge.get_value() for name, gauge in self.gauges.items()
                },
                "histograms": {
                    name: hist.get_statistics()
                    for name, hist in self.histograms.items()
                },
                "time_series": {
                    name: {
                        "latest": ts.get_latest().value if ts.get_latest() else None,
                        "count": len(ts.points),
                        "average_1m": ts.get_average(
                            since=datetime.now() - timedelta(minutes=1)
                        ),
                        "average_5m": ts.get_average(
                            since=datetime.now() - timedelta(minutes=5)
                        ),
                        "max_5m": ts.get_max(
                            since=datetime.now() - timedelta(minutes=5)
                        ),
                        "min_5m": ts.get_min(
                            since=datetime.now() - timedelta(minutes=5)
                        ),
                    }
                    for name, ts in self.time_series.items()
                },
            }

            return metrics

    def export_to_json(self) -> str:
        """导出为JSON格式"""
        metrics = self.get_all_metrics()
        return json.dumps(metrics, default=str, indent=2)


class PerformanceTimer:
    """性能计时器（上下文管理器）"""

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        metric_name: str,
        tags: Dict[str, str] = None,
    ):
        self.metrics_collector = metrics_collector
        self.metric_name = metric_name
        self.tags = tags or {}
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics_collector.observe_histogram(
                self.metric_name, duration, self.tags
            )


# 全局指标收集器
_global_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器"""
    global _global_metrics_collector

    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()

    return _global_metrics_collector


def timer(metric_name: str, tags: Dict[str, str] = None) -> PerformanceTimer:
    """创建性能计时器"""
    return PerformanceTimer(get_metrics_collector(), metric_name, tags)


def increment(metric_name: str, amount: float = 1.0, tags: Dict[str, str] = None):
    """增加计数器（便捷函数）"""
    get_metrics_collector().increment_counter(metric_name, amount, tags)


def gauge(metric_name: str, value: float, tags: Dict[str, str] = None):
    """设置仪表值（便捷函数）"""
    get_metrics_collector().set_gauge(metric_name, value, tags)


def histogram(metric_name: str, value: float, tags: Dict[str, str] = None):
    """观察直方图值（便捷函数）"""
    get_metrics_collector().observe_histogram(metric_name, value, tags)


def time_series(metric_name: str, value: float, tags: Dict[str, str] = None):
    """记录时间序列（便捷函数）"""
    get_metrics_collector().record_time_series(metric_name, value, tags)
