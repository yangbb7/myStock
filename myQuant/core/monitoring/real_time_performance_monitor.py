import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
from collections import deque, defaultdict
import statistics
import warnings
warnings.filterwarnings('ignore')

class MetricType(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    TRADING_PERFORMANCE = "trading_performance"
    SYSTEM_HEALTH = "system_health"
    QUEUE_DEPTH = "queue_depth"
    CONNECTION_STATUS = "connection_status"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    NETWORK_IO = "network_io"
    DISK_IO = "disk_io"
    CUSTOM = "custom"

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MonitoringStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    ERROR = "error"
    INITIALIZING = "initializing"

class ThresholdType(Enum):
    ABSOLUTE = "absolute"
    PERCENTAGE = "percentage"
    PERCENTILE = "percentile"
    MOVING_AVERAGE = "moving_average"
    STANDARD_DEVIATION = "standard_deviation"

@dataclass
class MetricPoint:
    """性能指标数据点"""
    timestamp: datetime
    metric_name: str
    metric_type: MetricType
    value: float
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlertThreshold:
    """告警阈值"""
    threshold_id: str
    metric_name: str
    threshold_type: ThresholdType
    warning_value: float
    critical_value: float
    comparison_operator: str  # >, <, >=, <=, ==, !=
    time_window: int  # seconds
    min_data_points: int
    severity: AlertSeverity
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlertEvent:
    """告警事件"""
    alert_id: str
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold_value: float
    severity: AlertSeverity
    message: str
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemHealthMetrics:
    """系统健康指标"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io_bytes: int
    disk_io_bytes: int
    process_count: int
    thread_count: int
    open_files: int
    load_average: Tuple[float, float, float]
    uptime_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradingPerformanceMetrics:
    """交易性能指标"""
    timestamp: datetime
    order_latency_ms: float
    fill_rate: float
    slippage_bps: float
    market_impact_bps: float
    execution_shortfall_bps: float
    orders_per_second: float
    trades_per_second: float
    pnl_per_second: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win_loss_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class RealTimePerformanceMonitor:
    """实时性能监控系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.status = MonitoringStatus.INITIALIZING
        
        # 数据存储
        self.metric_buffer = deque(maxlen=config.get('buffer_size', 10000))
        self.alert_buffer = deque(maxlen=config.get('alert_buffer_size', 1000))
        self.thresholds: Dict[str, AlertThreshold] = {}
        
        # 监控线程
        self.monitor_threads: Dict[str, threading.Thread] = {}
        self.stop_event = threading.Event()
        
        # 性能统计
        self.performance_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.last_metrics: Dict[str, MetricPoint] = {}
        
        # 订阅者
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # 初始化配置
        self._initialize_monitoring()
    
    def _initialize_monitoring(self):
        """初始化监控配置"""
        try:
            # 加载阈值配置
            self._load_thresholds()
            
            # 初始化系统监控
            self._initialize_system_monitoring()
            
            # 初始化交易监控
            self._initialize_trading_monitoring()
            
            self.status = MonitoringStatus.ACTIVE
            self.logger.info("Performance monitoring initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring: {e}")
            self.status = MonitoringStatus.ERROR
            raise
    
    def _load_thresholds(self):
        """加载告警阈值配置"""
        default_thresholds = {
            "cpu_usage": AlertThreshold(
                threshold_id="cpu_usage_alert",
                metric_name="cpu_usage",
                threshold_type=ThresholdType.ABSOLUTE,
                warning_value=80.0,
                critical_value=95.0,
                comparison_operator=">=",
                time_window=60,
                min_data_points=3,
                severity=AlertSeverity.HIGH
            ),
            "memory_usage": AlertThreshold(
                threshold_id="memory_usage_alert",
                metric_name="memory_usage",
                threshold_type=ThresholdType.ABSOLUTE,
                warning_value=85.0,
                critical_value=95.0,
                comparison_operator=">=",
                time_window=30,
                min_data_points=2,
                severity=AlertSeverity.HIGH
            ),
            "order_latency": AlertThreshold(
                threshold_id="order_latency_alert",
                metric_name="order_latency_ms",
                threshold_type=ThresholdType.ABSOLUTE,
                warning_value=100.0,
                critical_value=500.0,
                comparison_operator=">=",
                time_window=10,
                min_data_points=5,
                severity=AlertSeverity.CRITICAL
            ),
            "error_rate": AlertThreshold(
                threshold_id="error_rate_alert",
                metric_name="error_rate",
                threshold_type=ThresholdType.PERCENTAGE,
                warning_value=5.0,
                critical_value=10.0,
                comparison_operator=">=",
                time_window=300,
                min_data_points=10,
                severity=AlertSeverity.HIGH
            ),
            "fill_rate": AlertThreshold(
                threshold_id="fill_rate_alert",
                metric_name="fill_rate",
                threshold_type=ThresholdType.ABSOLUTE,
                warning_value=90.0,
                critical_value=80.0,
                comparison_operator="<=",
                time_window=60,
                min_data_points=5,
                severity=AlertSeverity.MEDIUM
            )
        }
        
        # 加载用户自定义阈值
        custom_thresholds = self.config.get('alert_thresholds', {})
        for name, threshold_config in custom_thresholds.items():
            self.thresholds[name] = AlertThreshold(**threshold_config)
        
        # 添加默认阈值
        for name, threshold in default_thresholds.items():
            if name not in self.thresholds:
                self.thresholds[name] = threshold
    
    def _initialize_system_monitoring(self):
        """初始化系统监控"""
        def system_monitor():
            while not self.stop_event.is_set():
                try:
                    # 收集系统指标
                    metrics = self._collect_system_metrics()
                    
                    # 记录指标
                    for metric in metrics:
                        self.record_metric(metric)
                    
                    # 检查告警
                    self._check_alerts(metrics)
                    
                    time.sleep(self.config.get('system_monitor_interval', 5))
                    
                except Exception as e:
                    self.logger.error(f"System monitoring error: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=system_monitor, daemon=True)
        thread.start()
        self.monitor_threads['system'] = thread
    
    def _initialize_trading_monitoring(self):
        """初始化交易监控"""
        def trading_monitor():
            while not self.stop_event.is_set():
                try:
                    # 收集交易指标
                    metrics = self._collect_trading_metrics()
                    
                    # 记录指标
                    for metric in metrics:
                        self.record_metric(metric)
                    
                    # 检查告警
                    self._check_alerts(metrics)
                    
                    time.sleep(self.config.get('trading_monitor_interval', 1))
                    
                except Exception as e:
                    self.logger.error(f"Trading monitoring error: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=trading_monitor, daemon=True)
        thread.start()
        self.monitor_threads['trading'] = thread
    
    def _collect_system_metrics(self) -> List[MetricPoint]:
        """收集系统指标"""
        metrics = []
        now = datetime.now()
        
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(MetricPoint(
                timestamp=now,
                metric_name="cpu_usage",
                metric_type=MetricType.RESOURCE_USAGE,
                value=cpu_percent,
                unit="percent"
            ))
            
            # 内存使用率
            memory = psutil.virtual_memory()
            metrics.append(MetricPoint(
                timestamp=now,
                metric_name="memory_usage",
                metric_type=MetricType.RESOURCE_USAGE,
                value=memory.percent,
                unit="percent"
            ))
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(MetricPoint(
                timestamp=now,
                metric_name="disk_usage",
                metric_type=MetricType.RESOURCE_USAGE,
                value=disk_percent,
                unit="percent"
            ))
            
            # 网络IO
            net_io = psutil.net_io_counters()
            metrics.append(MetricPoint(
                timestamp=now,
                metric_name="network_io_bytes",
                metric_type=MetricType.NETWORK_IO,
                value=net_io.bytes_sent + net_io.bytes_recv,
                unit="bytes"
            ))
            
            # 磁盘IO
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics.append(MetricPoint(
                    timestamp=now,
                    metric_name="disk_io_bytes",
                    metric_type=MetricType.DISK_IO,
                    value=disk_io.read_bytes + disk_io.write_bytes,
                    unit="bytes"
                ))
            
            # 进程数
            process_count = len(psutil.pids())
            metrics.append(MetricPoint(
                timestamp=now,
                metric_name="process_count",
                metric_type=MetricType.SYSTEM_HEALTH,
                value=process_count,
                unit="count"
            ))
            
            # 系统负载
            load_avg = psutil.getloadavg()
            metrics.append(MetricPoint(
                timestamp=now,
                metric_name="load_average_1min",
                metric_type=MetricType.SYSTEM_HEALTH,
                value=load_avg[0],
                unit="ratio"
            ))
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def _collect_trading_metrics(self) -> List[MetricPoint]:
        """收集交易指标"""
        metrics = []
        now = datetime.now()
        
        try:
            # 模拟交易指标收集
            # 实际实现中应该从交易引擎获取真实数据
            
            # 订单延迟
            order_latency = np.random.exponential(50)  # 模拟延迟
            metrics.append(MetricPoint(
                timestamp=now,
                metric_name="order_latency_ms",
                metric_type=MetricType.LATENCY,
                value=order_latency,
                unit="milliseconds"
            ))
            
            # 成交率
            fill_rate = np.random.uniform(85, 99)  # 模拟成交率
            metrics.append(MetricPoint(
                timestamp=now,
                metric_name="fill_rate",
                metric_type=MetricType.TRADING_PERFORMANCE,
                value=fill_rate,
                unit="percent"
            ))
            
            # 滑点
            slippage = np.random.uniform(0.5, 5.0)  # 模拟滑点
            metrics.append(MetricPoint(
                timestamp=now,
                metric_name="slippage_bps",
                metric_type=MetricType.TRADING_PERFORMANCE,
                value=slippage,
                unit="bps"
            ))
            
            # 吞吐量
            orders_per_second = np.random.poisson(100)  # 模拟订单频率
            metrics.append(MetricPoint(
                timestamp=now,
                metric_name="orders_per_second",
                metric_type=MetricType.THROUGHPUT,
                value=orders_per_second,
                unit="ops"
            ))
            
            # 错误率
            error_rate = np.random.uniform(0, 10)  # 模拟错误率
            metrics.append(MetricPoint(
                timestamp=now,
                metric_name="error_rate",
                metric_type=MetricType.ERROR_RATE,
                value=error_rate,
                unit="percent"
            ))
            
        except Exception as e:
            self.logger.error(f"Error collecting trading metrics: {e}")
        
        return metrics
    
    def record_metric(self, metric: MetricPoint):
        """记录性能指标"""
        try:
            # 添加到缓冲区
            self.metric_buffer.append(metric)
            
            # 更新最新指标
            self.last_metrics[metric.metric_name] = metric
            
            # 更新统计信息
            self._update_statistics(metric)
            
            # 通知订阅者
            self._notify_subscribers('metric_recorded', metric)
            
        except Exception as e:
            self.logger.error(f"Error recording metric: {e}")
    
    def _update_statistics(self, metric: MetricPoint):
        """更新统计信息"""
        try:
            metric_name = metric.metric_name
            
            if metric_name not in self.performance_stats:
                self.performance_stats[metric_name] = {
                    'count': 0,
                    'values': deque(maxlen=1000),
                    'min': float('inf'),
                    'max': float('-inf'),
                    'sum': 0.0,
                    'last_updated': metric.timestamp
                }
            
            stats = self.performance_stats[metric_name]
            stats['count'] += 1
            stats['values'].append(metric.value)
            stats['min'] = min(stats['min'], metric.value)
            stats['max'] = max(stats['max'], metric.value)
            stats['sum'] += metric.value
            stats['last_updated'] = metric.timestamp
            
            # 计算统计值
            if len(stats['values']) > 0:
                stats['mean'] = statistics.mean(stats['values'])
                if len(stats['values']) > 1:
                    stats['std'] = statistics.stdev(stats['values'])
                    stats['p95'] = np.percentile(list(stats['values']), 95)
                    stats['p99'] = np.percentile(list(stats['values']), 99)
                
        except Exception as e:
            self.logger.error(f"Error updating statistics: {e}")
    
    def _check_alerts(self, metrics: List[MetricPoint]):
        """检查告警条件"""
        for metric in metrics:
            try:
                if metric.metric_name in self.thresholds:
                    threshold = self.thresholds[metric.metric_name]
                    
                    if not threshold.enabled:
                        continue
                    
                    # 检查是否触发告警
                    alert = self._evaluate_threshold(metric, threshold)
                    
                    if alert:
                        self.alert_buffer.append(alert)
                        self._notify_subscribers('alert_triggered', alert)
                        self.logger.warning(f"Alert triggered: {alert.message}")
                        
            except Exception as e:
                self.logger.error(f"Error checking alerts: {e}")
    
    def _evaluate_threshold(self, metric: MetricPoint, threshold: AlertThreshold) -> Optional[AlertEvent]:
        """评估阈值条件"""
        try:
            # 获取历史数据
            historical_values = self._get_historical_values(
                metric.metric_name, 
                threshold.time_window
            )
            
            if len(historical_values) < threshold.min_data_points:
                return None
            
            # 计算比较值
            if threshold.threshold_type == ThresholdType.ABSOLUTE:
                compare_value = metric.value
            elif threshold.threshold_type == ThresholdType.MOVING_AVERAGE:
                compare_value = statistics.mean(historical_values)
            elif threshold.threshold_type == ThresholdType.PERCENTILE:
                compare_value = np.percentile(historical_values, 95)
            else:
                compare_value = metric.value
            
            # 检查阈值
            is_warning = self._compare_values(
                compare_value, 
                threshold.warning_value, 
                threshold.comparison_operator
            )
            
            is_critical = self._compare_values(
                compare_value, 
                threshold.critical_value, 
                threshold.comparison_operator
            )
            
            if is_critical:
                severity = AlertSeverity.CRITICAL
                threshold_value = threshold.critical_value
            elif is_warning:
                severity = AlertSeverity.HIGH
                threshold_value = threshold.warning_value
            else:
                return None
            
            # 创建告警事件
            alert = AlertEvent(
                alert_id=f"{threshold.threshold_id}_{int(time.time())}",
                timestamp=metric.timestamp,
                metric_name=metric.metric_name,
                current_value=compare_value,
                threshold_value=threshold_value,
                severity=severity,
                message=f"{metric.metric_name} {threshold.comparison_operator} {threshold_value} (current: {compare_value:.2f})"
            )
            
            return alert
            
        except Exception as e:
            self.logger.error(f"Error evaluating threshold: {e}")
            return None
    
    def _compare_values(self, current: float, threshold: float, operator: str) -> bool:
        """比较数值"""
        operators = {
            '>': lambda x, y: x > y,
            '<': lambda x, y: x < y,
            '>=': lambda x, y: x >= y,
            '<=': lambda x, y: x <= y,
            '==': lambda x, y: x == y,
            '!=': lambda x, y: x != y
        }
        
        return operators.get(operator, lambda x, y: False)(current, threshold)
    
    def _get_historical_values(self, metric_name: str, time_window: int) -> List[float]:
        """获取历史数据"""
        cutoff_time = datetime.now() - timedelta(seconds=time_window)
        
        values = []
        for metric in reversed(self.metric_buffer):
            if metric.metric_name == metric_name and metric.timestamp >= cutoff_time:
                values.append(metric.value)
        
        return values
    
    def subscribe(self, event_type: str, callback: Callable):
        """订阅事件"""
        self.subscribers[event_type].append(callback)
    
    def _notify_subscribers(self, event_type: str, data: Any):
        """通知订阅者"""
        for callback in self.subscribers[event_type]:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error notifying subscriber: {e}")
    
    def get_metrics(self, metric_name: Optional[str] = None, 
                   time_range: Optional[Tuple[datetime, datetime]] = None) -> List[MetricPoint]:
        """获取指标数据"""
        metrics = list(self.metric_buffer)
        
        if metric_name:
            metrics = [m for m in metrics if m.metric_name == metric_name]
        
        if time_range:
            start_time, end_time = time_range
            metrics = [m for m in metrics if start_time <= m.timestamp <= end_time]
        
        return metrics
    
    def get_alerts(self, severity: Optional[AlertSeverity] = None, 
                   unresolved_only: bool = False) -> List[AlertEvent]:
        """获取告警事件"""
        alerts = list(self.alert_buffer)
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if unresolved_only:
            alerts = [a for a in alerts if not a.resolved]
        
        return alerts
    
    def get_statistics(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """获取统计信息"""
        if metric_name:
            return self.performance_stats.get(metric_name, {})
        
        return dict(self.performance_stats)
    
    def add_threshold(self, threshold: AlertThreshold):
        """添加告警阈值"""
        self.thresholds[threshold.metric_name] = threshold
    
    def remove_threshold(self, metric_name: str):
        """移除告警阈值"""
        if metric_name in self.thresholds:
            del self.thresholds[metric_name]
    
    def resolve_alert(self, alert_id: str):
        """解决告警"""
        for alert in self.alert_buffer:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                self._notify_subscribers('alert_resolved', alert)
                break
    
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        health = {
            'status': self.status.value,
            'active_monitors': len(self.monitor_threads),
            'total_metrics': len(self.metric_buffer),
            'active_alerts': len([a for a in self.alert_buffer if not a.resolved]),
            'last_update': datetime.now().isoformat()
        }
        
        # 添加关键指标
        for metric_name in ['cpu_usage', 'memory_usage', 'disk_usage']:
            if metric_name in self.last_metrics:
                health[metric_name] = self.last_metrics[metric_name].value
        
        return health
    
    def stop(self):
        """停止监控"""
        self.logger.info("Stopping performance monitoring...")
        self.stop_event.set()
        
        # 等待所有线程结束
        for thread in self.monitor_threads.values():
            thread.join(timeout=5)
        
        self.status = MonitoringStatus.INACTIVE
        self.logger.info("Performance monitoring stopped")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'stop_event') and not self.stop_event.is_set():
            self.stop()