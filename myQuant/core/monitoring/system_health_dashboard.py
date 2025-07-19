import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import json
import threading
import time
from pathlib import Path
from collections import defaultdict, deque
import psutil
import warnings
warnings.filterwarnings('ignore')

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"

class ComponentType(Enum):
    SYSTEM = "system"
    TRADING_ENGINE = "trading_engine"
    DATA_FEED = "data_feed"
    RISK_MANAGER = "risk_manager"
    ORDER_MANAGER = "order_manager"
    PORTFOLIO_MANAGER = "portfolio_manager"
    DATABASE = "database"
    NETWORK = "network"
    STORAGE = "storage"
    EXTERNAL_API = "external_api"
    CUSTOM = "custom"

class MetricCategory(Enum):
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"
    RELIABILITY = "reliability"
    SECURITY = "security"
    CAPACITY = "capacity"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"

@dataclass
class HealthMetric:
    """健康指标"""
    metric_id: str
    metric_name: str
    category: MetricCategory
    value: float
    unit: str
    timestamp: datetime
    status: HealthStatus
    threshold_warning: float
    threshold_critical: float
    description: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComponentHealth:
    """组件健康状态"""
    component_id: str
    component_name: str
    component_type: ComponentType
    status: HealthStatus
    last_updated: datetime
    uptime: float
    metrics: List[HealthMetric]
    dependencies: List[str] = field(default_factory=list)
    error_count: int = 0
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemHealthSnapshot:
    """系统健康快照"""
    timestamp: datetime
    overall_status: HealthStatus
    component_count: int
    healthy_components: int
    warning_components: int
    critical_components: int
    total_metrics: int
    system_uptime: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: float
    active_connections: int
    error_rate: float
    response_time: float
    throughput: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthAlert:
    """健康告警"""
    alert_id: str
    timestamp: datetime
    component_id: str
    metric_name: str
    severity: HealthStatus
    message: str
    current_value: float
    threshold_value: float
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class SystemHealthDashboard:
    """系统健康监控面板"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 组件管理
        self.components: Dict[str, ComponentHealth] = {}
        self.component_configs: Dict[str, Dict[str, Any]] = {}
        
        # 数据存储
        self.health_history = deque(maxlen=config.get('history_size', 1000))
        self.alert_history = deque(maxlen=config.get('alert_history_size', 500))
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # 运行状态
        self.is_running = False
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # 回调函数
        self.alert_callbacks: List[callable] = []
        self.status_change_callbacks: List[callable] = []
        
        # 初始化
        self._initialize_components()
        self._load_configurations()
    
    def _initialize_components(self):
        """初始化组件"""
        try:
            # 默认系统组件
            default_components = [
                {
                    'component_id': 'system_cpu',
                    'component_name': 'CPU Monitor',
                    'component_type': ComponentType.SYSTEM,
                    'metrics': [
                        {
                            'metric_id': 'cpu_usage',
                            'metric_name': 'CPU Usage',
                            'category': MetricCategory.PERFORMANCE,
                            'unit': 'percent',
                            'threshold_warning': 80.0,
                            'threshold_critical': 95.0,
                            'description': 'CPU utilization percentage'
                        }
                    ]
                },
                {
                    'component_id': 'system_memory',
                    'component_name': 'Memory Monitor',
                    'component_type': ComponentType.SYSTEM,
                    'metrics': [
                        {
                            'metric_id': 'memory_usage',
                            'metric_name': 'Memory Usage',
                            'category': MetricCategory.PERFORMANCE,
                            'unit': 'percent',
                            'threshold_warning': 85.0,
                            'threshold_critical': 95.0,
                            'description': 'Memory utilization percentage'
                        }
                    ]
                },
                {
                    'component_id': 'system_disk',
                    'component_name': 'Disk Monitor',
                    'component_type': ComponentType.STORAGE,
                    'metrics': [
                        {
                            'metric_id': 'disk_usage',
                            'metric_name': 'Disk Usage',
                            'category': MetricCategory.CAPACITY,
                            'unit': 'percent',
                            'threshold_warning': 80.0,
                            'threshold_critical': 90.0,
                            'description': 'Disk space utilization percentage'
                        }
                    ]
                },
                {
                    'component_id': 'trading_engine',
                    'component_name': 'Trading Engine',
                    'component_type': ComponentType.TRADING_ENGINE,
                    'metrics': [
                        {
                            'metric_id': 'order_latency',
                            'metric_name': 'Order Latency',
                            'category': MetricCategory.LATENCY,
                            'unit': 'milliseconds',
                            'threshold_warning': 100.0,
                            'threshold_critical': 500.0,
                            'description': 'Average order processing latency'
                        },
                        {
                            'metric_id': 'order_success_rate',
                            'metric_name': 'Order Success Rate',
                            'category': MetricCategory.RELIABILITY,
                            'unit': 'percent',
                            'threshold_warning': 95.0,
                            'threshold_critical': 90.0,
                            'description': 'Percentage of successful orders'
                        }
                    ]
                },
                {
                    'component_id': 'data_feed',
                    'component_name': 'Market Data Feed',
                    'component_type': ComponentType.DATA_FEED,
                    'metrics': [
                        {
                            'metric_id': 'data_latency',
                            'metric_name': 'Data Latency',
                            'category': MetricCategory.LATENCY,
                            'unit': 'milliseconds',
                            'threshold_warning': 50.0,
                            'threshold_critical': 200.0,
                            'description': 'Market data feed latency'
                        },
                        {
                            'metric_id': 'data_quality',
                            'metric_name': 'Data Quality',
                            'category': MetricCategory.RELIABILITY,
                            'unit': 'percent',
                            'threshold_warning': 98.0,
                            'threshold_critical': 95.0,
                            'description': 'Data quality score'
                        }
                    ]
                }
            ]
            
            # 创建组件
            for component_config in default_components:
                self._create_component(component_config)
            
            # 加载用户自定义组件
            custom_components = self.config.get('custom_components', [])
            for component_config in custom_components:
                self._create_component(component_config)
            
            self.logger.info(f"Initialized {len(self.components)} health monitoring components")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _create_component(self, config: Dict[str, Any]):
        """创建组件"""
        try:
            component_id = config['component_id']
            
            # 创建指标
            metrics = []
            for metric_config in config.get('metrics', []):
                metric = HealthMetric(
                    metric_id=metric_config['metric_id'],
                    metric_name=metric_config['metric_name'],
                    category=MetricCategory(metric_config['category']),
                    value=0.0,
                    unit=metric_config['unit'],
                    timestamp=datetime.now(),
                    status=HealthStatus.UNKNOWN,
                    threshold_warning=metric_config['threshold_warning'],
                    threshold_critical=metric_config['threshold_critical'],
                    description=metric_config['description']
                )
                metrics.append(metric)
            
            # 创建组件
            component = ComponentHealth(
                component_id=component_id,
                component_name=config['component_name'],
                component_type=ComponentType(config['component_type']),
                status=HealthStatus.UNKNOWN,
                last_updated=datetime.now(),
                uptime=0.0,
                metrics=metrics,
                dependencies=config.get('dependencies', [])
            )
            
            self.components[component_id] = component
            self.component_configs[component_id] = config
            
        except Exception as e:
            self.logger.error(f"Error creating component: {e}")
    
    def _load_configurations(self):
        """加载配置"""
        try:
            # 加载告警配置
            alert_config = self.config.get('alert_config', {})
            
            # 加载监控配置
            monitoring_config = self.config.get('monitoring_config', {})
            self.monitoring_interval = monitoring_config.get('interval', 5)
            
            # 加载仪表盘配置
            dashboard_config = self.config.get('dashboard_config', {})
            
        except Exception as e:
            self.logger.error(f"Error loading configurations: {e}")
    
    def start(self):
        """启动健康监控"""
        if self.is_running:
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # 启动监控线程
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("System health monitoring started")
    
    def stop(self):
        """停止健康监控"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping system health monitoring...")
        self.stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.is_running = False
        self.logger.info("System health monitoring stopped")
    
    def _monitoring_loop(self):
        """监控主循环"""
        start_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                # 更新所有组件状态
                for component_id in self.components:
                    self._update_component_health(component_id)
                
                # 创建系统健康快照
                snapshot = self._create_health_snapshot()
                self.health_history.append(snapshot)
                
                # 检查告警
                self._check_alerts()
                
                # 清理过期数据
                self._cleanup_old_data()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def _update_component_health(self, component_id: str):
        """更新组件健康状态"""
        try:
            component = self.components[component_id]
            config = self.component_configs[component_id]
            
            # 更新指标
            for metric in component.metrics:
                new_value = self._collect_metric_value(component_id, metric.metric_id)
                
                if new_value is not None:
                    metric.value = new_value
                    metric.timestamp = datetime.now()
                    
                    # 更新状态
                    if new_value >= metric.threshold_critical:
                        metric.status = HealthStatus.CRITICAL
                    elif new_value >= metric.threshold_warning:
                        metric.status = HealthStatus.WARNING
                    else:
                        metric.status = HealthStatus.HEALTHY
                    
                    # 存储到缓冲区
                    self.metrics_buffer[f"{component_id}_{metric.metric_id}"].append({
                        'timestamp': metric.timestamp,
                        'value': metric.value,
                        'status': metric.status
                    })
            
            # 更新组件整体状态
            component_statuses = [metric.status for metric in component.metrics]
            
            if HealthStatus.CRITICAL in component_statuses:
                component.status = HealthStatus.CRITICAL
            elif HealthStatus.WARNING in component_statuses:
                component.status = HealthStatus.WARNING
            else:
                component.status = HealthStatus.HEALTHY
            
            component.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating component health for {component_id}: {e}")
            self.components[component_id].status = HealthStatus.UNKNOWN
    
    def _collect_metric_value(self, component_id: str, metric_id: str) -> Optional[float]:
        """收集指标值"""
        try:
            # 系统指标
            if metric_id == 'cpu_usage':
                return psutil.cpu_percent(interval=1)
            elif metric_id == 'memory_usage':
                return psutil.virtual_memory().percent
            elif metric_id == 'disk_usage':
                return psutil.disk_usage('/').percent
            
            # 网络指标
            elif metric_id == 'network_io':
                net_io = psutil.net_io_counters()
                return net_io.bytes_sent + net_io.bytes_recv
            
            # 交易引擎指标（模拟）
            elif metric_id == 'order_latency':
                return np.random.exponential(50)  # 模拟延迟
            elif metric_id == 'order_success_rate':
                return np.random.uniform(95, 100)  # 模拟成功率
            
            # 数据源指标（模拟）
            elif metric_id == 'data_latency':
                return np.random.exponential(20)  # 模拟数据延迟
            elif metric_id == 'data_quality':
                return np.random.uniform(98, 100)  # 模拟数据质量
            
            # 自定义指标
            else:
                return self._collect_custom_metric(component_id, metric_id)
                
        except Exception as e:
            self.logger.error(f"Error collecting metric {metric_id}: {e}")
            return None
    
    def _collect_custom_metric(self, component_id: str, metric_id: str) -> Optional[float]:
        """收集自定义指标"""
        # 这里可以实现自定义指标收集逻辑
        # 例如从外部API获取数据，查询数据库等
        return None
    
    def _create_health_snapshot(self) -> SystemHealthSnapshot:
        """创建系统健康快照"""
        now = datetime.now()
        
        # 统计组件状态
        component_count = len(self.components)
        healthy_components = sum(1 for c in self.components.values() if c.status == HealthStatus.HEALTHY)
        warning_components = sum(1 for c in self.components.values() if c.status == HealthStatus.WARNING)
        critical_components = sum(1 for c in self.components.values() if c.status == HealthStatus.CRITICAL)
        
        # 确定整体状态
        if critical_components > 0:
            overall_status = HealthStatus.CRITICAL
        elif warning_components > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # 收集系统指标
        total_metrics = sum(len(c.metrics) for c in self.components.values())
        
        # 获取系统资源使用率
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        # 网络IO
        net_io = psutil.net_io_counters()
        network_io = net_io.bytes_sent + net_io.bytes_recv
        
        # 连接数
        active_connections = len(psutil.net_connections())
        
        # 模拟其他指标
        error_rate = np.random.uniform(0, 5)
        response_time = np.random.exponential(100)
        throughput = np.random.poisson(1000)
        
        snapshot = SystemHealthSnapshot(
            timestamp=now,
            overall_status=overall_status,
            component_count=component_count,
            healthy_components=healthy_components,
            warning_components=warning_components,
            critical_components=critical_components,
            total_metrics=total_metrics,
            system_uptime=time.time() - psutil.boot_time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            active_connections=active_connections,
            error_rate=error_rate,
            response_time=response_time,
            throughput=throughput
        )
        
        return snapshot
    
    def _check_alerts(self):
        """检查告警"""
        try:
            for component in self.components.values():
                for metric in component.metrics:
                    if metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                        # 检查是否需要发送告警
                        if self._should_send_alert(component.component_id, metric.metric_id, metric.status):
                            alert = self._create_alert(component, metric)
                            self.alert_history.append(alert)
                            
                            # 调用告警回调
                            for callback in self.alert_callbacks:
                                try:
                                    callback(alert)
                                except Exception as e:
                                    self.logger.error(f"Error in alert callback: {e}")
                            
                            self.logger.warning(f"Health alert: {alert.message}")
        
        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")
    
    def _should_send_alert(self, component_id: str, metric_id: str, status: HealthStatus) -> bool:
        """判断是否应该发送告警"""
        # 检查是否已经有相同的未解决告警
        for alert in reversed(self.alert_history):
            if (alert.component_id == component_id and 
                alert.metric_name == metric_id and 
                not alert.resolved):
                return False
        
        return True
    
    def _create_alert(self, component: ComponentHealth, metric: HealthMetric) -> HealthAlert:
        """创建告警"""
        threshold_value = metric.threshold_critical if metric.status == HealthStatus.CRITICAL else metric.threshold_warning
        
        alert = HealthAlert(
            alert_id=f"{component.component_id}_{metric.metric_id}_{int(time.time())}",
            timestamp=datetime.now(),
            component_id=component.component_id,
            metric_name=metric.metric_name,
            severity=metric.status,
            message=f"{component.component_name} - {metric.metric_name}: {metric.value:.2f}{metric.unit} (threshold: {threshold_value}{metric.unit})",
            current_value=metric.value,
            threshold_value=threshold_value
        )
        
        return alert
    
    def _cleanup_old_data(self):
        """清理过期数据"""
        try:
            # 清理指标缓冲区中的过期数据
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for metric_key, buffer in self.metrics_buffer.items():
                # 移除过期数据
                while buffer and buffer[0]['timestamp'] < cutoff_time:
                    buffer.popleft()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def add_alert_callback(self, callback: callable):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    def add_status_change_callback(self, callback: callable):
        """添加状态变化回调"""
        self.status_change_callbacks.append(callback)
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        if not self.health_history:
            return {}
        
        latest_snapshot = self.health_history[-1]
        
        return {
            'overall_status': latest_snapshot.overall_status.value,
            'timestamp': latest_snapshot.timestamp.isoformat(),
            'components': {
                component_id: {
                    'status': component.status.value,
                    'last_updated': component.last_updated.isoformat(),
                    'metrics': [
                        {
                            'name': metric.metric_name,
                            'value': metric.value,
                            'unit': metric.unit,
                            'status': metric.status.value
                        }
                        for metric in component.metrics
                    ]
                }
                for component_id, component in self.components.items()
            },
            'system_metrics': {
                'cpu_usage': latest_snapshot.cpu_usage,
                'memory_usage': latest_snapshot.memory_usage,
                'disk_usage': latest_snapshot.disk_usage,
                'network_io': latest_snapshot.network_io,
                'active_connections': latest_snapshot.active_connections,
                'error_rate': latest_snapshot.error_rate,
                'response_time': latest_snapshot.response_time,
                'throughput': latest_snapshot.throughput
            }
        }
    
    def get_historical_data(self, component_id: str, metric_id: str, 
                          time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """获取历史数据"""
        metric_key = f"{component_id}_{metric_id}"
        
        if metric_key not in self.metrics_buffer:
            return []
        
        data = list(self.metrics_buffer[metric_key])
        
        if time_range:
            start_time, end_time = time_range
            data = [d for d in data if start_time <= d['timestamp'] <= end_time]
        
        return data
    
    def get_alerts(self, component_id: Optional[str] = None, 
                  unresolved_only: bool = False) -> List[HealthAlert]:
        """获取告警"""
        alerts = list(self.alert_history)
        
        if component_id:
            alerts = [a for a in alerts if a.component_id == component_id]
        
        if unresolved_only:
            alerts = [a for a in alerts if not a.resolved]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str):
        """确认告警"""
        for alert in self.alert_history:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                break
    
    def resolve_alert(self, alert_id: str):
        """解决告警"""
        for alert in self.alert_history:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                break
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表盘数据"""
        current_status = self.get_current_status()
        
        # 获取最近的健康快照
        recent_snapshots = list(self.health_history)[-60:]  # 最近60个快照
        
        # 计算趋势
        trends = {}
        if len(recent_snapshots) > 1:
            for metric in ['cpu_usage', 'memory_usage', 'disk_usage', 'error_rate']:
                values = [getattr(snapshot, metric) for snapshot in recent_snapshots]
                trend = 'up' if values[-1] > values[0] else 'down'
                trends[metric] = trend
        
        # 获取未解决的告警
        unresolved_alerts = self.get_alerts(unresolved_only=True)
        
        return {
            'current_status': current_status,
            'trends': trends,
            'unresolved_alerts': len(unresolved_alerts),
            'alert_summary': {
                'critical': len([a for a in unresolved_alerts if a.severity == HealthStatus.CRITICAL]),
                'warning': len([a for a in unresolved_alerts if a.severity == HealthStatus.WARNING])
            },
            'system_performance': {
                'uptime': recent_snapshots[-1].system_uptime if recent_snapshots else 0,
                'total_components': len(self.components),
                'healthy_components': len([c for c in self.components.values() if c.status == HealthStatus.HEALTHY]),
                'monitoring_active': self.is_running
            }
        }
    
    def export_health_report(self, file_path: str, time_range: Optional[Tuple[datetime, datetime]] = None):
        """导出健康报告"""
        try:
            # 获取数据
            snapshots = list(self.health_history)
            
            if time_range:
                start_time, end_time = time_range
                snapshots = [s for s in snapshots if start_time <= s.timestamp <= end_time]
            
            # 创建报告
            report = {
                'generated_at': datetime.now().isoformat(),
                'time_range': {
                    'start': snapshots[0].timestamp.isoformat() if snapshots else None,
                    'end': snapshots[-1].timestamp.isoformat() if snapshots else None
                },
                'summary': {
                    'total_snapshots': len(snapshots),
                    'components_monitored': len(self.components),
                    'alerts_generated': len(self.alert_history)
                },
                'components': {
                    component_id: {
                        'name': component.component_name,
                        'type': component.component_type.value,
                        'current_status': component.status.value,
                        'last_updated': component.last_updated.isoformat(),
                        'metrics': [
                            {
                                'name': metric.metric_name,
                                'current_value': metric.value,
                                'unit': metric.unit,
                                'status': metric.status.value,
                                'thresholds': {
                                    'warning': metric.threshold_warning,
                                    'critical': metric.threshold_critical
                                }
                            }
                            for metric in component.metrics
                        ]
                    }
                    for component_id, component in self.components.items()
                },
                'alerts': [
                    {
                        'id': alert.alert_id,
                        'timestamp': alert.timestamp.isoformat(),
                        'component': alert.component_id,
                        'metric': alert.metric_name,
                        'severity': alert.severity.value,
                        'message': alert.message,
                        'acknowledged': alert.acknowledged,
                        'resolved': alert.resolved,
                        'resolution_time': alert.resolution_time.isoformat() if alert.resolution_time else None
                    }
                    for alert in self.alert_history
                ]
            }
            
            # 保存报告
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Health report exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting health report: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_components': len(self.components),
            'monitoring_active': self.is_running,
            'health_snapshots': len(self.health_history),
            'total_alerts': len(self.alert_history),
            'unresolved_alerts': len([a for a in self.alert_history if not a.resolved]),
            'component_status_distribution': {
                'healthy': len([c for c in self.components.values() if c.status == HealthStatus.HEALTHY]),
                'warning': len([c for c in self.components.values() if c.status == HealthStatus.WARNING]),
                'critical': len([c for c in self.components.values() if c.status == HealthStatus.CRITICAL]),
                'unknown': len([c for c in self.components.values() if c.status == HealthStatus.UNKNOWN])
            }
        }
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'is_running') and self.is_running:
            self.stop()