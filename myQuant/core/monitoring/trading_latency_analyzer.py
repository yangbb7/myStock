import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
import time
import threading
from collections import deque, defaultdict
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class LatencyType(Enum):
    ORDER_PLACEMENT = "order_placement"
    ORDER_EXECUTION = "order_execution"
    MARKET_DATA = "market_data"
    RISK_CHECK = "risk_check"
    POSITION_UPDATE = "position_update"
    TRADE_CONFIRMATION = "trade_confirmation"
    SETTLEMENT = "settlement"
    NETWORK = "network"
    SYSTEM_INTERNAL = "system_internal"
    END_TO_END = "end_to_end"

class LatencyCategory(Enum):
    ULTRA_LOW = "ultra_low"      # < 1ms
    LOW = "low"                  # 1-10ms
    MEDIUM = "medium"            # 10-100ms
    HIGH = "high"                # 100-1000ms
    CRITICAL = "critical"        # > 1000ms

class MeasurementPoint(Enum):
    ENTRY = "entry"
    PROCESSING = "processing"
    VALIDATION = "validation"
    EXECUTION = "execution"
    CONFIRMATION = "confirmation"
    EXIT = "exit"

@dataclass
class LatencyMeasurement:
    """延迟测量"""
    measurement_id: str
    latency_type: LatencyType
    timestamp: datetime
    start_time: float
    end_time: float
    duration_ms: float
    category: LatencyCategory
    component: str
    operation: str
    success: bool
    error_message: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LatencyBreakdown:
    """延迟分解"""
    total_latency_ms: float
    breakdown: Dict[str, float]
    critical_path: List[str]
    bottleneck_component: str
    bottleneck_percentage: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LatencyBenchmark:
    """延迟基准"""
    benchmark_id: str
    latency_type: LatencyType
    percentile_50: float
    percentile_95: float
    percentile_99: float
    percentile_99_9: float
    average: float
    min_latency: float
    max_latency: float
    std_deviation: float
    sample_count: int
    measurement_period: Tuple[datetime, datetime]
    target_latency: float
    sla_compliance: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LatencyAlert:
    """延迟告警"""
    alert_id: str
    timestamp: datetime
    latency_type: LatencyType
    component: str
    current_latency: float
    threshold_latency: float
    severity: str
    message: str
    consecutive_violations: int
    duration_minutes: float
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class TradingLatencyAnalyzer:
    """交易延迟分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 数据存储
        self.measurements = deque(maxlen=config.get('max_measurements', 100000))
        self.alerts = deque(maxlen=config.get('max_alerts', 1000))
        self.benchmarks: Dict[str, LatencyBenchmark] = {}
        
        # 实时监控
        self.active_measurements: Dict[str, Dict[str, Any]] = {}
        self.thresholds: Dict[LatencyType, Dict[str, float]] = {}
        
        # 统计数据
        self.statistics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.performance_trends: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # 运行状态
        self.is_running = False
        self.analysis_thread = None
        self.stop_event = threading.Event()
        
        # 回调函数
        self.alert_callbacks: List[callable] = []
        self.measurement_callbacks: List[callable] = []
        
        # 初始化
        self._initialize_thresholds()
        self._initialize_benchmarks()
    
    def _initialize_thresholds(self):
        """初始化阈值"""
        default_thresholds = {
            LatencyType.ORDER_PLACEMENT: {
                'warning': 50.0,
                'critical': 100.0,
                'target': 10.0
            },
            LatencyType.ORDER_EXECUTION: {
                'warning': 100.0,
                'critical': 500.0,
                'target': 50.0
            },
            LatencyType.MARKET_DATA: {
                'warning': 5.0,
                'critical': 20.0,
                'target': 1.0
            },
            LatencyType.RISK_CHECK: {
                'warning': 20.0,
                'critical': 100.0,
                'target': 5.0
            },
            LatencyType.POSITION_UPDATE: {
                'warning': 30.0,
                'critical': 150.0,
                'target': 10.0
            },
            LatencyType.END_TO_END: {
                'warning': 200.0,
                'critical': 1000.0,
                'target': 100.0
            }
        }
        
        # 加载用户配置
        custom_thresholds = self.config.get('latency_thresholds', {})
        
        for latency_type, thresholds in default_thresholds.items():
            if latency_type.value in custom_thresholds:
                thresholds.update(custom_thresholds[latency_type.value])
            self.thresholds[latency_type] = thresholds
    
    def _initialize_benchmarks(self):
        """初始化基准"""
        # 加载历史基准数据
        benchmark_file = Path(self.config.get('benchmark_file', 'latency_benchmarks.json'))
        
        if benchmark_file.exists():
            try:
                with open(benchmark_file, 'r') as f:
                    benchmark_data = json.load(f)
                
                for benchmark_id, data in benchmark_data.items():
                    self.benchmarks[benchmark_id] = LatencyBenchmark(**data)
                
                self.logger.info(f"Loaded {len(self.benchmarks)} latency benchmarks")
                
            except Exception as e:
                self.logger.error(f"Error loading benchmarks: {e}")
    
    def start(self):
        """启动延迟分析"""
        if self.is_running:
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # 启动分析线程
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        
        self.logger.info("Trading latency analyzer started")
    
    def stop(self):
        """停止延迟分析"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping trading latency analyzer...")
        self.stop_event.set()
        
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5)
        
        self.is_running = False
        self.logger.info("Trading latency analyzer stopped")
    
    def _analysis_loop(self):
        """分析主循环"""
        while not self.stop_event.is_set():
            try:
                # 更新统计数据
                self._update_statistics()
                
                # 检查告警
                self._check_latency_alerts()
                
                # 更新基准
                self._update_benchmarks()
                
                # 清理过期数据
                self._cleanup_expired_data()
                
                time.sleep(self.config.get('analysis_interval', 10))
                
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                time.sleep(1)
    
    def start_measurement(self, measurement_id: str, latency_type: LatencyType, 
                         component: str, operation: str, context: Dict[str, Any] = None) -> str:
        """开始延迟测量"""
        start_time = time.time()
        
        measurement_data = {
            'measurement_id': measurement_id,
            'latency_type': latency_type,
            'component': component,
            'operation': operation,
            'start_time': start_time,
            'context': context or {},
            'checkpoints': []
        }
        
        self.active_measurements[measurement_id] = measurement_data
        
        return measurement_id
    
    def add_checkpoint(self, measurement_id: str, checkpoint_name: str, 
                      checkpoint_time: Optional[float] = None):
        """添加检查点"""
        if measurement_id not in self.active_measurements:
            return
        
        if checkpoint_time is None:
            checkpoint_time = time.time()
        
        measurement = self.active_measurements[measurement_id]
        measurement['checkpoints'].append({
            'name': checkpoint_name,
            'time': checkpoint_time,
            'elapsed': (checkpoint_time - measurement['start_time']) * 1000
        })
    
    def end_measurement(self, measurement_id: str, success: bool = True, 
                       error_message: Optional[str] = None, 
                       tags: Dict[str, str] = None) -> Optional[LatencyMeasurement]:
        """结束延迟测量"""
        if measurement_id not in self.active_measurements:
            return None
        
        end_time = time.time()
        measurement_data = self.active_measurements.pop(measurement_id)
        
        duration_ms = (end_time - measurement_data['start_time']) * 1000
        category = self._categorize_latency(duration_ms)
        
        measurement = LatencyMeasurement(
            measurement_id=measurement_id,
            latency_type=measurement_data['latency_type'],
            timestamp=datetime.now(),
            start_time=measurement_data['start_time'],
            end_time=end_time,
            duration_ms=duration_ms,
            category=category,
            component=measurement_data['component'],
            operation=measurement_data['operation'],
            success=success,
            error_message=error_message,
            context=measurement_data['context'],
            tags=tags or {},
            metadata={
                'checkpoints': measurement_data['checkpoints']
            }
        )
        
        # 存储测量结果
        self.measurements.append(measurement)
        
        # 通知回调
        for callback in self.measurement_callbacks:
            try:
                callback(measurement)
            except Exception as e:
                self.logger.error(f"Error in measurement callback: {e}")
        
        return measurement
    
    def _categorize_latency(self, duration_ms: float) -> LatencyCategory:
        """分类延迟"""
        if duration_ms < 1:
            return LatencyCategory.ULTRA_LOW
        elif duration_ms < 10:
            return LatencyCategory.LOW
        elif duration_ms < 100:
            return LatencyCategory.MEDIUM
        elif duration_ms < 1000:
            return LatencyCategory.HIGH
        else:
            return LatencyCategory.CRITICAL
    
    def _update_statistics(self):
        """更新统计数据"""
        try:
            # 按类型分组计算统计数据
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(minutes=5)
            
            recent_measurements = [
                m for m in self.measurements 
                if m.timestamp >= cutoff_time
            ]
            
            # 按延迟类型分组
            grouped_measurements = defaultdict(list)
            for measurement in recent_measurements:
                grouped_measurements[measurement.latency_type].append(measurement)
            
            # 计算统计数据
            for latency_type, measurements in grouped_measurements.items():
                if not measurements:
                    continue
                
                durations = [m.duration_ms for m in measurements]
                
                stats = {
                    'count': len(measurements),
                    'mean': np.mean(durations),
                    'median': np.median(durations),
                    'std': np.std(durations),
                    'min': np.min(durations),
                    'max': np.max(durations),
                    'p95': np.percentile(durations, 95),
                    'p99': np.percentile(durations, 99),
                    'success_rate': sum(1 for m in measurements if m.success) / len(measurements) * 100,
                    'timestamp': current_time
                }
                
                self.statistics[latency_type.value] = stats
                
                # 添加到趋势数据
                self.performance_trends[latency_type.value].append({
                    'timestamp': current_time,
                    'mean_latency': stats['mean'],
                    'p95_latency': stats['p95'],
                    'success_rate': stats['success_rate']
                })
                
        except Exception as e:
            self.logger.error(f"Error updating statistics: {e}")
    
    def _check_latency_alerts(self):
        """检查延迟告警"""
        try:
            for latency_type, stats in self.statistics.items():
                if not stats:
                    continue
                
                latency_type_enum = LatencyType(latency_type)
                thresholds = self.thresholds.get(latency_type_enum, {})
                
                current_p95 = stats.get('p95', 0)
                warning_threshold = thresholds.get('warning', float('inf'))
                critical_threshold = thresholds.get('critical', float('inf'))
                
                # 检查是否超过阈值
                if current_p95 > critical_threshold:
                    self._create_alert(
                        latency_type_enum, 
                        'critical', 
                        current_p95, 
                        critical_threshold,
                        f"P95 latency {current_p95:.2f}ms exceeds critical threshold {critical_threshold}ms"
                    )
                elif current_p95 > warning_threshold:
                    self._create_alert(
                        latency_type_enum, 
                        'warning', 
                        current_p95, 
                        warning_threshold,
                        f"P95 latency {current_p95:.2f}ms exceeds warning threshold {warning_threshold}ms"
                    )
                
        except Exception as e:
            self.logger.error(f"Error checking latency alerts: {e}")
    
    def _create_alert(self, latency_type: LatencyType, severity: str, 
                     current_latency: float, threshold_latency: float, message: str):
        """创建告警"""
        # 检查是否已经存在相同的未解决告警
        existing_alert = None
        for alert in reversed(self.alerts):
            if (alert.latency_type == latency_type and 
                alert.severity == severity and 
                not alert.resolved):
                existing_alert = alert
                break
        
        if existing_alert:
            # 更新现有告警
            existing_alert.consecutive_violations += 1
            existing_alert.duration_minutes = (datetime.now() - existing_alert.timestamp).total_seconds() / 60
            existing_alert.current_latency = current_latency
        else:
            # 创建新告警
            alert = LatencyAlert(
                alert_id=f"{latency_type.value}_{severity}_{int(time.time())}",
                timestamp=datetime.now(),
                latency_type=latency_type,
                component='system',
                current_latency=current_latency,
                threshold_latency=threshold_latency,
                severity=severity,
                message=message,
                consecutive_violations=1,
                duration_minutes=0
            )
            
            self.alerts.append(alert)
            
            # 通知回调
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
            
            self.logger.warning(f"Latency alert: {message}")
    
    def _update_benchmarks(self):
        """更新基准"""
        try:
            # 每小时更新一次基准
            if len(self.measurements) < 100:
                return
            
            current_time = datetime.now()
            
            # 获取过去24小时的数据
            cutoff_time = current_time - timedelta(hours=24)
            historical_measurements = [
                m for m in self.measurements 
                if m.timestamp >= cutoff_time and m.success
            ]
            
            # 按类型分组
            grouped_measurements = defaultdict(list)
            for measurement in historical_measurements:
                grouped_measurements[measurement.latency_type].append(measurement)
            
            # 计算基准
            for latency_type, measurements in grouped_measurements.items():
                if len(measurements) < 10:
                    continue
                
                durations = [m.duration_ms for m in measurements]
                
                benchmark = LatencyBenchmark(
                    benchmark_id=f"{latency_type.value}_{current_time.strftime('%Y%m%d')}",
                    latency_type=latency_type,
                    percentile_50=np.percentile(durations, 50),
                    percentile_95=np.percentile(durations, 95),
                    percentile_99=np.percentile(durations, 99),
                    percentile_99_9=np.percentile(durations, 99.9),
                    average=np.mean(durations),
                    min_latency=np.min(durations),
                    max_latency=np.max(durations),
                    std_deviation=np.std(durations),
                    sample_count=len(durations),
                    measurement_period=(cutoff_time, current_time),
                    target_latency=self.thresholds.get(latency_type, {}).get('target', 0),
                    sla_compliance=self._calculate_sla_compliance(measurements, latency_type)
                )
                
                self.benchmarks[benchmark.benchmark_id] = benchmark
                
        except Exception as e:
            self.logger.error(f"Error updating benchmarks: {e}")
    
    def _calculate_sla_compliance(self, measurements: List[LatencyMeasurement], 
                                 latency_type: LatencyType) -> float:
        """计算SLA合规性"""
        if not measurements:
            return 0.0
        
        target_latency = self.thresholds.get(latency_type, {}).get('target', float('inf'))
        
        compliant_count = sum(1 for m in measurements if m.duration_ms <= target_latency)
        return (compliant_count / len(measurements)) * 100
    
    def _cleanup_expired_data(self):
        """清理过期数据"""
        try:
            # 清理超过24小时的告警
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            # 清理已解决的告警
            self.alerts = deque(
                [alert for alert in self.alerts 
                 if not alert.resolved or alert.timestamp >= cutoff_time],
                maxlen=self.alerts.maxlen
            )
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired data: {e}")
    
    def get_latency_breakdown(self, measurement_id: str) -> Optional[LatencyBreakdown]:
        """获取延迟分解"""
        measurement = None
        for m in self.measurements:
            if m.measurement_id == measurement_id:
                measurement = m
                break
        
        if not measurement or 'checkpoints' not in measurement.metadata:
            return None
        
        checkpoints = measurement.metadata['checkpoints']
        if not checkpoints:
            return None
        
        # 计算各阶段耗时
        breakdown = {}
        previous_time = 0
        
        for checkpoint in checkpoints:
            elapsed = checkpoint['elapsed']
            stage_duration = elapsed - previous_time
            breakdown[checkpoint['name']] = stage_duration
            previous_time = elapsed
        
        # 找到瓶颈
        bottleneck_component = max(breakdown.keys(), key=lambda k: breakdown[k])
        bottleneck_percentage = (breakdown[bottleneck_component] / measurement.duration_ms) * 100
        
        # 构建关键路径
        critical_path = sorted(breakdown.keys(), key=lambda k: breakdown[k], reverse=True)
        
        return LatencyBreakdown(
            total_latency_ms=measurement.duration_ms,
            breakdown=breakdown,
            critical_path=critical_path,
            bottleneck_component=bottleneck_component,
            bottleneck_percentage=bottleneck_percentage,
            timestamp=measurement.timestamp
        )
    
    def get_performance_report(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """获取性能报告"""
        if time_range:
            start_time, end_time = time_range
            filtered_measurements = [
                m for m in self.measurements 
                if start_time <= m.timestamp <= end_time
            ]
        else:
            filtered_measurements = list(self.measurements)
        
        if not filtered_measurements:
            return {}
        
        # 按类型分组
        grouped_measurements = defaultdict(list)
        for measurement in filtered_measurements:
            grouped_measurements[measurement.latency_type].append(measurement)
        
        report = {
            'report_period': {
                'start': min(m.timestamp for m in filtered_measurements).isoformat(),
                'end': max(m.timestamp for m in filtered_measurements).isoformat()
            },
            'summary': {
                'total_measurements': len(filtered_measurements),
                'successful_measurements': sum(1 for m in filtered_measurements if m.success),
                'failed_measurements': sum(1 for m in filtered_measurements if not m.success),
                'unique_components': len(set(m.component for m in filtered_measurements)),
                'latency_types': len(grouped_measurements)
            },
            'latency_analysis': {}
        }
        
        # 分析各延迟类型
        for latency_type, measurements in grouped_measurements.items():
            durations = [m.duration_ms for m in measurements if m.success]
            
            if not durations:
                continue
            
            analysis = {
                'measurement_count': len(measurements),
                'success_rate': sum(1 for m in measurements if m.success) / len(measurements) * 100,
                'latency_statistics': {
                    'mean': np.mean(durations),
                    'median': np.median(durations),
                    'std': np.std(durations),
                    'min': np.min(durations),
                    'max': np.max(durations),
                    'p50': np.percentile(durations, 50),
                    'p95': np.percentile(durations, 95),
                    'p99': np.percentile(durations, 99),
                    'p99_9': np.percentile(durations, 99.9)
                },
                'category_distribution': {
                    category.value: sum(1 for m in measurements if m.category == category)
                    for category in LatencyCategory
                },
                'component_breakdown': {}
            }
            
            # 按组件分析
            component_measurements = defaultdict(list)
            for measurement in measurements:
                component_measurements[measurement.component].append(measurement)
            
            for component, comp_measurements in component_measurements.items():
                comp_durations = [m.duration_ms for m in comp_measurements if m.success]
                if comp_durations:
                    analysis['component_breakdown'][component] = {
                        'count': len(comp_measurements),
                        'mean_latency': np.mean(comp_durations),
                        'p95_latency': np.percentile(comp_durations, 95),
                        'success_rate': sum(1 for m in comp_measurements if m.success) / len(comp_measurements) * 100
                    }
            
            report['latency_analysis'][latency_type.value] = analysis
        
        return report
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """获取实时指标"""
        return {
            'active_measurements': len(self.active_measurements),
            'total_measurements': len(self.measurements),
            'recent_statistics': dict(self.statistics),
            'alert_count': len([a for a in self.alerts if not a.resolved]),
            'analyzer_status': 'running' if self.is_running else 'stopped'
        }
    
    def get_alerts(self, severity: Optional[str] = None, 
                  resolved: Optional[bool] = None) -> List[LatencyAlert]:
        """获取告警"""
        alerts = list(self.alerts)
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str):
        """确认告警"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                break
    
    def resolve_alert(self, alert_id: str):
        """解决告警"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                break
    
    def add_alert_callback(self, callback: callable):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    def add_measurement_callback(self, callback: callable):
        """添加测量回调"""
        self.measurement_callbacks.append(callback)
    
    def export_measurements(self, file_path: str, 
                          time_range: Optional[Tuple[datetime, datetime]] = None):
        """导出测量数据"""
        try:
            if time_range:
                start_time, end_time = time_range
                filtered_measurements = [
                    m for m in self.measurements 
                    if start_time <= m.timestamp <= end_time
                ]
            else:
                filtered_measurements = list(self.measurements)
            
            # 转换为DataFrame
            data = []
            for measurement in filtered_measurements:
                data.append({
                    'measurement_id': measurement.measurement_id,
                    'timestamp': measurement.timestamp.isoformat(),
                    'latency_type': measurement.latency_type.value,
                    'duration_ms': measurement.duration_ms,
                    'category': measurement.category.value,
                    'component': measurement.component,
                    'operation': measurement.operation,
                    'success': measurement.success,
                    'error_message': measurement.error_message
                })
            
            df = pd.DataFrame(data)
            
            # 根据文件扩展名选择格式
            if file_path.endswith('.csv'):
                df.to_csv(file_path, index=False)
            elif file_path.endswith('.json'):
                df.to_json(file_path, orient='records', date_format='iso')
            elif file_path.endswith('.xlsx'):
                df.to_excel(file_path, index=False)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            self.logger.info(f"Exported {len(filtered_measurements)} measurements to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting measurements: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_measurements': len(self.measurements),
            'active_measurements': len(self.active_measurements),
            'total_alerts': len(self.alerts),
            'unresolved_alerts': len([a for a in self.alerts if not a.resolved]),
            'benchmarks': len(self.benchmarks),
            'analyzer_running': self.is_running,
            'configured_thresholds': len(self.thresholds)
        }
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'is_running') and self.is_running:
            self.stop()