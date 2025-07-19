import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import psutil
import threading
import time
from collections import deque, defaultdict
import json
from pathlib import Path
import gc
import sys
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    THREAD = "thread"
    CONNECTION = "connection"
    CACHE = "cache"
    QUEUE = "queue"
    CUSTOM = "custom"

class OptimizationStrategy(Enum):
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"
    CUSTOM = "custom"

class ResourceStatus(Enum):
    OPTIMAL = "optimal"
    UNDERUTILIZED = "underutilized"
    OVERUTILIZED = "overutilized"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class OptimizationAction(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    REBALANCE = "rebalance"
    CACHE_CLEAR = "cache_clear"
    GARBAGE_COLLECT = "garbage_collect"
    THREAD_POOL_RESIZE = "thread_pool_resize"
    CONNECTION_POOL_RESIZE = "connection_pool_resize"
    QUEUE_RESIZE = "queue_resize"
    PRIORITY_ADJUSTMENT = "priority_adjustment"
    CUSTOM_ACTION = "custom_action"

@dataclass
class ResourceMetrics:
    """资源指标"""
    resource_type: ResourceType
    timestamp: datetime
    current_usage: float
    maximum_capacity: float
    utilization_percentage: float
    available_capacity: float
    peak_usage: float
    average_usage: float
    trend: str  # 'increasing', 'decreasing', 'stable'
    efficiency_score: float
    bottleneck_risk: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationRecommendation:
    """优化建议"""
    recommendation_id: str
    timestamp: datetime
    resource_type: ResourceType
    current_status: ResourceStatus
    recommended_action: OptimizationAction
    priority: str  # 'high', 'medium', 'low'
    expected_improvement: float
    implementation_cost: float
    risk_level: str  # 'low', 'medium', 'high'
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationResult:
    """优化结果"""
    result_id: str
    timestamp: datetime
    recommendation_id: str
    action_taken: OptimizationAction
    success: bool
    before_metrics: ResourceMetrics
    after_metrics: ResourceMetrics
    improvement_achieved: float
    execution_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResourceTarget:
    """资源目标"""
    resource_type: ResourceType
    target_utilization: float
    min_utilization: float
    max_utilization: float
    efficiency_target: float
    performance_target: float
    cost_target: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class ResourceOptimizer:
    """资源使用优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 优化配置
        self.optimization_strategy = OptimizationStrategy(config.get('strategy', 'balanced'))
        self.optimization_interval = config.get('optimization_interval', 60)
        self.auto_optimization = config.get('auto_optimization', True)
        
        # 资源监控
        self.resource_metrics: Dict[ResourceType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.resource_targets: Dict[ResourceType, ResourceTarget] = {}
        
        # 优化历史
        self.optimization_history = deque(maxlen=config.get('history_size', 500))
        self.recommendations = deque(maxlen=config.get('recommendations_size', 100))
        
        # 运行状态
        self.is_running = False
        self.optimization_thread = None
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # 执行器
        self.thread_pool = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        
        # 回调函数
        self.optimization_callbacks: List[callable] = []
        self.alert_callbacks: List[callable] = []
        
        # 初始化
        self._initialize_targets()
        self._initialize_optimizers()
    
    def _initialize_targets(self):
        """初始化资源目标"""
        default_targets = {
            ResourceType.CPU: ResourceTarget(
                resource_type=ResourceType.CPU,
                target_utilization=70.0,
                min_utilization=30.0,
                max_utilization=85.0,
                efficiency_target=0.8,
                performance_target=0.9,
                cost_target=0.7
            ),
            ResourceType.MEMORY: ResourceTarget(
                resource_type=ResourceType.MEMORY,
                target_utilization=75.0,
                min_utilization=40.0,
                max_utilization=90.0,
                efficiency_target=0.85,
                performance_target=0.9,
                cost_target=0.8
            ),
            ResourceType.DISK: ResourceTarget(
                resource_type=ResourceType.DISK,
                target_utilization=60.0,
                min_utilization=20.0,
                max_utilization=80.0,
                efficiency_target=0.75,
                performance_target=0.85,
                cost_target=0.8
            ),
            ResourceType.NETWORK: ResourceTarget(
                resource_type=ResourceType.NETWORK,
                target_utilization=50.0,
                min_utilization=10.0,
                max_utilization=70.0,
                efficiency_target=0.8,
                performance_target=0.9,
                cost_target=0.9
            ),
            ResourceType.THREAD: ResourceTarget(
                resource_type=ResourceType.THREAD,
                target_utilization=65.0,
                min_utilization=25.0,
                max_utilization=85.0,
                efficiency_target=0.8,
                performance_target=0.85,
                cost_target=0.75
            )
        }
        
        # 加载用户自定义目标
        custom_targets = self.config.get('resource_targets', {})
        
        for resource_type, target in default_targets.items():
            if resource_type.value in custom_targets:
                custom_config = custom_targets[resource_type.value]
                for key, value in custom_config.items():
                    if hasattr(target, key):
                        setattr(target, key, value)
            
            self.resource_targets[resource_type] = target
    
    def _initialize_optimizers(self):
        """初始化优化器"""
        # 注册优化器
        self.optimizers = {
            ResourceType.CPU: self._optimize_cpu,
            ResourceType.MEMORY: self._optimize_memory,
            ResourceType.DISK: self._optimize_disk,
            ResourceType.NETWORK: self._optimize_network,
            ResourceType.THREAD: self._optimize_thread_pool,
            ResourceType.CACHE: self._optimize_cache,
            ResourceType.QUEUE: self._optimize_queue
        }
    
    def start(self):
        """启动资源优化"""
        if self.is_running:
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # 启动监控线程
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # 启动优化线程
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        self.logger.info("Resource optimizer started")
    
    def stop(self):
        """停止资源优化"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping resource optimizer...")
        self.stop_event.set()
        
        # 停止线程
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        
        # 关闭线程池
        self.thread_pool.shutdown(wait=True)
        
        self.is_running = False
        self.logger.info("Resource optimizer stopped")
    
    def _monitoring_loop(self):
        """监控主循环"""
        while not self.stop_event.is_set():
            try:
                # 收集所有资源指标
                for resource_type in self.resource_targets:
                    metrics = self._collect_resource_metrics(resource_type)
                    if metrics:
                        self.resource_metrics[resource_type].append(metrics)
                
                time.sleep(self.config.get('monitoring_interval', 10))
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def _optimization_loop(self):
        """优化主循环"""
        while not self.stop_event.is_set():
            try:
                if self.auto_optimization:
                    # 生成优化建议
                    recommendations = self._generate_recommendations()
                    
                    # 执行优化
                    for recommendation in recommendations:
                        if recommendation.priority == 'high':
                            self._execute_optimization(recommendation)
                
                time.sleep(self.optimization_interval)
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                time.sleep(1)
    
    def _collect_resource_metrics(self, resource_type: ResourceType) -> Optional[ResourceMetrics]:
        """收集资源指标"""
        try:
            current_time = datetime.now()
            
            if resource_type == ResourceType.CPU:
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_count = psutil.cpu_count()
                
                return ResourceMetrics(
                    resource_type=resource_type,
                    timestamp=current_time,
                    current_usage=cpu_percent,
                    maximum_capacity=100.0,
                    utilization_percentage=cpu_percent,
                    available_capacity=100.0 - cpu_percent,
                    peak_usage=self._get_peak_usage(resource_type),
                    average_usage=self._get_average_usage(resource_type),
                    trend=self._calculate_trend(resource_type),
                    efficiency_score=self._calculate_efficiency(resource_type, cpu_percent),
                    bottleneck_risk=self._calculate_bottleneck_risk(resource_type, cpu_percent),
                    metadata={'cpu_count': cpu_count}
                )
            
            elif resource_type == ResourceType.MEMORY:
                memory = psutil.virtual_memory()
                
                return ResourceMetrics(
                    resource_type=resource_type,
                    timestamp=current_time,
                    current_usage=memory.used,
                    maximum_capacity=memory.total,
                    utilization_percentage=memory.percent,
                    available_capacity=memory.available,
                    peak_usage=self._get_peak_usage(resource_type),
                    average_usage=self._get_average_usage(resource_type),
                    trend=self._calculate_trend(resource_type),
                    efficiency_score=self._calculate_efficiency(resource_type, memory.percent),
                    bottleneck_risk=self._calculate_bottleneck_risk(resource_type, memory.percent),
                    metadata={'total_memory': memory.total}
                )
            
            elif resource_type == ResourceType.DISK:
                disk = psutil.disk_usage('/')
                
                return ResourceMetrics(
                    resource_type=resource_type,
                    timestamp=current_time,
                    current_usage=disk.used,
                    maximum_capacity=disk.total,
                    utilization_percentage=(disk.used / disk.total) * 100,
                    available_capacity=disk.free,
                    peak_usage=self._get_peak_usage(resource_type),
                    average_usage=self._get_average_usage(resource_type),
                    trend=self._calculate_trend(resource_type),
                    efficiency_score=self._calculate_efficiency(resource_type, (disk.used / disk.total) * 100),
                    bottleneck_risk=self._calculate_bottleneck_risk(resource_type, (disk.used / disk.total) * 100),
                    metadata={'total_disk': disk.total}
                )
            
            elif resource_type == ResourceType.NETWORK:
                net_io = psutil.net_io_counters()
                
                return ResourceMetrics(
                    resource_type=resource_type,
                    timestamp=current_time,
                    current_usage=net_io.bytes_sent + net_io.bytes_recv,
                    maximum_capacity=self._estimate_network_capacity(),
                    utilization_percentage=self._calculate_network_utilization(),
                    available_capacity=self._estimate_network_capacity() - (net_io.bytes_sent + net_io.bytes_recv),
                    peak_usage=self._get_peak_usage(resource_type),
                    average_usage=self._get_average_usage(resource_type),
                    trend=self._calculate_trend(resource_type),
                    efficiency_score=self._calculate_efficiency(resource_type, self._calculate_network_utilization()),
                    bottleneck_risk=self._calculate_bottleneck_risk(resource_type, self._calculate_network_utilization()),
                    metadata={'bytes_sent': net_io.bytes_sent, 'bytes_recv': net_io.bytes_recv}
                )
            
            elif resource_type == ResourceType.THREAD:
                thread_count = threading.active_count()
                max_threads = self.config.get('max_threads', 100)
                
                return ResourceMetrics(
                    resource_type=resource_type,
                    timestamp=current_time,
                    current_usage=thread_count,
                    maximum_capacity=max_threads,
                    utilization_percentage=(thread_count / max_threads) * 100,
                    available_capacity=max_threads - thread_count,
                    peak_usage=self._get_peak_usage(resource_type),
                    average_usage=self._get_average_usage(resource_type),
                    trend=self._calculate_trend(resource_type),
                    efficiency_score=self._calculate_efficiency(resource_type, (thread_count / max_threads) * 100),
                    bottleneck_risk=self._calculate_bottleneck_risk(resource_type, (thread_count / max_threads) * 100),
                    metadata={'active_threads': thread_count}
                )
            
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error collecting metrics for {resource_type}: {e}")
            return None
    
    def _get_peak_usage(self, resource_type: ResourceType) -> float:
        """获取峰值使用率"""
        if resource_type not in self.resource_metrics:
            return 0.0
        
        metrics = list(self.resource_metrics[resource_type])
        if not metrics:
            return 0.0
        
        return max(m.utilization_percentage for m in metrics)
    
    def _get_average_usage(self, resource_type: ResourceType) -> float:
        """获取平均使用率"""
        if resource_type not in self.resource_metrics:
            return 0.0
        
        metrics = list(self.resource_metrics[resource_type])
        if not metrics:
            return 0.0
        
        return sum(m.utilization_percentage for m in metrics) / len(metrics)
    
    def _calculate_trend(self, resource_type: ResourceType) -> str:
        """计算趋势"""
        if resource_type not in self.resource_metrics:
            return 'stable'
        
        metrics = list(self.resource_metrics[resource_type])
        if len(metrics) < 2:
            return 'stable'
        
        recent_values = [m.utilization_percentage for m in metrics[-10:]]
        
        if len(recent_values) < 2:
            return 'stable'
        
        # 计算趋势
        trend_slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        
        if trend_slope > 1:
            return 'increasing'
        elif trend_slope < -1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_efficiency(self, resource_type: ResourceType, utilization: float) -> float:
        """计算效率分数"""
        target = self.resource_targets.get(resource_type)
        if not target:
            return 0.5
        
        # 效率分数基于与目标利用率的接近程度
        distance_from_target = abs(utilization - target.target_utilization)
        max_distance = max(target.target_utilization, 100 - target.target_utilization)
        
        efficiency = 1.0 - (distance_from_target / max_distance)
        return max(0.0, min(1.0, efficiency))
    
    def _calculate_bottleneck_risk(self, resource_type: ResourceType, utilization: float) -> float:
        """计算瓶颈风险"""
        target = self.resource_targets.get(resource_type)
        if not target:
            return 0.5
        
        if utilization >= target.max_utilization:
            return 1.0
        elif utilization >= target.target_utilization:
            return (utilization - target.target_utilization) / (target.max_utilization - target.target_utilization)
        else:
            return 0.0
    
    def _estimate_network_capacity(self) -> float:
        """估算网络容量"""
        # 简化实现，实际应该基于网络接口速度
        return 1000000000.0  # 1GB/s
    
    def _calculate_network_utilization(self) -> float:
        """计算网络利用率"""
        # 简化实现，实际应该基于网络流量监控
        return np.random.uniform(5, 25)  # 模拟网络利用率
    
    def _generate_recommendations(self) -> List[OptimizationRecommendation]:
        """生成优化建议"""
        recommendations = []
        
        for resource_type, metrics_deque in self.resource_metrics.items():
            if not metrics_deque:
                continue
            
            latest_metrics = metrics_deque[-1]
            target = self.resource_targets.get(resource_type)
            
            if not target:
                continue
            
            # 检查是否需要优化
            recommendation = self._analyze_resource_optimization(latest_metrics, target)
            
            if recommendation:
                recommendations.append(recommendation)
                self.recommendations.append(recommendation)
        
        return recommendations
    
    def _analyze_resource_optimization(self, metrics: ResourceMetrics, 
                                     target: ResourceTarget) -> Optional[OptimizationRecommendation]:
        """分析资源优化需求"""
        utilization = metrics.utilization_percentage
        
        # 确定状态
        if utilization > target.max_utilization:
            status = ResourceStatus.OVERUTILIZED
            priority = 'high'
        elif utilization < target.min_utilization:
            status = ResourceStatus.UNDERUTILIZED
            priority = 'medium'
        elif metrics.bottleneck_risk > 0.7:
            status = ResourceStatus.CRITICAL
            priority = 'high'
        else:
            status = ResourceStatus.OPTIMAL
            priority = 'low'
        
        if status == ResourceStatus.OPTIMAL:
            return None
        
        # 确定优化动作
        if status == ResourceStatus.OVERUTILIZED:
            if metrics.resource_type == ResourceType.CPU:
                action = OptimizationAction.SCALE_UP
            elif metrics.resource_type == ResourceType.MEMORY:
                action = OptimizationAction.GARBAGE_COLLECT
            elif metrics.resource_type == ResourceType.THREAD:
                action = OptimizationAction.THREAD_POOL_RESIZE
            else:
                action = OptimizationAction.REBALANCE
        elif status == ResourceStatus.UNDERUTILIZED:
            action = OptimizationAction.SCALE_DOWN
        else:
            action = OptimizationAction.REBALANCE
        
        # 计算预期改善
        expected_improvement = self._calculate_expected_improvement(metrics, target, action)
        
        return OptimizationRecommendation(
            recommendation_id=f"{metrics.resource_type.value}_{int(time.time())}",
            timestamp=datetime.now(),
            resource_type=metrics.resource_type,
            current_status=status,
            recommended_action=action,
            priority=priority,
            expected_improvement=expected_improvement,
            implementation_cost=self._calculate_implementation_cost(action),
            risk_level=self._calculate_risk_level(action),
            description=self._generate_recommendation_description(metrics, action),
            parameters=self._generate_optimization_parameters(metrics, target, action)
        )
    
    def _calculate_expected_improvement(self, metrics: ResourceMetrics, 
                                      target: ResourceTarget, 
                                      action: OptimizationAction) -> float:
        """计算预期改善"""
        current_efficiency = metrics.efficiency_score
        target_efficiency = target.efficiency_target
        
        # 根据动作类型估算改善
        if action == OptimizationAction.SCALE_UP:
            return min(0.3, target_efficiency - current_efficiency)
        elif action == OptimizationAction.SCALE_DOWN:
            return min(0.2, target_efficiency - current_efficiency)
        elif action == OptimizationAction.GARBAGE_COLLECT:
            return min(0.4, target_efficiency - current_efficiency)
        elif action == OptimizationAction.REBALANCE:
            return min(0.25, target_efficiency - current_efficiency)
        else:
            return min(0.1, target_efficiency - current_efficiency)
    
    def _calculate_implementation_cost(self, action: OptimizationAction) -> float:
        """计算实施成本"""
        cost_map = {
            OptimizationAction.SCALE_UP: 0.8,
            OptimizationAction.SCALE_DOWN: 0.3,
            OptimizationAction.GARBAGE_COLLECT: 0.1,
            OptimizationAction.REBALANCE: 0.5,
            OptimizationAction.CACHE_CLEAR: 0.2,
            OptimizationAction.THREAD_POOL_RESIZE: 0.4
        }
        
        return cost_map.get(action, 0.5)
    
    def _calculate_risk_level(self, action: OptimizationAction) -> str:
        """计算风险等级"""
        risk_map = {
            OptimizationAction.SCALE_UP: 'medium',
            OptimizationAction.SCALE_DOWN: 'low',
            OptimizationAction.GARBAGE_COLLECT: 'low',
            OptimizationAction.REBALANCE: 'medium',
            OptimizationAction.CACHE_CLEAR: 'low',
            OptimizationAction.THREAD_POOL_RESIZE: 'medium'
        }
        
        return risk_map.get(action, 'medium')
    
    def _generate_recommendation_description(self, metrics: ResourceMetrics, 
                                           action: OptimizationAction) -> str:
        """生成建议描述"""
        resource_name = metrics.resource_type.value.upper()
        utilization = metrics.utilization_percentage
        
        if action == OptimizationAction.SCALE_UP:
            return f"{resource_name} utilization at {utilization:.1f}% - recommend scaling up"
        elif action == OptimizationAction.SCALE_DOWN:
            return f"{resource_name} utilization at {utilization:.1f}% - recommend scaling down"
        elif action == OptimizationAction.GARBAGE_COLLECT:
            return f"{resource_name} utilization at {utilization:.1f}% - recommend garbage collection"
        elif action == OptimizationAction.REBALANCE:
            return f"{resource_name} utilization at {utilization:.1f}% - recommend rebalancing"
        else:
            return f"{resource_name} optimization recommended - {action.value}"
    
    def _generate_optimization_parameters(self, metrics: ResourceMetrics, 
                                        target: ResourceTarget, 
                                        action: OptimizationAction) -> Dict[str, Any]:
        """生成优化参数"""
        parameters = {
            'current_utilization': metrics.utilization_percentage,
            'target_utilization': target.target_utilization,
            'resource_type': metrics.resource_type.value
        }
        
        if action == OptimizationAction.THREAD_POOL_RESIZE:
            current_threads = metrics.current_usage
            target_threads = int(current_threads * (target.target_utilization / metrics.utilization_percentage))
            parameters['target_thread_count'] = target_threads
        
        return parameters
    
    def _execute_optimization(self, recommendation: OptimizationRecommendation) -> OptimizationResult:
        """执行优化"""
        start_time = time.time()
        
        try:
            # 获取优化前的指标
            before_metrics = self._collect_resource_metrics(recommendation.resource_type)
            
            # 执行优化
            success = self._perform_optimization_action(recommendation)
            
            # 等待一段时间让优化生效
            time.sleep(5)
            
            # 获取优化后的指标
            after_metrics = self._collect_resource_metrics(recommendation.resource_type)
            
            # 计算改善
            improvement = 0.0
            if before_metrics and after_metrics:
                improvement = after_metrics.efficiency_score - before_metrics.efficiency_score
            
            result = OptimizationResult(
                result_id=f"opt_{int(time.time())}",
                timestamp=datetime.now(),
                recommendation_id=recommendation.recommendation_id,
                action_taken=recommendation.recommended_action,
                success=success,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_achieved=improvement,
                execution_time=time.time() - start_time
            )
            
            self.optimization_history.append(result)
            
            # 通知回调
            for callback in self.optimization_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    self.logger.error(f"Error in optimization callback: {e}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing optimization: {e}")
            
            result = OptimizationResult(
                result_id=f"opt_{int(time.time())}",
                timestamp=datetime.now(),
                recommendation_id=recommendation.recommendation_id,
                action_taken=recommendation.recommended_action,
                success=False,
                before_metrics=before_metrics,
                after_metrics=None,
                improvement_achieved=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
            
            self.optimization_history.append(result)
            return result
    
    def _perform_optimization_action(self, recommendation: OptimizationRecommendation) -> bool:
        """执行优化动作"""
        try:
            action = recommendation.recommended_action
            resource_type = recommendation.resource_type
            
            if action == OptimizationAction.GARBAGE_COLLECT:
                # 执行垃圾回收
                gc.collect()
                return True
            
            elif action == OptimizationAction.CACHE_CLEAR:
                # 清理缓存
                self._clear_caches()
                return True
            
            elif action == OptimizationAction.THREAD_POOL_RESIZE:
                # 调整线程池大小
                target_threads = recommendation.parameters.get('target_thread_count', 50)
                return self._resize_thread_pool(target_threads)
            
            elif action == OptimizationAction.REBALANCE:
                # 资源重平衡
                return self._rebalance_resources(resource_type)
            
            else:
                # 其他动作的实现
                self.logger.warning(f"Optimization action {action} not implemented")
                return False
                
        except Exception as e:
            self.logger.error(f"Error performing optimization action: {e}")
            return False
    
    def _clear_caches(self):
        """清理缓存"""
        # 实现缓存清理逻辑
        pass
    
    def _resize_thread_pool(self, target_size: int) -> bool:
        """调整线程池大小"""
        try:
            # 重新创建线程池
            self.thread_pool.shutdown(wait=False)
            self.thread_pool = ThreadPoolExecutor(max_workers=target_size)
            return True
        except Exception as e:
            self.logger.error(f"Error resizing thread pool: {e}")
            return False
    
    def _rebalance_resources(self, resource_type: ResourceType) -> bool:
        """资源重平衡"""
        try:
            if resource_type == ResourceType.CPU:
                # CPU重平衡逻辑
                pass
            elif resource_type == ResourceType.MEMORY:
                # 内存重平衡逻辑
                gc.collect()
            elif resource_type == ResourceType.DISK:
                # 磁盘重平衡逻辑
                pass
            
            return True
        except Exception as e:
            self.logger.error(f"Error rebalancing {resource_type}: {e}")
            return False
    
    # 优化器方法
    def _optimize_cpu(self, metrics: ResourceMetrics) -> bool:
        """CPU优化"""
        # 实现CPU优化逻辑
        return True
    
    def _optimize_memory(self, metrics: ResourceMetrics) -> bool:
        """内存优化"""
        gc.collect()
        return True
    
    def _optimize_disk(self, metrics: ResourceMetrics) -> bool:
        """磁盘优化"""
        # 实现磁盘优化逻辑
        return True
    
    def _optimize_network(self, metrics: ResourceMetrics) -> bool:
        """网络优化"""
        # 实现网络优化逻辑
        return True
    
    def _optimize_thread_pool(self, metrics: ResourceMetrics) -> bool:
        """线程池优化"""
        # 实现线程池优化逻辑
        return True
    
    def _optimize_cache(self, metrics: ResourceMetrics) -> bool:
        """缓存优化"""
        self._clear_caches()
        return True
    
    def _optimize_queue(self, metrics: ResourceMetrics) -> bool:
        """队列优化"""
        # 实现队列优化逻辑
        return True
    
    def manual_optimization(self, resource_type: ResourceType, 
                          action: OptimizationAction) -> OptimizationResult:
        """手动优化"""
        recommendation = OptimizationRecommendation(
            recommendation_id=f"manual_{int(time.time())}",
            timestamp=datetime.now(),
            resource_type=resource_type,
            current_status=ResourceStatus.UNKNOWN,
            recommended_action=action,
            priority='medium',
            expected_improvement=0.2,
            implementation_cost=0.5,
            risk_level='medium',
            description=f"Manual optimization: {action.value} for {resource_type.value}"
        )
        
        return self._execute_optimization(recommendation)
    
    def get_current_metrics(self) -> Dict[str, ResourceMetrics]:
        """获取当前指标"""
        current_metrics = {}
        
        for resource_type, metrics_deque in self.resource_metrics.items():
            if metrics_deque:
                current_metrics[resource_type.value] = metrics_deque[-1]
        
        return current_metrics
    
    def get_optimization_history(self, resource_type: Optional[ResourceType] = None) -> List[OptimizationResult]:
        """获取优化历史"""
        history = list(self.optimization_history)
        
        if resource_type:
            history = [r for r in history if r.before_metrics and r.before_metrics.resource_type == resource_type]
        
        return history
    
    def get_recommendations(self, resource_type: Optional[ResourceType] = None) -> List[OptimizationRecommendation]:
        """获取建议"""
        recommendations = list(self.recommendations)
        
        if resource_type:
            recommendations = [r for r in recommendations if r.resource_type == resource_type]
        
        return recommendations
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'optimizer_status': 'running' if self.is_running else 'stopped',
            'optimization_strategy': self.optimization_strategy.value,
            'resource_summary': {},
            'optimization_summary': {
                'total_optimizations': len(self.optimization_history),
                'successful_optimizations': len([r for r in self.optimization_history if r.success]),
                'average_improvement': np.mean([r.improvement_achieved for r in self.optimization_history if r.success]) if self.optimization_history else 0.0
            }
        }
        
        # 资源摘要
        for resource_type, metrics_deque in self.resource_metrics.items():
            if metrics_deque:
                latest_metrics = metrics_deque[-1]
                target = self.resource_targets.get(resource_type)
                
                report['resource_summary'][resource_type.value] = {
                    'current_utilization': latest_metrics.utilization_percentage,
                    'target_utilization': target.target_utilization if target else None,
                    'efficiency_score': latest_metrics.efficiency_score,
                    'bottleneck_risk': latest_metrics.bottleneck_risk,
                    'trend': latest_metrics.trend,
                    'status': self._determine_resource_status(latest_metrics, target).value if target else 'unknown'
                }
        
        return report
    
    def _determine_resource_status(self, metrics: ResourceMetrics, target: ResourceTarget) -> ResourceStatus:
        """确定资源状态"""
        utilization = metrics.utilization_percentage
        
        if utilization > target.max_utilization:
            return ResourceStatus.OVERUTILIZED
        elif utilization < target.min_utilization:
            return ResourceStatus.UNDERUTILIZED
        elif metrics.bottleneck_risk > 0.7:
            return ResourceStatus.CRITICAL
        else:
            return ResourceStatus.OPTIMAL
    
    def add_optimization_callback(self, callback: callable):
        """添加优化回调"""
        self.optimization_callbacks.append(callback)
    
    def add_alert_callback(self, callback: callable):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'optimizer_running': self.is_running,
            'monitored_resources': len(self.resource_metrics),
            'optimization_history': len(self.optimization_history),
            'pending_recommendations': len(self.recommendations),
            'optimization_strategy': self.optimization_strategy.value,
            'auto_optimization_enabled': self.auto_optimization
        }
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'is_running') and self.is_running:
            self.stop()