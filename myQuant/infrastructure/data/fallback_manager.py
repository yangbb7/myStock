# -*- coding: utf-8 -*-
"""
故障转移管理器 - 管理数据源故障转移和恢复策略
"""

import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .quality_monitor import DataQualityMonitor, DataQualityMetrics


class DataSourceStatus(Enum):
    """数据源状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    DISABLED = "disabled"


class FallbackStrategy(Enum):
    """故障转移策略"""
    FAIL_FAST = "fail_fast"           # 快速失败，不重试
    RETRY_WITH_BACKOFF = "retry_with_backoff"  # 指数退避重试
    CIRCUIT_BREAKER = "circuit_breaker"        # 熔断器模式
    GRACEFUL_DEGRADATION = "graceful_degradation"  # 优雅降级


class DataSourceHealth:
    """数据源健康状态"""
    
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.status = DataSourceStatus.HEALTHY
        self.last_success_time = datetime.now()
        self.last_failure_time = None
        self.consecutive_failures = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.average_response_time = 0.0
        self.quality_score = 1.0
        self.circuit_breaker_until = None
        
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def is_circuit_breaker_open(self) -> bool:
        """熔断器是否开启"""
        if self.circuit_breaker_until is None:
            return False
        return datetime.now() < self.circuit_breaker_until
    
    def record_success(self, response_time: float, quality_score: float = 1.0):
        """记录成功请求"""
        self.last_success_time = datetime.now()
        self.consecutive_failures = 0
        self.total_requests += 1
        self.successful_requests += 1
        self.quality_score = quality_score
        
        # 更新平均响应时间
        if self.average_response_time == 0:
            self.average_response_time = response_time
        else:
            self.average_response_time = (self.average_response_time * 0.8 + response_time * 0.2)
        
        # 更新状态
        if self.status in [DataSourceStatus.FAILED, DataSourceStatus.RECOVERING]:
            if self.consecutive_failures == 0:
                self.status = DataSourceStatus.HEALTHY
                self.circuit_breaker_until = None
    
    def record_failure(self, error_msg: str = ""):
        """记录失败请求"""
        self.last_failure_time = datetime.now()
        self.consecutive_failures += 1
        self.total_requests += 1
        
        # 更新状态
        if self.consecutive_failures >= 3:
            self.status = DataSourceStatus.FAILED
            # 启动熔断器
            self.circuit_breaker_until = datetime.now() + timedelta(minutes=5)
        elif self.consecutive_failures >= 1:
            self.status = DataSourceStatus.DEGRADED


class FallbackManager:
    """故障转移管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 数据源健康状态
        self.source_health: Dict[str, DataSourceHealth] = {}
        
        # 数据质量监控器
        self.quality_monitor = DataQualityMonitor(config.get('quality_config', {}))
        
        # 故障转移配置
        self.fallback_strategy = FallbackStrategy(
            config.get('fallback_strategy', 'graceful_degradation')
        )
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay_base = config.get('retry_delay_base', 1.0)
        self.circuit_breaker_threshold = config.get('circuit_breaker_threshold', 5)
        self.quality_threshold = config.get('quality_threshold', 0.8)
        
        # 数据源优先级配置
        self.source_priorities = config.get('source_priorities', {})
        
    def register_data_source(self, source_name: str, priority: int = 10):
        """注册数据源
        
        Args:
            source_name: 数据源名称
            priority: 优先级（数字越小优先级越高）
        """
        if source_name not in self.source_health:
            self.source_health[source_name] = DataSourceHealth(source_name)
            self.source_priorities[source_name] = priority
            self.logger.info(f"注册数据源: {source_name}, 优先级: {priority}")
    
    def get_available_sources(self, exclude_failed: bool = True) -> List[str]:
        """获取可用数据源列表
        
        Args:
            exclude_failed: 是否排除失败的数据源
            
        Returns:
            List[str]: 按优先级排序的可用数据源列表
        """
        available_sources = []
        
        for source_name, health in self.source_health.items():
            # 排除被禁用的数据源
            if health.status == DataSourceStatus.DISABLED:
                continue
                
            # 可选择排除失败的数据源
            if exclude_failed and health.status == DataSourceStatus.FAILED:
                continue
                
            # 检查熔断器状态
            if health.is_circuit_breaker_open:
                continue
                
            available_sources.append(source_name)
        
        # 按优先级排序
        available_sources.sort(key=lambda x: self.source_priorities.get(x, 10))
        
        return available_sources
    
    def execute_with_fallback(self, 
                            data_fetchers: Dict[str, Callable], 
                            *args, **kwargs) -> Tuple[Any, str, DataQualityMetrics]:
        """执行数据获取，支持故障转移
        
        Args:
            data_fetchers: 数据获取函数字典 {source_name: fetch_function}
            *args: 传递给数据获取函数的参数
            **kwargs: 传递给数据获取函数的关键字参数
            
        Returns:
            Tuple[Any, str, DataQualityMetrics]: (数据, 成功的数据源, 质量指标)
        """
        available_sources = self.get_available_sources()
        
        if not available_sources:
            raise Exception("没有可用的数据源")
        
        last_exception = None
        
        for source_name in available_sources:
            if source_name not in data_fetchers:
                continue
                
            try:
                # 记录开始时间
                start_time = time.time()
                
                # 执行数据获取
                data = data_fetchers[source_name](*args, **kwargs)
                
                # 计算响应时间
                response_time = time.time() - start_time
                
                # 验证数据质量
                quality_metrics = self._validate_data_quality(data, source_name)
                
                # 检查数据质量是否可接受
                if not self.quality_monitor.is_data_acceptable(quality_metrics):
                    self.logger.warning(
                        f"数据源{source_name}数据质量不合格: "
                        f"评分{quality_metrics.overall_score:.2f}"
                    )
                    # 记录质量问题但不算作完全失败
                    self.source_health[source_name].record_success(
                        response_time, quality_metrics.overall_score
                    )
                    continue
                
                # 记录成功
                self.source_health[source_name].record_success(
                    response_time, quality_metrics.overall_score
                )
                
                self.logger.info(
                    f"数据源{source_name}获取数据成功，"
                    f"响应时间: {response_time:.2f}s, "
                    f"质量评分: {quality_metrics.overall_score:.2f}"
                )
                
                return data, source_name, quality_metrics
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"数据源{source_name}获取数据失败: {e}")
                
                # 记录失败
                if source_name in self.source_health:
                    self.source_health[source_name].record_failure(str(e))
                
                # 根据策略决定是否继续尝试
                if self.fallback_strategy == FallbackStrategy.FAIL_FAST:
                    break
                elif self.fallback_strategy == FallbackStrategy.RETRY_WITH_BACKOFF:
                    self._apply_retry_delay(source_name)
        
        # 所有数据源都失败
        error_msg = f"所有数据源都无法获取数据，最后错误: {last_exception}"
        self.logger.error(error_msg)
        raise Exception(error_msg)
    
    def _validate_data_quality(self, data: Any, source_name: str) -> DataQualityMetrics:
        """验证数据质量
        
        Args:
            data: 待验证的数据
            source_name: 数据源名称
            
        Returns:
            DataQualityMetrics: 质量评估结果
        """
        try:
            # 对于实时数据（字典格式）
            if isinstance(data, dict):
                return self.quality_monitor.validate_realtime_data(data)
            
            # 对于历史数据（DataFrame格式）
            elif hasattr(data, 'empty') and hasattr(data, 'columns'):  # pandas DataFrame
                return self.quality_monitor.validate_historical_data(data, source_name)
            
            # 其他数据类型，进行基本验证
            else:
                metrics = DataQualityMetrics()
                if data is None:
                    metrics.issues.append("数据为空")
                    metrics.overall_score = 0.0
                else:
                    metrics.overall_score = 1.0
                return metrics
                
        except Exception as e:
            self.logger.error(f"数据质量验证失败: {e}")
            metrics = DataQualityMetrics()
            metrics.issues.append(f"质量验证异常: {e}")
            metrics.overall_score = 0.5  # 给中等评分，表示不确定
            return metrics
    
    def _apply_retry_delay(self, source_name: str):
        """应用重试延迟（指数退避）"""
        if source_name in self.source_health:
            health = self.source_health[source_name]
            delay = self.retry_delay_base * (2 ** min(health.consecutive_failures - 1, 5))
            self.logger.info(f"数据源{source_name}重试延迟: {delay:.1f}秒")
            time.sleep(delay)
    
    def get_health_report(self) -> Dict[str, Any]:
        """获取健康状态报告
        
        Returns:
            Dict[str, Any]: 健康状态报告
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_sources': len(self.source_health),
            'healthy_sources': 0,
            'degraded_sources': 0,
            'failed_sources': 0,
            'disabled_sources': 0,
            'sources': {}
        }
        
        for source_name, health in self.source_health.items():
            # 统计各状态数量
            if health.status == DataSourceStatus.HEALTHY:
                report['healthy_sources'] += 1
            elif health.status == DataSourceStatus.DEGRADED:
                report['degraded_sources'] += 1
            elif health.status == DataSourceStatus.FAILED:
                report['failed_sources'] += 1
            elif health.status == DataSourceStatus.DISABLED:
                report['disabled_sources'] += 1
            
            # 详细信息
            report['sources'][source_name] = {
                'status': health.status.value,
                'success_rate': health.success_rate,
                'consecutive_failures': health.consecutive_failures,
                'total_requests': health.total_requests,
                'successful_requests': health.successful_requests,
                'average_response_time': health.average_response_time,
                'quality_score': health.quality_score,
                'last_success_time': health.last_success_time.isoformat() if health.last_success_time else None,
                'last_failure_time': health.last_failure_time.isoformat() if health.last_failure_time else None,
                'circuit_breaker_open': health.is_circuit_breaker_open,
                'priority': self.source_priorities.get(source_name, 10)
            }
        
        # 添加质量监控报告
        report['quality_report'] = self.quality_monitor.get_quality_report()
        
        return report
    
    def disable_source(self, source_name: str, reason: str = ""):
        """禁用数据源
        
        Args:
            source_name: 数据源名称
            reason: 禁用原因
        """
        if source_name in self.source_health:
            self.source_health[source_name].status = DataSourceStatus.DISABLED
            self.logger.warning(f"数据源{source_name}已被禁用: {reason}")
    
    def enable_source(self, source_name: str):
        """启用数据源
        
        Args:
            source_name: 数据源名称
        """
        if source_name in self.source_health:
            self.source_health[source_name].status = DataSourceStatus.HEALTHY
            self.source_health[source_name].consecutive_failures = 0
            self.source_health[source_name].circuit_breaker_until = None
            self.logger.info(f"数据源{source_name}已被启用")
    
    def reset_circuit_breaker(self, source_name: str):
        """重置熔断器
        
        Args:
            source_name: 数据源名称
        """
        if source_name in self.source_health:
            health = self.source_health[source_name]
            health.circuit_breaker_until = None
            health.status = DataSourceStatus.RECOVERING
            self.logger.info(f"数据源{source_name}熔断器已重置")
    
    def cleanup_history(self, days_to_keep: int = 7):
        """清理历史数据
        
        Args:
            days_to_keep: 保留天数
        """
        self.quality_monitor.cleanup_cache(days_to_keep * 24)
        self.logger.info(f"已清理{days_to_keep}天前的历史数据")
    
    def set_source_priority(self, source_name: str, priority: int):
        """设置数据源优先级
        
        Args:
            source_name: 数据源名称
            priority: 优先级（数字越小优先级越高）
        """
        self.source_priorities[source_name] = priority
        self.logger.info(f"数据源{source_name}优先级已设置为{priority}")
    
    def get_recommended_sources(self, count: int = 3) -> List[str]:
        """获取推荐的数据源
        
        Args:
            count: 推荐数量
            
        Returns:
            List[str]: 推荐的数据源列表
        """
        # 计算综合评分（成功率 * 质量评分 / 响应时间）
        source_scores = []
        
        for source_name, health in self.source_health.items():
            if health.status == DataSourceStatus.DISABLED:
                continue
                
            # 综合评分算法
            success_weight = 0.4
            quality_weight = 0.3
            speed_weight = 0.2
            freshness_weight = 0.1
            
            success_score = health.success_rate
            quality_score = health.quality_score
            
            # 响应时间评分（越快越好）
            speed_score = 1.0 / (1.0 + health.average_response_time) if health.average_response_time > 0 else 1.0
            
            # 数据新鲜度评分
            if health.last_success_time:
                hours_since_success = (datetime.now() - health.last_success_time).total_seconds() / 3600
                freshness_score = 1.0 / (1.0 + hours_since_success)
            else:
                freshness_score = 0.0
            
            comprehensive_score = (
                success_score * success_weight +
                quality_score * quality_weight +
                speed_score * speed_weight +
                freshness_score * freshness_weight
            )
            
            source_scores.append((source_name, comprehensive_score))
        
        # 按评分排序并返回前N个
        source_scores.sort(key=lambda x: x[1], reverse=True)
        return [source[0] for source in source_scores[:count]]