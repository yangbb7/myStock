# -*- coding: utf-8 -*-
"""
实时风险监控器 - 提供实时风险监控和预警功能
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskAlert:
    """风险预警"""
    alert_id: str
    risk_type: str
    level: RiskLevel
    message: str
    timestamp: datetime
    portfolio_id: Optional[str] = None
    symbol: Optional[str] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskLimit:
    """风险限制"""
    limit_id: str
    limit_type: str  # 'position_limit', 'var_limit', 'drawdown_limit', etc.
    threshold: float
    current_value: float = 0.0
    is_breached: bool = False
    alert_level: RiskLevel = RiskLevel.MEDIUM
    enabled: bool = True


class RealTimeRiskMonitor:
    """实时风险监控器"""
    
    def __init__(self, risk_calculator=None, alert_manager=None):
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.risk_limits: Dict[str, RiskLimit] = {}
        self.alert_handlers: List[Callable[[RiskAlert], None]] = []
        self.monitoring_interval = 1.0  # 监控间隔（秒）
        self.active_alerts: Dict[str, RiskAlert] = {}
        self.risk_calculator = risk_calculator
        self.alert_manager = alert_manager
        
    async def start_monitoring(self):
        """开始实时监控"""
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info("实时风险监控已启动")
        
        # 启动监控循环
        asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """停止实时监控"""
        self.is_running = False
        self.logger.info("实时风险监控已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                await self.check_all_limits()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"风险监控循环出错: {e}")
                await asyncio.sleep(1)  # 出错时等待略短时间
    
    def add_risk_limit(self, limit: RiskLimit):
        """添加风险限制"""
        self.risk_limits[limit.limit_id] = limit
        self.logger.info(f"添加风险限制: {limit.limit_id}")
    
    def remove_risk_limit(self, limit_id: str):
        """移除风险限制"""
        if limit_id in self.risk_limits:
            del self.risk_limits[limit_id]
            self.logger.info(f"移除风险限制: {limit_id}")
    
    def update_risk_limit_value(self, limit_id: str, current_value: float):
        """更新风险限制当前值"""
        if limit_id in self.risk_limits:
            limit = self.risk_limits[limit_id]
            limit.current_value = current_value
            
            # 检查是否超限
            was_breached = limit.is_breached
            limit.is_breached = self._check_limit_breach(limit)
            
            # 如果从正常变为超限，发送预警
            if not was_breached and limit.is_breached:
                alert = self._create_limit_alert(limit)
                asyncio.create_task(self.send_alert(alert))
    
    def _check_limit_breach(self, limit: RiskLimit) -> bool:
        """检查限制是否超限"""
        if not limit.enabled:
            return False
        
        if limit.limit_type in ['position_limit', 'var_limit']:
            return abs(limit.current_value) > limit.threshold
        elif limit.limit_type == 'drawdown_limit':
            return limit.current_value < -limit.threshold  # 回撤是负值
        else:
            return limit.current_value > limit.threshold
    
    def _create_limit_alert(self, limit: RiskLimit) -> RiskAlert:
        """创建限制预警"""
        return RiskAlert(
            alert_id=f"limit_breach_{limit.limit_id}_{datetime.now().timestamp()}",
            risk_type=limit.limit_type,
            level=limit.alert_level,
            message=f"风险限制超限: {limit.limit_type}, 当前值: {limit.current_value}, 阈值: {limit.threshold}",
            timestamp=datetime.now(),
            current_value=limit.current_value,
            threshold=limit.threshold,
            metadata={'limit_id': limit.limit_id}
        )
    
    async def check_all_limits(self):
        """检查所有风险限制"""
        for limit in self.risk_limits.values():
            if limit.enabled:
                # 这里应该从实际的数据源更新限制值
                # 现在只是示例，实际应该连接到投资组合管理器等
                pass
    
    def add_alert_handler(self, handler: Callable[[RiskAlert], None]):
        """添加预警处理器"""
        self.alert_handlers.append(handler)
    
    async def send_alert(self, alert: RiskAlert):
        """发送风险预警"""
        self.active_alerts[alert.alert_id] = alert
        self.logger.warning(f"风险预警: {alert.message}")
        
        # 调用所有注册的预警处理器
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"预警处理器出错: {e}")
    
    def get_active_alerts(self, risk_level: Optional[RiskLevel] = None) -> List[RiskAlert]:
        """获取活跃预警"""
        alerts = list(self.active_alerts.values())
        if risk_level:
            alerts = [alert for alert in alerts if alert.level == risk_level]
        return alerts
    
    def clear_alert(self, alert_id: str):
        """清除预警"""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            self.logger.info(f"清除预警: {alert_id}")
    
    def clear_all_alerts(self):
        """清除所有预警"""
        self.active_alerts.clear()
        self.logger.info("清除所有预警")
    
    def get_risk_limits_status(self) -> Dict[str, Dict[str, Any]]:
        """获取风险限制状态"""
        status = {}
        for limit_id, limit in self.risk_limits.items():
            status[limit_id] = {
                'limit_type': limit.limit_type,
                'threshold': limit.threshold,
                'current_value': limit.current_value,
                'is_breached': limit.is_breached,
                'enabled': limit.enabled,
                'utilization': abs(limit.current_value / limit.threshold) if limit.threshold != 0 else 0
            }
        return status
