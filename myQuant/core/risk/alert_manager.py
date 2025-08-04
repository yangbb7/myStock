# -*- coding: utf-8 -*-
"""
预警管理器 - 管理风险预警的生成、分发和处理
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict


class AlertLevel(Enum):
    """预警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """预警类型"""
    POSITION_LIMIT = "position_limit"
    VAR_LIMIT = "var_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_RISK = "liquidity_risk"
    CONCENTRATION_RISK = "concentration_risk"
    MARGIN_CALL = "margin_call"
    SYSTEM_ERROR = "system_error"


@dataclass
class RiskAlert:
    """风险预警"""
    alert_id: str
    alert_type: AlertType
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    portfolio_id: Optional[str] = None
    symbol: Optional[str] = None
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    is_active: bool = True
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """预警规则"""
    rule_id: str
    alert_type: AlertType
    level: AlertLevel
    condition: str  # 触发条件描述
    threshold: float
    enabled: bool = True
    cooldown_period: int = 300  # 冷却期（秒）
    last_triggered: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertChannel(Enum):
    """预警通道"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    DATABASE = "database"
    LOG = "log"


class AlertManager:
    """预警管理器"""
    
    def __init__(self, database_manager=None):
        self.logger = logging.getLogger(__name__)
        self.database_manager = database_manager
        
        # 预警存储
        self.active_alerts: Dict[str, RiskAlert] = {}
        self.alert_history: List[RiskAlert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # 预警处理器
        self.alert_handlers: Dict[AlertChannel, List[Callable[[RiskAlert], None]]] = defaultdict(list)
        
        # 配置
        self.max_history_size = 10000
        self.auto_resolve_timeout = 3600  # 1小时后自动解决
        
    def add_alert_rule(self, rule: AlertRule):
        """添加预警规则"""
        self.alert_rules[rule.rule_id] = rule
        self.logger.info(f"添加预警规则: {rule.rule_id}")
    
    def remove_alert_rule(self, rule_id: str):
        """移除预警规则"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.logger.info(f"移除预警规则: {rule_id}")
    
    def update_alert_rule(self, rule_id: str, **kwargs):
        """更新预警规则"""
        if rule_id in self.alert_rules:
            rule = self.alert_rules[rule_id]
            for key, value in kwargs.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            self.logger.info(f"更新预警规则: {rule_id}")
    
    def add_alert_handler(self, channel: AlertChannel, handler: Callable[[RiskAlert], None]):
        """添加预警处理器"""
        self.alert_handlers[channel].append(handler)
        self.logger.info(f"添加预警处理器: {channel.value}")
    
    def create_alert(self, alert_type: AlertType, level: AlertLevel, 
                    title: str, message: str, **kwargs) -> RiskAlert:
        """创建预警"""
        alert_id = f"{alert_type.value}_{datetime.now().timestamp()}"
        
        alert = RiskAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(),
            **kwargs
        )
        
        return alert
    
    async def trigger_alert(self, alert: RiskAlert, channels: Optional[List[AlertChannel]] = None):
        """触发预警"""
        # 检查是否在冷却期内
        if self._is_in_cooldown(alert):
            return
        
        # 添加到活跃预警
        self.active_alerts[alert.alert_id] = alert
        
        # 添加到历史记录
        self.alert_history.append(alert)
        self._cleanup_history()
        
        # 发送预警
        await self._send_alert(alert, channels)
        
        # 更新规则的最后触发时间
        self._update_rule_last_triggered(alert)
        
        self.logger.warning(f"触发预警: {alert.title} - {alert.message}")
    
    def _is_in_cooldown(self, alert: RiskAlert) -> bool:
        """检查是否在冷却期内"""
        for rule in self.alert_rules.values():
            if (rule.alert_type == alert.alert_type and 
                rule.last_triggered and 
                (datetime.now() - rule.last_triggered).seconds < rule.cooldown_period):
                return True
        return False
    
    def _update_rule_last_triggered(self, alert: RiskAlert):
        """更新规则的最后触发时间"""
        for rule in self.alert_rules.values():
            if rule.alert_type == alert.alert_type:
                rule.last_triggered = datetime.now()
    
    async def _send_alert(self, alert: RiskAlert, channels: Optional[List[AlertChannel]] = None):
        """发送预警"""
        if channels is None:
            channels = list(self.alert_handlers.keys())
        
        for channel in channels:
            if channel in self.alert_handlers:
                for handler in self.alert_handlers[channel]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(alert)
                        else:
                            handler(alert)
                    except Exception as e:
                        self.logger.error(f"预警处理器错误 ({channel.value}): {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """确认预警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()
            self.logger.info(f"预警已确认: {alert_id} by {acknowledged_by}")
    
    def resolve_alert(self, alert_id: str):
        """解决预警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            alert.is_active = False
            
            # 从活跃预警中移除
            del self.active_alerts[alert_id]
            
            self.logger.info(f"预警已解决: {alert_id}")
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None, 
                         alert_type: Optional[AlertType] = None) -> List[RiskAlert]:
        """获取活跃预警"""
        alerts = list(self.active_alerts.values())
        
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        if alert_type:
            alerts = [alert for alert in alerts if alert.alert_type == alert_type]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_history(self, start_time: Optional[datetime] = None, 
                         end_time: Optional[datetime] = None, 
                         limit: int = 100) -> List[RiskAlert]:
        """获取预警历史"""
        alerts = self.alert_history.copy()
        
        if start_time:
            alerts = [alert for alert in alerts if alert.timestamp >= start_time]
        
        if end_time:
            alerts = [alert for alert in alerts if alert.timestamp <= end_time]
        
        alerts = sorted(alerts, key=lambda x: x.timestamp, reverse=True)
        return alerts[:limit]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取预警统计信息"""
        active_count = len(self.active_alerts)
        total_count = len(self.alert_history)
        
        # 按级别统计
        level_stats = defaultdict(int)
        for alert in self.alert_history:
            level_stats[alert.level.value] += 1
        
        # 按类型统计
        type_stats = defaultdict(int)
        for alert in self.alert_history:
            type_stats[alert.alert_type.value] += 1
        
        # 按时间统计（最近24小时）
        recent_alerts = [
            alert for alert in self.alert_history 
            if (datetime.now() - alert.timestamp).total_seconds() < 86400
        ]
        
        return {
            'active_alerts': active_count,
            'total_alerts': total_count,
            'recent_alerts_24h': len(recent_alerts),
            'level_distribution': dict(level_stats),
            'type_distribution': dict(type_stats),
            'acknowledged_count': sum(1 for alert in self.alert_history if alert.acknowledged),
            'resolved_count': sum(1 for alert in self.alert_history if alert.resolved)
        }
    
    def _cleanup_history(self):
        """清理历史记录"""
        if len(self.alert_history) > self.max_history_size:
            # 保留最近的记录
            self.alert_history = sorted(
                self.alert_history, 
                key=lambda x: x.timestamp, 
                reverse=True
            )[:self.max_history_size]
    
    async def auto_resolve_alerts(self):
        """自动解决超时预警"""
        current_time = datetime.now()
        alerts_to_resolve = []
        
        for alert_id, alert in self.active_alerts.items():
            if (current_time - alert.timestamp).total_seconds() > self.auto_resolve_timeout:
                alerts_to_resolve.append(alert_id)
        
        for alert_id in alerts_to_resolve:
            self.resolve_alert(alert_id)
            self.logger.info(f"自动解决超时预警: {alert_id}")
    
    def clear_all_alerts(self):
        """清除所有预警"""
        self.active_alerts.clear()
        self.alert_history.clear()
        self.logger.info("清除所有预警")
