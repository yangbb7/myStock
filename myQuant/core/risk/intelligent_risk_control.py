# -*- coding: utf-8 -*-
"""
智能风控系统 - 自动止损止盈和仓位管理
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
import logging
from collections import defaultdict

from myQuant.core.models.positions import Position
from myQuant.core.models.orders import Order, OrderType, OrderStatus
from myQuant.core.models.signals import TradingSignal, SignalType


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StopLossType(Enum):
    """止损类型"""
    FIXED = "fixed"               # 固定止损
    TRAILING = "trailing"         # 移动止损
    TIME = "time"                 # 时间止损
    VOLATILITY = "volatility"     # 波动率止损
    ATR = "atr"                   # ATR止损


@dataclass
class RiskRule:
    """风险规则"""
    rule_id: str
    name: str
    condition: str  # 条件表达式
    action: str     # 动作类型
    params: Dict[str, Any] = field(default_factory=dict)
    priority: int = 10
    enabled: bool = True


@dataclass
class StopLossConfig:
    """止损配置"""
    stop_type: StopLossType
    stop_percent: float
    trailing_percent: Optional[float] = None
    time_days: Optional[int] = None
    atr_multiplier: Optional[float] = None
    
    
@dataclass
class PositionLimit:
    """仓位限制"""
    max_position_size: float      # 单个仓位最大金额
    max_position_percent: float   # 单个仓位最大比例
    max_total_exposure: float     # 总仓位最大比例
    max_sector_exposure: float    # 单个板块最大比例
    max_correlation: float        # 最大相关性系数


@dataclass
class RiskAlert:
    """风险警报"""
    alert_id: str
    timestamp: datetime
    level: RiskLevel
    type: str
    message: str
    position_id: Optional[str] = None
    suggested_action: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelligentRiskControl:
    """智能风控系统"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 风险规则
        self.risk_rules: List[RiskRule] = []
        self._init_default_rules()
        
        # 仓位限制
        self.position_limits = PositionLimit(
            max_position_size=self.config.get("max_position_size", 100000),
            max_position_percent=self.config.get("max_position_percent", 0.1),
            max_total_exposure=self.config.get("max_total_exposure", 0.95),
            max_sector_exposure=self.config.get("max_sector_exposure", 0.3),
            max_correlation=self.config.get("max_correlation", 0.7)
        )
        
        # 止损配置
        self.stop_loss_configs: Dict[str, StopLossConfig] = {}
        self._init_stop_loss_configs()
        
        # 风险警报
        self.active_alerts: List[RiskAlert] = []
        
        # 历史数据缓存
        self.price_history: Dict[str, pd.DataFrame] = {}
        self.volatility_cache: Dict[str, float] = {}
        
        # 监控状态
        self.monitoring_active = True
        self.last_check_time = datetime.now()
        
    def _init_default_rules(self):
        """初始化默认风险规则"""
        # 仓位集中度规则
        self.add_risk_rule(RiskRule(
            rule_id="concentration_check",
            name="仓位集中度检查",
            condition="position_percent > 0.15",
            action="alert",
            params={"level": "high", "message": "单一持仓超过15%"},
            priority=8
        ))
        
        # 亏损扩大规则
        self.add_risk_rule(RiskRule(
            rule_id="loss_expansion",
            name="亏损扩大检查",
            condition="unrealized_pnl_percent < -0.1 and price_trend == 'down'",
            action="reduce_position",
            params={"reduce_percent": 0.5},
            priority=9
        ))
        
        # 波动率激增规则
        self.add_risk_rule(RiskRule(
            rule_id="volatility_spike",
            name="波动率激增检查",
            condition="current_volatility > historical_volatility * 2",
            action="tighten_stops",
            params={"stop_adjustment": 0.5},
            priority=7
        ))
        
        # 相关性过高规则
        self.add_risk_rule(RiskRule(
            rule_id="high_correlation",
            name="相关性检查",
            condition="portfolio_correlation > 0.8",
            action="alert",
            params={"level": "medium", "message": "投资组合相关性过高"},
            priority=6
        ))
    
    def _init_stop_loss_configs(self):
        """初始化止损配置"""
        # 默认止损配置
        self.stop_loss_configs["default"] = StopLossConfig(
            stop_type=StopLossType.FIXED,
            stop_percent=0.02
        )
        
        # 趋势策略止损
        self.stop_loss_configs["trend"] = StopLossConfig(
            stop_type=StopLossType.TRAILING,
            stop_percent=0.03,
            trailing_percent=0.02
        )
        
        # 波动策略止损
        self.stop_loss_configs["volatility"] = StopLossConfig(
            stop_type=StopLossType.ATR,
            stop_percent=0.05,
            atr_multiplier=2.0
        )
        
        # 短线策略止损
        self.stop_loss_configs["short_term"] = StopLossConfig(
            stop_type=StopLossType.TIME,
            stop_percent=0.01,
            time_days=3
        )
    
    def add_risk_rule(self, rule: RiskRule):
        """添加风险规则"""
        self.risk_rules.append(rule)
        self.risk_rules.sort(key=lambda x: x.priority, reverse=True)
        self.logger.info(f"Added risk rule: {rule.name}")
    
    def check_position_risk(
        self,
        position: Position,
        current_price: float,
        market_data: Optional[pd.DataFrame] = None
    ) -> List[RiskAlert]:
        """检查单个仓位风险"""
        alerts = []
        
        # 计算未实现盈亏
        unrealized_pnl = (current_price - position.average_price) * position.quantity
        unrealized_pnl_percent = unrealized_pnl / (position.average_price * position.quantity)
        
        # 止损检查
        stop_loss_alert = self._check_stop_loss(position, current_price, unrealized_pnl_percent)
        if stop_loss_alert:
            alerts.append(stop_loss_alert)
        
        # 止盈检查
        take_profit_alert = self._check_take_profit(position, unrealized_pnl_percent)
        if take_profit_alert:
            alerts.append(take_profit_alert)
        
        # 时间止损检查
        time_stop_alert = self._check_time_stop(position)
        if time_stop_alert:
            alerts.append(time_stop_alert)
        
        # 波动率检查
        if market_data is not None:
            volatility_alert = self._check_volatility_risk(position, market_data)
            if volatility_alert:
                alerts.append(volatility_alert)
        
        return alerts
    
    def _check_stop_loss(
        self,
        position: Position,
        current_price: float,
        unrealized_pnl_percent: float
    ) -> Optional[RiskAlert]:
        """检查止损"""
        # 获取该仓位的止损配置
        stop_config = self._get_stop_loss_config(position)
        
        # 固定止损
        if stop_config.stop_type == StopLossType.FIXED:
            if unrealized_pnl_percent <= -stop_config.stop_percent:
                return RiskAlert(
                    alert_id=f"stop_loss_{position.symbol}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    level=RiskLevel.HIGH,
                    type="stop_loss",
                    message=f"{position.symbol} 触发止损，亏损 {unrealized_pnl_percent*100:.2f}%",
                    position_id=position.position_id,
                    suggested_action="close_position",
                    metadata={"stop_price": current_price, "loss_percent": unrealized_pnl_percent}
                )
        
        # 移动止损
        elif stop_config.stop_type == StopLossType.TRAILING:
            # 计算移动止损价
            if hasattr(position, 'highest_price'):
                trailing_stop_price = position.highest_price * (1 - stop_config.trailing_percent)
                if current_price <= trailing_stop_price:
                    return RiskAlert(
                        alert_id=f"trailing_stop_{position.symbol}_{datetime.now().timestamp()}",
                        timestamp=datetime.now(),
                        level=RiskLevel.HIGH,
                        type="trailing_stop",
                        message=f"{position.symbol} 触发移动止损",
                        position_id=position.position_id,
                        suggested_action="close_position",
                        metadata={"stop_price": trailing_stop_price, "highest_price": position.highest_price}
                    )
        
        return None
    
    def _check_take_profit(
        self,
        position: Position,
        unrealized_pnl_percent: float
    ) -> Optional[RiskAlert]:
        """检查止盈"""
        # 分批止盈策略
        take_profit_levels = [
            (0.05, 0.3),   # 盈利5%时平仓30%
            (0.10, 0.3),   # 盈利10%时平仓30%
            (0.15, 0.4),   # 盈利15%时平仓剩余
        ]
        
        for profit_level, close_percent in take_profit_levels:
            if unrealized_pnl_percent >= profit_level:
                # 检查是否已经在该级别止盈过
                if not hasattr(position, 'profit_taken_levels'):
                    position.profit_taken_levels = set()
                
                if profit_level not in position.profit_taken_levels:
                    position.profit_taken_levels.add(profit_level)
                    return RiskAlert(
                        alert_id=f"take_profit_{position.symbol}_{datetime.now().timestamp()}",
                        timestamp=datetime.now(),
                        level=RiskLevel.MEDIUM,
                        type="take_profit",
                        message=f"{position.symbol} 达到止盈点 {profit_level*100}%",
                        position_id=position.position_id,
                        suggested_action="reduce_position",
                        metadata={
                            "profit_level": profit_level,
                            "close_percent": close_percent,
                            "profit_percent": unrealized_pnl_percent
                        }
                    )
        
        return None
    
    def _check_time_stop(self, position: Position) -> Optional[RiskAlert]:
        """检查时间止损"""
        # 获取持仓时间
        holding_days = (datetime.now() - position.entry_time).days
        
        # 默认30天时间止损
        max_holding_days = self.config.get("max_holding_days", 30)
        
        if holding_days >= max_holding_days:
            return RiskAlert(
                alert_id=f"time_stop_{position.symbol}_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                level=RiskLevel.MEDIUM,
                type="time_stop",
                message=f"{position.symbol} 持仓超过 {max_holding_days} 天",
                position_id=position.position_id,
                suggested_action="review_position",
                metadata={"holding_days": holding_days}
            )
        
        return None
    
    def _check_volatility_risk(
        self,
        position: Position,
        market_data: pd.DataFrame
    ) -> Optional[RiskAlert]:
        """检查波动率风险"""
        # 计算当前波动率
        returns = market_data['close'].pct_change().dropna()
        current_volatility = returns.std() * np.sqrt(252)
        
        # 获取历史波动率
        symbol = position.symbol
        if symbol not in self.volatility_cache:
            # 计算30天历史波动率
            if len(market_data) >= 30:
                hist_returns = market_data['close'].iloc[-30:].pct_change().dropna()
                self.volatility_cache[symbol] = hist_returns.std() * np.sqrt(252)
            else:
                self.volatility_cache[symbol] = current_volatility
        
        historical_volatility = self.volatility_cache[symbol]
        
        # 波动率激增检查
        if current_volatility > historical_volatility * 1.5:
            return RiskAlert(
                alert_id=f"volatility_spike_{position.symbol}_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                level=RiskLevel.HIGH,
                type="volatility_spike",
                message=f"{position.symbol} 波动率激增 {current_volatility/historical_volatility:.1f}x",
                position_id=position.position_id,
                suggested_action="reduce_position",
                metadata={
                    "current_volatility": current_volatility,
                    "historical_volatility": historical_volatility
                }
            )
        
        return None
    
    def check_portfolio_risk(
        self,
        positions: List[Position],
        portfolio_value: float,
        market_data: Dict[str, pd.DataFrame]
    ) -> List[RiskAlert]:
        """检查投资组合风险"""
        alerts = []
        
        # 计算仓位分布
        position_values = {}
        total_exposure = 0
        
        for position in positions:
            if position.symbol in market_data:
                current_price = market_data[position.symbol]['close'].iloc[-1]
                position_value = position.quantity * current_price
                position_values[position.symbol] = position_value
                total_exposure += position_value
        
        # 检查总仓位
        exposure_percent = total_exposure / portfolio_value if portfolio_value > 0 else 0
        if exposure_percent > self.position_limits.max_total_exposure:
            alerts.append(RiskAlert(
                alert_id=f"high_exposure_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                level=RiskLevel.HIGH,
                type="high_exposure",
                message=f"总仓位过高 {exposure_percent*100:.1f}%",
                suggested_action="reduce_exposure",
                metadata={"exposure_percent": exposure_percent}
            ))
        
        # 检查单一仓位集中度
        for symbol, value in position_values.items():
            position_percent = value / portfolio_value if portfolio_value > 0 else 0
            if position_percent > self.position_limits.max_position_percent:
                alerts.append(RiskAlert(
                    alert_id=f"concentration_{symbol}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    level=RiskLevel.MEDIUM,
                    type="concentration",
                    message=f"{symbol} 仓位过于集中 {position_percent*100:.1f}%",
                    suggested_action="reduce_position",
                    metadata={"symbol": symbol, "position_percent": position_percent}
                ))
        
        # 检查相关性风险
        correlation_alert = self._check_correlation_risk(positions, market_data)
        if correlation_alert:
            alerts.append(correlation_alert)
        
        return alerts
    
    def _check_correlation_risk(
        self,
        positions: List[Position],
        market_data: Dict[str, pd.DataFrame]
    ) -> Optional[RiskAlert]:
        """检查相关性风险"""
        if len(positions) < 2:
            return None
        
        # 构建收益率矩阵
        symbols = [p.symbol for p in positions if p.symbol in market_data]
        if len(symbols) < 2:
            return None
        
        returns_data = {}
        for symbol in symbols:
            if symbol in market_data and len(market_data[symbol]) >= 20:
                returns = market_data[symbol]['close'].pct_change().dropna()
                returns_data[symbol] = returns.iloc[-20:]  # 最近20天
        
        if len(returns_data) < 2:
            return None
        
        # 计算相关性矩阵
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        
        # 找出高相关性对
        high_correlations = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > self.position_limits.max_correlation:
                    high_correlations.append((symbols[i], symbols[j], corr))
        
        if high_correlations:
            pairs_str = ", ".join([f"{s1}-{s2}({corr:.2f})" for s1, s2, corr in high_correlations])
            return RiskAlert(
                alert_id=f"high_correlation_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                level=RiskLevel.MEDIUM,
                type="high_correlation",
                message=f"持仓相关性过高: {pairs_str}",
                suggested_action="diversify",
                metadata={"correlations": high_correlations}
            )
        
        return None
    
    def generate_risk_orders(
        self,
        alerts: List[RiskAlert],
        positions: Dict[str, Position]
    ) -> List[Order]:
        """根据风险警报生成交易订单"""
        orders = []
        
        for alert in alerts:
            if alert.suggested_action == "close_position" and alert.position_id:
                # 生成平仓订单
                position = positions.get(alert.position_id)
                if position:
                    order = Order(
                        order_id=f"risk_{alert.alert_id}",
                        symbol=position.symbol,
                        order_type=OrderType.MARKET,
                        quantity=-position.quantity,  # 卖出
                        timestamp=datetime.now(),
                        metadata={
                            "reason": alert.type,
                            "alert_id": alert.alert_id
                        }
                    )
                    orders.append(order)
                    self.logger.info(f"Generated risk order: {order.order_id} for {alert.message}")
            
            elif alert.suggested_action == "reduce_position" and alert.position_id:
                # 生成减仓订单
                position = positions.get(alert.position_id)
                if position:
                    reduce_percent = alert.metadata.get("close_percent", 0.5)
                    reduce_quantity = int(position.quantity * reduce_percent)
                    if reduce_quantity > 0:
                        order = Order(
                            order_id=f"risk_reduce_{alert.alert_id}",
                            symbol=position.symbol,
                            order_type=OrderType.MARKET,
                            quantity=-reduce_quantity,
                            timestamp=datetime.now(),
                            metadata={
                                "reason": alert.type,
                                "alert_id": alert.alert_id,
                                "reduce_percent": reduce_percent
                            }
                        )
                        orders.append(order)
        
        return orders
    
    def update_position_tracking(self, position: Position, current_price: float):
        """更新仓位跟踪信息"""
        # 更新最高价（用于移动止损）
        if not hasattr(position, 'highest_price'):
            position.highest_price = current_price
        else:
            position.highest_price = max(position.highest_price, current_price)
        
        # 更新最低价
        if not hasattr(position, 'lowest_price'):
            position.lowest_price = current_price
        else:
            position.lowest_price = min(position.lowest_price, current_price)
    
    def _get_stop_loss_config(self, position: Position) -> StopLossConfig:
        """获取仓位的止损配置"""
        # 根据策略类型选择止损配置
        strategy_type = position.metadata.get("strategy_type", "default")
        return self.stop_loss_configs.get(strategy_type, self.stop_loss_configs["default"])
    
    def get_risk_summary(self, positions: List[Position]) -> Dict[str, Any]:
        """获取风险摘要"""
        summary = {
            "total_positions": len(positions),
            "risk_level": RiskLevel.LOW.value,
            "active_alerts": len(self.active_alerts),
            "risk_metrics": {
                "max_drawdown": 0.0,
                "value_at_risk": 0.0,
                "expected_shortfall": 0.0,
                "portfolio_volatility": 0.0
            },
            "position_distribution": {},
            "recommendations": []
        }
        
        # 计算风险指标
        if positions:
            # 这里可以添加更复杂的风险计算
            critical_alerts = [a for a in self.active_alerts if a.level == RiskLevel.CRITICAL]
            high_alerts = [a for a in self.active_alerts if a.level == RiskLevel.HIGH]
            
            if critical_alerts:
                summary["risk_level"] = RiskLevel.CRITICAL.value
            elif high_alerts:
                summary["risk_level"] = RiskLevel.HIGH.value
            elif self.active_alerts:
                summary["risk_level"] = RiskLevel.MEDIUM.value
            
            # 添加建议
            if summary["risk_level"] in [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value]:
                summary["recommendations"].append("建议减少仓位或加强风控措施")
        
        return summary