# -*- coding: utf-8 -*-
"""
事件类型定义 - 定义系统中各种事件类型
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional

from .event_bus import Event, EventPriority


class EventType(Enum):
    """事件类型枚举"""
    
    # 市场数据相关
    MARKET_DATA = "market_data"
    REAL_TIME_QUOTE = "real_time_quote"
    
    # 交易相关
    ORDER_CREATED = "order_created"
    ORDER_UPDATED = "order_updated"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    TRADE_EXECUTED = "trade_executed"
    
    # 策略相关
    SIGNAL_GENERATED = "signal_generated"
    STRATEGY_STARTED = "strategy_started"
    STRATEGY_STOPPED = "strategy_stopped"
    STRATEGY_ERROR = "strategy_error"
    
    # 持仓相关
    POSITION_OPENED = "position_opened"
    POSITION_UPDATED = "position_updated"
    POSITION_CLOSED = "position_closed"
    
    # 投资组合相关
    PORTFOLIO_UPDATED = "portfolio_updated"
    PORTFOLIO_REBALANCED = "portfolio_rebalanced"
    
    # 风险管理相关
    RISK_LIMIT_BREACH = "risk_limit_breach"
    RISK_WARNING = "risk_warning"
    
    # 绩效相关
    PERFORMANCE_UPDATED = "performance_updated"
    
    # 系统相关
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_ERROR = "system_error"
    HEALTH_CHECK = "health_check"
    
    # 数据相关
    DATA_UPDATED = "data_updated"
    DATA_ERROR = "data_error"
    DATA_MISSING = "data_missing"
    
    # 回测相关
    BACKTEST_STARTED = "backtest_started"
    BACKTEST_PROGRESS = "backtest_progress"
    BACKTEST_COMPLETED = "backtest_completed"
    BACKTEST_ERROR = "backtest_error"
    
    # 告警相关
    ALERT_TRIGGERED = "alert_triggered"


@dataclass
class MarketDataEvent(Event):
    """市场数据事件"""

    def __init__(
        self, symbol: str, price_data: Dict[str, Any], source: str = "market_data"
    ):
        super().__init__(
            type="market_data",
            data={"symbol": symbol, "price_data": price_data},
            source=source,
            priority=EventPriority.HIGH,
        )

    @property
    def symbol(self) -> str:
        return self.data["symbol"]

    @property
    def price_data(self) -> Dict[str, Any]:
        return self.data["price_data"]


@dataclass
class RealTimeQuoteEvent(Event):
    """实时报价事件"""

    def __init__(
        self, symbol: str, quote_data: Dict[str, Any], source: str = "real_time_feed"
    ):
        super().__init__(
            type="real_time_quote",
            data={"symbol": symbol, "quote_data": quote_data},
            source=source,
            priority=EventPriority.CRITICAL,
        )

    @property
    def symbol(self) -> str:
        return self.data["symbol"]

    @property
    def quote_data(self) -> Dict[str, Any]:
        return self.data["quote_data"]


@dataclass
class OrderEvent(Event):
    """订单事件"""

    def __init__(
        self,
        order_id: str,
        order_data: Dict[str, Any],
        action: str,
        source: str = "order_manager",
    ):
        super().__init__(
            type="order",
            data={
                "order_id": order_id,
                "order_data": order_data,
                "action": action,  # create, update, fill, cancel, reject
            },
            source=source,
            priority=EventPriority.HIGH,
        )

    @property
    def order_id(self) -> str:
        return self.data["order_id"]

    @property
    def order_data(self) -> Dict[str, Any]:
        return self.data["order_data"]

    @property
    def action(self) -> str:
        return self.data["action"]


@dataclass
class TradeEvent(Event):
    """交易执行事件"""

    def __init__(
        self,
        trade_id: str,
        order_id: str,
        trade_data: Dict[str, Any],
        source: str = "execution_engine",
    ):
        super().__init__(
            type="trade",
            data={"trade_id": trade_id, "order_id": order_id, "trade_data": trade_data},
            source=source,
            priority=EventPriority.HIGH,
        )

    @property
    def trade_id(self) -> str:
        return self.data["trade_id"]

    @property
    def order_id(self) -> str:
        return self.data["order_id"]

    @property
    def trade_data(self) -> Dict[str, Any]:
        return self.data["trade_data"]


@dataclass
class PositionEvent(Event):
    """持仓事件"""

    def __init__(
        self,
        symbol: str,
        position_data: Dict[str, Any],
        action: str,
        source: str = "portfolio_manager",
    ):
        super().__init__(
            type="position",
            data={
                "symbol": symbol,
                "position_data": position_data,
                "action": action,  # open, update, close
            },
            source=source,
            priority=EventPriority.NORMAL,
        )

    @property
    def symbol(self) -> str:
        return self.data["symbol"]

    @property
    def position_data(self) -> Dict[str, Any]:
        return self.data["position_data"]

    @property
    def action(self) -> str:
        return self.data["action"]


@dataclass
class PortfolioEvent(Event):
    """投资组合事件"""

    def __init__(
        self,
        portfolio_id: str,
        portfolio_data: Dict[str, Any],
        action: str,
        source: str = "portfolio_manager",
    ):
        super().__init__(
            type="portfolio",
            data={
                "portfolio_id": portfolio_id,
                "portfolio_data": portfolio_data,
                "action": action,  # update, rebalance, performance
            },
            source=source,
            priority=EventPriority.NORMAL,
        )

    @property
    def portfolio_id(self) -> str:
        return self.data["portfolio_id"]

    @property
    def portfolio_data(self) -> Dict[str, Any]:
        return self.data["portfolio_data"]

    @property
    def action(self) -> str:
        return self.data["action"]


@dataclass
class StrategyEvent(Event):
    """策略事件"""

    def __init__(
        self,
        strategy_id: str,
        strategy_data: Dict[str, Any],
        action: str,
        source: str = "strategy_engine",
    ):
        super().__init__(
            type="strategy",
            data={
                "strategy_id": strategy_id,
                "strategy_data": strategy_data,
                "action": action,  # signal, start, stop, update, error
            },
            source=source,
            priority=EventPriority.HIGH,
        )

    @property
    def strategy_id(self) -> str:
        return self.data["strategy_id"]

    @property
    def strategy_data(self) -> Dict[str, Any]:
        return self.data["strategy_data"]

    @property
    def action(self) -> str:
        return self.data["action"]


@dataclass
class SignalEvent(Event):
    """交易信号事件"""

    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        signal_data: Dict[str, Any],
        source: str = "strategy_engine",
    ):
        super().__init__(
            type="signal",
            data={
                "strategy_id": strategy_id,
                "symbol": symbol,
                "signal_data": signal_data,
            },
            source=source,
            priority=EventPriority.HIGH,
        )

    @property
    def strategy_id(self) -> str:
        return self.data["strategy_id"]

    @property
    def symbol(self) -> str:
        return self.data["symbol"]

    @property
    def signal_data(self) -> Dict[str, Any]:
        return self.data["signal_data"]


@dataclass
class RiskEvent(Event):
    """风险事件"""

    def __init__(
        self,
        risk_type: str,
        risk_data: Dict[str, Any],
        severity: str = "warning",
        source: str = "risk_manager",
    ):
        super().__init__(
            type="risk",
            data={
                "risk_type": risk_type,  # position_limit, var_limit, drawdown, etc.
                "risk_data": risk_data,
                "severity": severity,  # info, warning, critical
            },
            source=source,
            priority=(
                EventPriority.CRITICAL if severity == "critical" else EventPriority.HIGH
            ),
        )

    @property
    def risk_type(self) -> str:
        return self.data["risk_type"]

    @property
    def risk_data(self) -> Dict[str, Any]:
        return self.data["risk_data"]

    @property
    def severity(self) -> str:
        return self.data["severity"]


@dataclass
class PerformanceEvent(Event):
    """性能分析事件"""

    def __init__(
        self,
        performance_data: Dict[str, Any],
        analysis_type: str = "daily",
        source: str = "performance_analyzer",
    ):
        super().__init__(
            type="performance",
            data={
                "performance_data": performance_data,
                "analysis_type": analysis_type,  # daily, weekly, monthly, real_time
            },
            source=source,
            priority=EventPriority.NORMAL,
        )

    @property
    def performance_data(self) -> Dict[str, Any]:
        return self.data["performance_data"]

    @property
    def analysis_type(self) -> str:
        return self.data["analysis_type"]


@dataclass
class SystemEvent(Event):
    """系统事件"""

    def __init__(
        self, system_data: Dict[str, Any], action: str, source: str = "system"
    ):
        super().__init__(
            type="system",
            data={
                "system_data": system_data,
                "action": action,  # startup, shutdown, error, health_check
            },
            source=source,
            priority=(
                EventPriority.CRITICAL if action == "error" else EventPriority.NORMAL
            ),
        )

    @property
    def system_data(self) -> Dict[str, Any]:
        return self.data["system_data"]

    @property
    def action(self) -> str:
        return self.data["action"]


@dataclass
class AlertEvent(Event):
    """告警事件"""

    def __init__(
        self,
        alert_data: Dict[str, Any],
        alert_type: str,
        severity: str = "warning",
        source: str = "alert_system",
    ):
        super().__init__(
            type="alert",
            data={
                "alert_data": alert_data,
                "alert_type": alert_type,  # price, volume, technical, fundamental
                "severity": severity,  # info, warning, critical
            },
            source=source,
            priority=(
                EventPriority.CRITICAL if severity == "critical" else EventPriority.HIGH
            ),
        )

    @property
    def alert_data(self) -> Dict[str, Any]:
        return self.data["alert_data"]

    @property
    def alert_type(self) -> str:
        return self.data["alert_type"]

    @property
    def severity(self) -> str:
        return self.data["severity"]


@dataclass
class BacktestEvent(Event):
    """回测事件"""

    def __init__(
        self,
        backtest_data: Dict[str, Any],
        action: str,
        source: str = "backtest_engine",
    ):
        super().__init__(
            type="backtest",
            data={
                "backtest_data": backtest_data,
                "action": action,  # start, progress, complete, error
            },
            source=source,
            priority=EventPriority.NORMAL,
        )

    @property
    def backtest_data(self) -> Dict[str, Any]:
        return self.data["backtest_data"]

    @property
    def action(self) -> str:
        return self.data["action"]


@dataclass
class DataEvent(Event):
    """数据事件"""

    def __init__(
        self, data_info: Dict[str, Any], action: str, source: str = "data_manager"
    ):
        super().__init__(
            type="data",
            data={
                "data_info": data_info,
                "action": action,  # update, missing, error, quality_check
            },
            source=source,
            priority=EventPriority.NORMAL,
        )

    @property
    def data_info(self) -> Dict[str, Any]:
        return self.data["data_info"]

    @property
    def action(self) -> str:
        return self.data["action"]


# 事件类型映射
EVENT_TYPE_MAPPING = {
    "market_data": MarketDataEvent,
    "real_time_quote": RealTimeQuoteEvent,
    "order": OrderEvent,
    "trade": TradeEvent,
    "position": PositionEvent,
    "portfolio": PortfolioEvent,
    "strategy": StrategyEvent,
    "signal": SignalEvent,
    "risk": RiskEvent,
    "performance": PerformanceEvent,
    "system": SystemEvent,
    "alert": AlertEvent,
    "backtest": BacktestEvent,
    "data": DataEvent,
}


def create_event(event_type: str, **kwargs) -> Event:
    """
    创建事件的工厂函数

    Args:
        event_type: 事件类型
        **kwargs: 事件参数

    Returns:
        Event: 事件实例
    """
    event_class = EVENT_TYPE_MAPPING.get(event_type, Event)

    if event_class == Event:
        # 通用事件
        return Event(type=event_type, data=kwargs)
    else:
        # 特定类型事件
        return event_class(**kwargs)


def get_supported_event_types() -> list:
    """获取支持的事件类型列表"""
    return list(EVENT_TYPE_MAPPING.keys())
