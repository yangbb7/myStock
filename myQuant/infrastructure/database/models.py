"""
数据库表模型

定义各个表的结构和字段，用于测试验证
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional


@dataclass
class StockTable:
    """股票基础信息表模型"""
    symbol: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market: str = "SZ"
    listing_date: Optional[str] = None
    total_shares: Optional[int] = None
    float_shares: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class KlineDailyTable:
    """K线数据表模型"""
    id: Optional[int] = None
    symbol: str = ""
    trade_date: str = ""
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    close_price: float = 0.0
    volume: int = 0
    turnover: float = 0.0


@dataclass
class RealTimeQuoteTable:
    """实时行情表模型"""
    symbol: str
    current_price: Optional[float] = None
    change_amount: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None
    turnover: Optional[float] = None
    bid_price_1: Optional[float] = None
    bid_volume_1: Optional[int] = None
    ask_price_1: Optional[float] = None
    ask_volume_1: Optional[int] = None
    last_updated: Optional[datetime] = None


@dataclass
class OrderTable:
    """订单表模型"""
    id: str
    user_id: int
    symbol: str
    order_type: str
    side: str
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    filled_quantity: int = 0
    average_fill_price: Optional[float] = None
    status: str = "PENDING"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class PositionTable:
    """持仓表模型"""
    id: Optional[int] = None
    user_id: int = 0
    symbol: str = ""
    quantity: int = 0
    average_price: float = 0.0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class TransactionTable:
    """交易记录表模型"""
    id: Optional[int] = None
    order_id: str = ""
    user_id: int = 0
    symbol: str = ""
    side: str = ""
    quantity: int = 0
    price: float = 0.0
    commission: float = 0.0
    executed_at: Optional[datetime] = None


@dataclass
class UserTable:
    """用户表模型"""
    id: Optional[int] = None
    username: str = ""
    email: Optional[str] = None
    password_hash: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_active: bool = True


@dataclass
class UserConfigTable:
    """用户配置表模型"""
    user_id: int
    risk_tolerance: float = 0.02
    max_position_size: float = 0.10
    notification_settings: Optional[str] = None
    trading_preferences: Optional[str] = None


@dataclass
class StrategyTable:
    """策略配置表模型"""
    id: Optional[int] = None
    user_id: int = 0
    name: str = ""
    type: str = ""
    parameters: str = ""
    is_active: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class AlertTable:
    """提醒设置表模型"""
    id: Optional[int] = None
    user_id: int = 0
    symbol: str = ""
    alert_type: str = ""
    condition_type: str = ""
    threshold_value: Optional[float] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    triggered_at: Optional[datetime] = None


@dataclass
class RiskMetricTable:
    """风险管理表模型"""
    id: Optional[int] = None
    user_id: int = 0
    date: str = ""
    portfolio_value: Optional[float] = None
    daily_pnl: Optional[float] = None
    max_drawdown: Optional[float] = None
    var_95: Optional[float] = None
    beta: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    calculated_at: Optional[datetime] = None