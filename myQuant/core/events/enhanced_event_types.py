# -*- coding: utf-8 -*-
"""
增强事件类型 - 支持复杂交易逻辑的高级事件系统
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
import uuid
import pandas as pd


class EventPriority(Enum):
    """事件优先级"""
    CRITICAL = 1    # 关键事件（系统错误、风险警告）
    HIGH = 2        # 高优先级（订单执行、信号生成）
    MEDIUM = 3      # 中等优先级（数据更新、策略调整）
    LOW = 4         # 低优先级（日志记录、统计更新）


class EventCategory(Enum):
    """事件分类"""
    MARKET_DATA = "market_data"      # 市场数据事件
    SIGNAL = "signal"                # 信号事件
    ORDER = "order"                  # 订单事件
    TRADE = "trade"                  # 交易事件
    RISK = "risk"                    # 风险事件
    PORTFOLIO = "portfolio"          # 组合事件
    STRATEGY = "strategy"            # 策略事件
    SYSTEM = "system"                # 系统事件
    ALERT = "alert"                  # 警告事件
    PERFORMANCE = "performance"      # 性能事件


class EventStatus(Enum):
    """事件状态"""
    CREATED = "created"
    PENDING = "pending" 
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BaseEvent(ABC):
    """
    基础事件类 - 所有事件的基类
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    priority: EventPriority = EventPriority.MEDIUM
    category: EventCategory = EventCategory.SYSTEM
    status: EventStatus = EventStatus.CREATED
    
    # 事件元数据
    source: Optional[str] = None
    target: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 事件链相关
    parent_event_id: Optional[str] = None
    child_event_ids: List[str] = field(default_factory=list)
    
    # 处理相关
    processed_by: List[str] = field(default_factory=list)
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def mark_processing(self, processor: str) -> None:
        """标记事件正在处理"""
        self.status = EventStatus.PROCESSING
        self.processed_by.append(processor)
    
    def mark_completed(self, processing_time: Optional[float] = None) -> None:
        """标记事件处理完成"""
        self.status = EventStatus.COMPLETED
        if processing_time:
            self.processing_time = processing_time
    
    def mark_failed(self, error_message: str) -> None:
        """标记事件处理失败"""
        self.status = EventStatus.FAILED
        self.error_message = error_message
        self.retry_count += 1
    
    def can_retry(self) -> bool:
        """检查是否可以重试"""
        return self.retry_count < self.max_retries
    
    def add_child_event(self, child_event_id: str) -> None:
        """添加子事件"""
        if child_event_id not in self.child_event_ids:
            self.child_event_ids.append(child_event_id)
    
    @abstractmethod
    def validate(self) -> bool:
        """验证事件数据"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.value,
            'category': self.category.value,
            'status': self.status.value,
            'source': self.source,
            'target': self.target,
            'metadata': self.metadata,
            'parent_event_id': self.parent_event_id,
            'child_event_ids': self.child_event_ids,
            'processed_by': self.processed_by,
            'processing_time': self.processing_time,
            'error_message': self.error_message,
            'retry_count': self.retry_count
        }


@dataclass
class MarketDataEvent(BaseEvent):
    """市场数据事件"""
    symbol: str = ""
    data_type: str = ""  # 'bar', 'tick', 'quote', 'trade'
    data: Union[Dict, pd.DataFrame] = field(default_factory=dict)
    
    def __post_init__(self):
        self.category = EventCategory.MARKET_DATA
        self.priority = EventPriority.HIGH
    
    def validate(self) -> bool:
        return bool(self.symbol and self.data_type)


@dataclass
class SignalEvent(BaseEvent):
    """交易信号事件"""
    strategy_name: str = ""
    symbol: str = ""
    signal_type: str = ""  # 'BUY', 'SELL', 'HOLD'
    signal_strength: float = 0.0
    price: float = 0.0
    quantity: int = 0
    confidence: float = 0.0
    reason: str = ""
    indicators: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.category = EventCategory.SIGNAL
        self.priority = EventPriority.HIGH
    
    def validate(self) -> bool:
        return bool(self.strategy_name and self.symbol and self.signal_type)


@dataclass
class OrderEvent(BaseEvent):
    """订单事件"""
    order_id: str = ""
    symbol: str = ""
    order_type: str = ""  # 'MARKET', 'LIMIT', 'STOP'
    side: str = ""  # 'BUY', 'SELL'
    quantity: int = 0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    strategy_name: str = ""
    order_status: str = "CREATED"
    
    def __post_init__(self):
        self.category = EventCategory.ORDER
        self.priority = EventPriority.HIGH
    
    def validate(self) -> bool:
        return bool(self.order_id and self.symbol and self.order_type and self.side)


@dataclass
class TradeEvent(BaseEvent):
    """交易执行事件"""
    trade_id: str = ""
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    quantity: int = 0
    price: float = 0.0
    commission: float = 0.0
    trade_time: Optional[datetime] = None
    strategy_name: str = ""
    
    def __post_init__(self):
        self.category = EventCategory.TRADE
        self.priority = EventPriority.HIGH
        if not self.trade_time:
            self.trade_time = self.timestamp
    
    def validate(self) -> bool:
        return bool(self.trade_id and self.order_id and self.symbol)


@dataclass
class RiskEvent(BaseEvent):
    """风险事件"""
    risk_type: str = ""  # 'POSITION_LIMIT', 'LOSS_LIMIT', 'VOLATILITY', 'CONCENTRATION'
    severity: str = ""   # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    symbol: Optional[str] = None
    strategy_name: Optional[str] = None
    risk_value: float = 0.0
    threshold: float = 0.0
    message: str = ""
    suggested_action: str = ""
    
    def __post_init__(self):
        self.category = EventCategory.RISK
        if self.severity == 'CRITICAL':
            self.priority = EventPriority.CRITICAL
        elif self.severity == 'HIGH':
            self.priority = EventPriority.HIGH
        else:
            self.priority = EventPriority.MEDIUM
    
    def validate(self) -> bool:
        return bool(self.risk_type and self.severity)


@dataclass
class PortfolioEvent(BaseEvent):
    """组合事件"""
    portfolio_id: str = ""
    event_type: str = ""  # 'REBALANCE', 'UPDATE', 'PERFORMANCE'
    total_value: float = 0.0
    cash: float = 0.0
    positions: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        self.category = EventCategory.PORTFOLIO
        self.priority = EventPriority.MEDIUM
    
    def validate(self) -> bool:
        return bool(self.portfolio_id and self.event_type)


@dataclass
class StrategyEvent(BaseEvent):
    """策略事件"""
    strategy_name: str = ""
    event_type: str = ""  # 'START', 'STOP', 'PAUSE', 'RESUME', 'PARAMETER_CHANGE'
    old_state: Optional[str] = None
    new_state: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        self.category = EventCategory.STRATEGY
        self.priority = EventPriority.MEDIUM
    
    def validate(self) -> bool:
        return bool(self.strategy_name and self.event_type)


@dataclass
class SystemEvent(BaseEvent):
    """系统事件"""
    system_component: str = ""
    event_type: str = ""  # 'START', 'STOP', 'ERROR', 'WARNING', 'INFO'
    message: str = ""
    stack_trace: Optional[str] = None
    system_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.category = EventCategory.SYSTEM
        if self.event_type == 'ERROR':
            self.priority = EventPriority.CRITICAL
        elif self.event_type == 'WARNING':
            self.priority = EventPriority.HIGH
        else:
            self.priority = EventPriority.LOW
    
    def validate(self) -> bool:
        return bool(self.system_component and self.event_type)


@dataclass
class AlertEvent(BaseEvent):
    """警告事件"""
    alert_type: str = ""  # 'PRICE_ALERT', 'VOLUME_ALERT', 'TECHNICAL_ALERT'
    symbol: str = ""
    condition: str = ""
    current_value: float = 0.0
    threshold_value: float = 0.0
    alert_message: str = ""
    
    def __post_init__(self):
        self.category = EventCategory.ALERT
        self.priority = EventPriority.MEDIUM
    
    def validate(self) -> bool:
        return bool(self.alert_type and self.symbol and self.condition)


@dataclass
class PerformanceEvent(BaseEvent):
    """性能事件"""
    component: str = ""
    metric_name: str = ""
    metric_value: float = 0.0
    measurement_period: str = ""
    benchmark_value: Optional[float] = None
    performance_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.category = EventCategory.PERFORMANCE
        self.priority = EventPriority.LOW
    
    def validate(self) -> bool:
        return bool(self.component and self.metric_name)


# 复合事件类
@dataclass
class CompositeEvent(BaseEvent):
    """复合事件 - 包含多个子事件的复杂事件"""
    event_name: str = ""
    sub_events: List[BaseEvent] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)  # 子事件执行顺序
    completion_condition: str = "ALL"  # ALL, ANY, CUSTOM
    custom_condition: Optional[Callable] = None
    
    def add_sub_event(self, event: BaseEvent, position: Optional[int] = None) -> None:
        """添加子事件"""
        if position is None:
            self.sub_events.append(event)
            self.execution_order.append(event.event_id)
        else:
            self.sub_events.insert(position, event)
            self.execution_order.insert(position, event.event_id)
        
        event.parent_event_id = self.event_id
        self.add_child_event(event.event_id)
    
    def is_complete(self) -> bool:
        """检查复合事件是否完成"""
        if self.completion_condition == "ALL":
            return all(event.status == EventStatus.COMPLETED for event in self.sub_events)
        elif self.completion_condition == "ANY":
            return any(event.status == EventStatus.COMPLETED for event in self.sub_events)
        elif self.completion_condition == "CUSTOM" and self.custom_condition:
            return self.custom_condition(self.sub_events)
        return False
    
    def validate(self) -> bool:
        return bool(self.event_name and self.sub_events)


# 事件工厂类
class EventFactory:
    """事件工厂 - 创建各类事件的工厂类"""
    
    @staticmethod
    def create_market_data_event(symbol: str, data_type: str, data: Any, **kwargs) -> MarketDataEvent:
        """创建市场数据事件"""
        return MarketDataEvent(
            symbol=symbol,
            data_type=data_type,
            data=data,
            **kwargs
        )
    
    @staticmethod
    def create_signal_event(strategy_name: str, symbol: str, signal_type: str, **kwargs) -> SignalEvent:
        """创建信号事件"""
        return SignalEvent(
            strategy_name=strategy_name,
            symbol=symbol,
            signal_type=signal_type,
            **kwargs
        )
    
    @staticmethod
    def create_order_event(order_id: str, symbol: str, order_type: str, side: str, **kwargs) -> OrderEvent:
        """创建订单事件"""
        return OrderEvent(
            order_id=order_id,
            symbol=symbol,
            order_type=order_type,
            side=side,
            **kwargs
        )
    
    @staticmethod
    def create_trade_event(trade_id: str, order_id: str, symbol: str, side: str, **kwargs) -> TradeEvent:
        """创建交易事件"""
        return TradeEvent(
            trade_id=trade_id,
            order_id=order_id,
            symbol=symbol,
            side=side,
            **kwargs
        )
    
    @staticmethod
    def create_risk_event(risk_type: str, severity: str, **kwargs) -> RiskEvent:
        """创建风险事件"""
        return RiskEvent(
            risk_type=risk_type,
            severity=severity,
            **kwargs
        )
    
    @staticmethod
    def create_composite_event(event_name: str, sub_events: List[BaseEvent], **kwargs) -> CompositeEvent:
        """创建复合事件"""
        composite = CompositeEvent(event_name=event_name, **kwargs)
        for event in sub_events:
            composite.add_sub_event(event)
        return composite