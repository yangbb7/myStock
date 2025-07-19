# -*- coding: utf-8 -*-
"""
myQuant.core - 核心模块
包含量化交易框架的核心功能组件
"""

from .backtest_engine import (BacktestEngine, Order, OrderSide, OrderStatus,
                              OrderType, Position, Trade)
from .data_manager import DataCache, DataManager, EmQuantProvider
from .exceptions import (APIException, BacktestException,
                         ConfigurationException, DataException,
                         ExceptionFactory, GlobalExceptionHandler,
                         MyQuantException, OrderException, PortfolioException,
                         RiskException, StrategyException, handle_exceptions)
from .portfolio_manager import PortfolioManager
from .portfolio_manager import Position as PortfolioPosition
from .portfolio_manager import PositionSide, RebalanceFrequency, Transaction
from .risk_manager import RiskCheckResult, RiskLevel, RiskLimit, RiskManager
from .strategy_engine import (BaseStrategy, Event, EventType, MAStrategy,
                              SignalType, StrategyEngine)

__all__ = [
    # Data Management
    "DataManager",
    "EmQuantProvider",
    "DataCache",
    # Strategy Engine
    "StrategyEngine",
    "BaseStrategy",
    "MAStrategy",
    "SignalType",
    "EventType",
    "Event",
    # Backtest Engine
    "BacktestEngine",
    "Order",
    "Trade",
    "Position",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    # Portfolio Management
    "PortfolioManager",
    "PortfolioPosition",
    "Transaction",
    "PositionSide",
    "RebalanceFrequency",
    # Risk Management
    "RiskManager",
    "RiskCheckResult",
    "RiskLevel",
    "RiskLimit",
    # Exception Handling
    "MyQuantException",
    "DataException",
    "StrategyException",
    "RiskException",
    "OrderException",
    "PortfolioException",
    "BacktestException",
    "ConfigurationException",
    "APIException",
    "ExceptionFactory",
    "GlobalExceptionHandler",
    "handle_exceptions",
]
