# -*- coding: utf-8 -*-
"""
myQuant - 量化交易框架
A comprehensive quantitative trading framework for Chinese stock market

主要功能模块:
- DataManager: 数据管理和技术指标计算
- StrategyEngine: 策略管理和信号生成
- BacktestEngine: 历史数据回测
- PortfolioManager: 投资组合管理
- RiskManager: 风险控制和监控
- OrderManager: 订单管理和执行
- PerformanceAnalyzer: 绩效分析
"""

__version__ = "1.0.0"
__author__ = "myQuant Team"
__email__ = "support@myquant.com"

from .core.analysis.performance_analyzer import PerformanceAnalyzer
from .core.backtest_engine import BacktestEngine, Order, Position, Trade
from .core.data_manager import DataManager, EmQuantProvider
from .core.managers.order_manager import OrderManager
from .core.portfolio_manager import PortfolioManager
from .core.risk_manager import RiskCheckResult, RiskLevel, RiskManager
from .core.strategy_engine import (BaseStrategy, MAStrategy, SignalType,
                                   StrategyEngine)
from .infrastructure.database import DatabaseManager, DatabaseConfig

__all__ = [
    # Core modules
    "DataManager",
    "StrategyEngine",
    "BacktestEngine",
    "PortfolioManager",
    "RiskManager",
    # Trading modules
    "OrderManager",
    # Analysis modules
    "PerformanceAnalyzer",
    # Data providers
    "EmQuantProvider",
    # Strategy classes
    "BaseStrategy",
    "MAStrategy",
    "SignalType",
    # Trading classes
    "Order",
    "Trade",
    "Position",
    # Risk classes
    "RiskCheckResult",
    "RiskLevel",
    # Database classes
    "DatabaseManager",
    "DatabaseConfig",
]


def get_version():
    """获取版本信息"""
    return __version__


def create_default_config():
    """创建默认配置 - 使用统一数据库配置"""
    # 获取统一数据库配置
    from .infrastructure.database.database_config import get_database_config
    db_config = get_database_config()
    
    return {
        # Top-level configuration
        "initial_capital": 1000000,
        "commission_rate": 0.0003,
        "risk_free_rate": 0.03,
        "data_manager": {
            "db_path": str(db_config.database_path),
            "cache_size": db_config.cache_size,
            "cache_ttl": db_config.cache_ttl,
        },
        "strategy_engine": {
            "max_strategies": 10,
            "event_queue_size": 1000,
            "enable_logging": True,
            "thread_pool_size": 4,
        },
        "backtest_engine": {
            "initial_capital": 1000000,
            "commission_rate": 0.0003,
            "slippage_rate": 0.001,
            "min_commission": 5.0,
            "frequency": "daily",
        },
        "portfolio_manager": {
            "initial_capital": 1000000,
            "base_currency": "CNY",
            "commission_rate": 0.0003,
            "min_commission": 5.0,
            "min_position_value": 1000,
            "max_positions": 50,
            "cash_buffer": 0.05,
            "rebalance_frequency": "monthly",
            "rebalance_threshold": 0.05,
        },
        "risk_manager": {
            "max_position_size": 0.1,
            "max_drawdown_limit": 0.2,
            "var_confidence": 0.95,
            "max_daily_trades": 100,
            "max_order_size": 0.05,
            "enable_dynamic_limits": True,
        },
        "order_manager": {
            "max_orders_per_second": 10,
            "max_pending_orders": 100,
            "order_timeout": 3600,
            "max_daily_orders": 1000,
        },
        "performance_analyzer": {
            "risk_free_rate": 0.03,
            "trading_days_per_year": 252,
            "confidence_levels": [0.95, 0.99],
            "benchmark_symbol": "000001.SH",
        },
    }


def setup_logging(level="INFO", log_file=None):
    """设置日志配置"""
    import logging
    import sys

    # 创建logger
    logger = logging.getLogger("myQuant")
    logger.setLevel(getattr(logging, level.upper()))

    # 清除现有handlers
    logger.handlers.clear()

    # 创建formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件handler (如果指定了文件)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info(f"myQuant v{__version__} 日志系统已启动")
    return logger


# 模块初始化时的默认设置
setup_logging()
