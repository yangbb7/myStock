# -*- coding: utf-8 -*-
"""
TradingSystem - 交易系统主模块
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .analysis.performance_analyzer import PerformanceAnalyzer
from .backtest_engine import BacktestEngine
from .engines.execution_engine import ExecutionEngine
from .managers.data_manager import DataManager
from .managers.order_manager import OrderManager
from .managers.portfolio_manager import PortfolioManager
from .managers.risk_manager import RiskManager
from .strategy_engine import StrategyEngine
from .optimizations import CacheManager, ObjectPool, BatchProcessor, MemoryManager
from .optimizations.cache_manager import price_cache, indicator_cache
from .optimizations.object_pool import tick_pool, signal_pool
from .optimizations.batch_processor import tick_batch_processor, signal_batch_processor
from .optimizations.memory_manager import memory_manager
from .market_time_manager import market_time_manager, MarketStatus


class TradingSystem:
    """交易系统主类"""

    def __init__(self, 
                 config: Dict[str, Any],
                 data_manager: DataManager = None,
                 strategy_engine: StrategyEngine = None,
                 backtest_engine: BacktestEngine = None,
                 risk_manager: RiskManager = None,
                 portfolio_manager: PortfolioManager = None,
                 order_manager: OrderManager = None,
                 execution_engine: ExecutionEngine = None,
                 performance_analyzer: PerformanceAnalyzer = None):
        
        # 兼容旧配置方式和新配置对象
        if hasattr(config, "__dict__"):
            # 如果是配置对象，转换为字典
            self.config = config.__dict__ if config else {}
        else:
            self.config = config or {}

        # 验证和修正配置
        self._validate_and_fix_config()
        
        self.logger = logging.getLogger(__name__)

        # 初始化各个组件，提供默认配置
        default_config = {
            "initial_capital": 1000000,
            "commission_rate": 0.0003,
            "min_commission": 5.0,
        }

        # 使用依赖注入的组件或创建新的实例
        self.data_manager = data_manager or DataManager(self.config.get("data_manager", {}))
        self.strategy_engine = strategy_engine or StrategyEngine(self.config.get("strategy_engine", {}))
        self.backtest_engine = backtest_engine or BacktestEngine(
            self.config.get("backtest_engine", default_config)
        )
        self.risk_manager = risk_manager or RiskManager(self.config.get("risk_manager", {}))
        self.portfolio_manager = portfolio_manager or PortfolioManager(
            self.config.get("portfolio_manager", default_config)
        )
        self.order_manager = order_manager or OrderManager(self.config.get("order_manager", {}))
        self.execution_engine = execution_engine or ExecutionEngine(self.config.get("execution_engine", default_config))
        self.performance_analyzer = performance_analyzer or PerformanceAnalyzer()

        # 设置组件关联
        self.backtest_engine.set_data_manager(self.data_manager)
        self.backtest_engine.set_strategy_engine(self.strategy_engine)
        self.backtest_engine.set_risk_manager(self.risk_manager)
        self.backtest_engine.set_portfolio_manager(self.portfolio_manager)
        
        # 启动执行引擎
        self.execution_engine.start()
        
        # 启动性能优化组件
        memory_manager.start()
        tick_batch_processor.start()
        signal_batch_processor.start()

        # 系统状态
        self.is_running = False
        self.daily_results = []
    
    def _validate_and_fix_config(self):
        """验证和修正配置"""
        # 验证初始资金
        if 'initial_capital' in self.config:
            if self.config['initial_capital'] < 0:
                self.config['initial_capital'] = 1000000  # 修正为默认值
        
        # 验证佣金率
        if 'commission_rate' in self.config:
            if self.config['commission_rate'] < 0 or self.config['commission_rate'] > 1:
                self.config['commission_rate'] = 0.0003  # 修正为默认值
        
        # 验证风险参数
        if 'max_position_size' in self.config:
            if self.config['max_position_size'] <= 0 or self.config['max_position_size'] > 1:
                self.config['max_position_size'] = 0.1  # 修正为默认值
        
        # 递归验证嵌套配置
        for key, value in self.config.items():
            if isinstance(value, dict):
                self._validate_nested_config(key, value)
    
    def _validate_nested_config(self, key: str, nested_config: dict):
        """验证嵌套配置"""
        # 验证各组件的特定配置
        if key == 'portfolio_manager':
            if 'initial_capital' in nested_config:
                if nested_config['initial_capital'] < 0:
                    nested_config['initial_capital'] = 1000000
        
        elif key == 'risk_manager':
            if 'max_position_size' in nested_config:
                if nested_config['max_position_size'] <= 0 or nested_config['max_position_size'] > 1:
                    nested_config['max_position_size'] = 0.1
            if 'max_drawdown_limit' in nested_config:
                if nested_config['max_drawdown_limit'] <= 0 or nested_config['max_drawdown_limit'] > 1:
                    nested_config['max_drawdown_limit'] = 0.2
        
        elif key == 'backtest_engine':
            if 'initial_capital' in nested_config:
                if nested_config['initial_capital'] < 0:
                    nested_config['initial_capital'] = 1000000
            if 'commission_rate' in nested_config:
                if nested_config['commission_rate'] < 0 or nested_config['commission_rate'] > 1:
                    nested_config['commission_rate'] = 0.0003

    def add_strategy(self, strategy):
        """添加策略"""
        return self.strategy_engine.add_strategy(strategy)

    def pre_market_setup(self):
        """开盘前准备"""
        self.logger.info("Starting pre-market setup")

        # 初始化所有策略
        for strategy in self.strategy_engine.strategies.values():
            strategy.initialize()

        # 清空日内数据
        self.daily_results = []
        
        # 启动系统运行状态
        self.is_running = True

        self.logger.info("Pre-market setup completed")

    def process_market_tick(self, tick_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理市场tick数据"""
        try:
            # 检查市场时间
            tick_time = tick_data.get("datetime", datetime.now())
            market_status = market_time_manager.get_market_status(tick_time)
            
            # 在非交易时间，只更新价格但不生成信号
            if market_status not in [MarketStatus.OPEN, MarketStatus.PRE_MARKET]:
                # 数据处理（价格更新）
                self.data_manager.process_bar(tick_data)
                self._update_portfolio_prices(tick_data)
                
                return {
                    "timestamp": tick_time,
                    "symbol": tick_data.get("symbol"),
                    "signals_count": 0,
                    "processed": True,
                    "market_status": market_status.value,
                    "trading_allowed": False,
                }
            
            # 数据处理
            self.data_manager.process_bar(tick_data)
            
            # 更新投资组合价格
            self._update_portfolio_prices(tick_data)

            # 策略处理（仅在交易时间）
            signals = self.strategy_engine.process_bar_data(tick_data)

            result = {
                "timestamp": tick_time,
                "symbol": tick_data.get("symbol"),
                "signals_count": len(signals),
                "processed": True,
                "market_status": market_status.value,
                "trading_allowed": True,
            }

            # 处理信号
            for signal in signals:
                # 风险检查
                risk_check = self.risk_manager.check_signal_risk(
                    signal, self.portfolio_manager.get_current_positions()
                )

                if risk_check.allowed:
                    # 生成订单
                    order = self.portfolio_manager.create_order_from_signal(signal)

                    # 提交订单
                    order_id = self.order_manager.create_order(order)
                    self.order_manager.submit_order(order_id)
                    
                    # 执行订单
                    execution_id = self.execution_engine.submit_execution(order)
                    
                    # 等待执行完成并更新投资组合
                    self._process_execution_result(execution_id, order_id)

            return result

        except Exception as e:
            self.logger.error(f"Error processing market tick: {e}")
            return {
                "timestamp": tick_data.get("datetime", datetime.now()),
                "symbol": tick_data.get("symbol"),
                "error": str(e),
                "processed": False,
            }

    def post_market_summary(self) -> Dict[str, Any]:
        """收盘后处理"""
        self.logger.info("Starting post-market summary")

        # 计算日内统计
        total_trades = len(self.order_manager.orders)
        portfolio_value = self.portfolio_manager.calculate_total_value()
        unrealized_pnl = self.portfolio_manager.calculate_unrealized_pnl()

        summary = {
            "trades_count": total_trades,
            "portfolio_value": portfolio_value,
            "pnl": unrealized_pnl,
            "timestamp": datetime.now(),
        }

        # 结束所有策略
        for strategy in self.strategy_engine.strategies.values():
            strategy.finalize()

        self.logger.info("Post-market summary completed")
        return summary
    
    def _process_execution_result(self, execution_id: str, order_id: str):
        """处理执行结果并更新投资组合"""
        import time
        
        # 等待执行完成
        max_wait = 5.0  # 最多等待5秒
        wait_time = 0.1
        total_wait = 0
        
        while total_wait < max_wait:
            status = self.execution_engine.get_execution_status(execution_id)
            if status and status.name in ['COMPLETED', 'FAILED', 'CANCELLED']:
                break
            time.sleep(wait_time)
            total_wait += wait_time
        
        # 获取执行结果
        result = self.execution_engine.get_execution_result(execution_id)
        
        if result and result.get('success'):
            # 更新投资组合
            self.portfolio_manager.update_position_from_execution(result)
            
            # 更新订单状态
            self.order_manager.update_order_status(order_id, 'FILLED', result)
            
            self.logger.info(f"交易执行成功: {result['symbol']} {result['side']} {result['quantity']} @ {result['executed_price']:.2f}")
        else:
            # 执行失败，更新订单状态
            self.order_manager.update_order_status(order_id, 'REJECTED', result)
            self.logger.warning(f"交易执行失败: {execution_id}")
    
    def _update_portfolio_prices(self, tick_data: Dict[str, Any]):
        """更新投资组合价格"""
        symbol = tick_data.get("symbol")
        current_price = tick_data.get("close")
        
        if symbol and current_price:
            # 使用缓存提高性能
            price_cache.set_price(symbol, current_price, tick_data.get("volume", 0))
            self.portfolio_manager.update_market_price(symbol, current_price)

    def start_live_trading(self):
        """启动实时交易"""
        self.is_running = True
        self.logger.info("Live trading started")

    def stop_live_trading(self):
        """停止实时交易"""
        self.is_running = False
        self.logger.info("Live trading stopped")

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        # 获取性能优化组件状态
        cache_stats = price_cache.get_stats()
        memory_stats = memory_manager.get_memory_info()
        
        return {
            "is_running": self.is_running,
            "strategies_count": len(self.strategy_engine.strategies),
            "portfolio_value": self.portfolio_manager.calculate_total_value(),
            "orders_count": len(self.order_manager.orders),
            "timestamp": datetime.now(),
            "performance": {
                "cache_hit_rate": cache_stats.get('hit_rate', 0),
                "cache_size": cache_stats.get('size', 0),
                "memory_usage_mb": memory_stats.get('current_memory_mb', 0),
                "gc_collections": memory_stats.get('gc_collections', 0)
            }
        }
