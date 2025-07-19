# -*- coding: utf-8 -*-
"""
Enhanced Trading System - 增强版模块化单体交易系统
将微服务架构的优势整合到单体架构中，提供高性能、低延迟的量化交易系统
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 导入现有组件
from .analysis.performance_analyzer import PerformanceAnalyzer
from .backtest_engine import BacktestEngine
from .engines.execution_engine import ExecutionEngine
from .managers.data_manager import DataManager
from .managers.order_manager import OrderManager
from .managers.portfolio_manager import PortfolioManager
from .managers.risk_manager import RiskManager
from .strategy_engine import StrategyEngine
from .events.event_bus import EventBus
from .market_time_manager import market_time_manager, MarketStatus


class SystemModule(Enum):
    """系统模块枚举"""
    DATA = "data"
    STRATEGY = "strategy"
    EXECUTION = "execution"
    RISK = "risk"
    PORTFOLIO = "portfolio"
    ANALYTICS = "analytics"


@dataclass
class SystemConfig:
    """系统配置"""
    # 基础配置
    initial_capital: float = 1000000.0
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    
    # 性能配置
    max_concurrent_orders: int = 100
    order_timeout: float = 30.0
    data_buffer_size: int = 1000
    
    # 风险管理
    max_position_size: float = 0.1
    max_drawdown_limit: float = 0.2
    max_daily_loss: float = 0.05
    
    # 模块启用状态
    enabled_modules: List[SystemModule] = field(default_factory=lambda: list(SystemModule))
    
    # 数据库配置
    database_url: str = "sqlite:///data/myquant.db"
    enable_persistence: bool = True
    
    # 监控配置
    enable_metrics: bool = True
    metrics_port: int = 8080
    
    # 日志配置
    log_level: str = "INFO"
    log_file: str = "logs/trading_system.log"


class ModuleInterface:
    """模块接口基类"""
    
    def __init__(self, name: str, system_config: SystemConfig):
        self.name = name
        self.config = system_config
        self.logger = logging.getLogger(f"myquant.{name}")
        self.is_initialized = False
        self.metrics = {}
        
    async def initialize(self):
        """初始化模块"""
        self.logger.info(f"Initializing {self.name} module")
        self.is_initialized = True
        
    async def shutdown(self):
        """关闭模块"""
        self.logger.info(f"Shutting down {self.name} module")
        self.is_initialized = False
        
    def get_health_status(self) -> Dict[str, Any]:
        """获取模块健康状态"""
        return {
            "module": self.name,
            "initialized": self.is_initialized,
            "metrics": self.metrics,
            "timestamp": datetime.now().isoformat()
        }


class DataModule(ModuleInterface):
    """数据模块 - 整合微服务的数据处理能力"""
    
    def __init__(self, system_config: SystemConfig):
        super().__init__("data", system_config)
        self.data_manager = DataManager(system_config.__dict__)
        self.market_data_cache = {}
        self.data_subscribers = []
        
    async def initialize(self):
        await super().initialize()
        # 数据管理器不需要异步初始化，已在构造函数中初始化
        self.logger.info("Data module initialized successfully")
        
    def subscribe_to_data(self, symbol: str, callback: Callable):
        """订阅数据更新"""
        self.data_subscribers.append((symbol, callback))
        
    async def get_market_data(self, symbol: str, period: str = "1d") -> Dict[str, Any]:
        """获取市场数据"""
        try:
            # 从缓存获取
            cache_key = f"{symbol}_{period}"
            if cache_key in self.market_data_cache:
                cached_data = self.market_data_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < 60:
                    return cached_data['data']
            
            # 获取新数据
            from datetime import date, timedelta
            end_date = date.today()
            start_date = end_date - timedelta(days=30)  # 默认获取30天数据
            data = self.data_manager.get_price_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            # 更新缓存
            self.market_data_cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            self.metrics['data_requests'] = self.metrics.get('data_requests', 0) + 1
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            raise
            
    async def process_realtime_data(self, tick_data: Dict[str, Any]):
        """处理实时数据"""
        # 更新数据管理器
        self.data_manager.process_bar(tick_data)
        
        # 通知订阅者
        symbol = tick_data.get('symbol')
        for sub_symbol, callback in self.data_subscribers:
            if sub_symbol == symbol or sub_symbol == '*':
                try:
                    await callback(tick_data)
                except Exception as e:
                    self.logger.error(f"Error in data callback: {e}")


class StrategyModule(ModuleInterface):
    """策略模块 - 整合微服务的策略管理能力"""
    
    def __init__(self, system_config: SystemConfig):
        super().__init__("strategy", system_config)
        self.strategy_engine = StrategyEngine(system_config.__dict__)
        self.active_strategies = {}
        self.strategy_performance = {}
        
    async def initialize(self):
        await super().initialize()
        self.logger.info("Strategy module initialized successfully")
        
    async def add_strategy(self, strategy_name: str, strategy_config: Dict[str, Any]):
        """添加策略"""
        try:
            # 创建一个简单的策略实例
            from myQuant.core.strategy_engine import MAStrategy
            strategy = MAStrategy(
                name=strategy_name,
                symbols=strategy_config.get('symbols', ['AAPL']),
                params=strategy_config
            )
            
            # 添加到策略引擎
            strategy_id = self.strategy_engine.add_strategy(strategy)
            self.active_strategies[strategy_name] = strategy
            self.strategy_performance[strategy_name] = {
                'signals_generated': 0,
                'successful_trades': 0,
                'total_pnl': 0.0
            }
            self.logger.info(f"Strategy {strategy_name} added successfully")
            return strategy_name
            
        except Exception as e:
            self.logger.error(f"Error adding strategy {strategy_name}: {e}")
            raise
            
    async def process_market_data(self, tick_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理市场数据并生成信号"""
        signals = []
        
        for strategy_name, strategy in self.active_strategies.items():
            try:
                # 调用策略的on_bar方法
                strategy_signals = strategy.on_bar(tick_data)
                for signal in strategy_signals:
                    signal['strategy'] = strategy_name
                    signals.append(signal)
                    
                self.strategy_performance[strategy_name]['signals_generated'] += len(strategy_signals)
                
            except Exception as e:
                self.logger.error(f"Error processing data in strategy {strategy_name}: {e}")
                
        return signals
        
    def get_strategy_performance(self) -> Dict[str, Any]:
        """获取策略性能统计"""
        return self.strategy_performance


class ExecutionModule(ModuleInterface):
    """执行模块 - 整合微服务的订单执行能力"""
    
    def __init__(self, system_config: SystemConfig):
        super().__init__("execution", system_config)
        self.execution_engine = ExecutionEngine(system_config.__dict__)
        self.order_manager = OrderManager(system_config.__dict__)
        self.pending_orders = {}
        self.execution_queue = None
        self.executor = ThreadPoolExecutor(max_workers=system_config.max_concurrent_orders)
        self.execution_task = None
        
    async def initialize(self):
        await super().initialize()
        self.execution_engine.start()
        # 创建执行队列
        self.execution_queue = asyncio.Queue()
        # 启动执行任务
        self.execution_task = asyncio.create_task(self._process_execution_queue())
        self.logger.info("Execution module initialized successfully")
        
    async def shutdown(self):
        """关闭执行模块"""
        await super().shutdown()
        if self.execution_task:
            self.execution_task.cancel()
            try:
                await self.execution_task
            except asyncio.CancelledError:
                pass
        self.executor.shutdown(wait=False)
        
    async def create_order(self, signal: Dict[str, Any]) -> str:
        """创建订单"""
        try:
            order = self.order_manager.create_order_from_signal(signal)
            order_id = self.order_manager.create_order(order)
            
            # 添加到执行队列
            if self.execution_queue is not None:
                await self.execution_queue.put((order_id, order))
            
            self.metrics['orders_created'] = self.metrics.get('orders_created', 0) + 1
            return order_id
            
        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            raise
            
    async def _process_execution_queue(self):
        """处理执行队列"""
        try:
            while True:
                try:
                    # 使用超时来检查停止条件
                    order_id, order = await asyncio.wait_for(
                        self.execution_queue.get(), 
                        timeout=0.1  # 更短的超时
                    )
                    
                    # 异步执行订单
                    future = self.executor.submit(self._execute_order, order_id, order)
                    
                    # 不等待完成，继续处理下一个订单
                    self.pending_orders[order_id] = future
                    
                except asyncio.TimeoutError:
                    # 超时时检查是否应该停止
                    if not hasattr(self, 'is_initialized') or not self.is_initialized:
                        break
                    continue
                except Exception as e:
                    self.logger.error(f"Error in execution queue: {e}")
        except asyncio.CancelledError:
            # 任务被取消时正确退出
            pass
        finally:
            # 清理工作
            pass
                
    def _execute_order(self, order_id: str, order: Dict[str, Any]) -> Dict[str, Any]:
        """执行订单"""
        try:
            # 提交到执行引擎
            execution_id = self.execution_engine.submit_execution(order)
            
            # 等待执行完成
            result = self.execution_engine.wait_for_execution(execution_id, 
                                                           timeout=self.config.order_timeout)
            
            # 更新订单状态
            if result and result.get('success'):
                self.order_manager.update_order_status(order_id, 'FILLED', result)
                self.metrics['orders_filled'] = self.metrics.get('orders_filled', 0) + 1
            else:
                self.order_manager.update_order_status(order_id, 'REJECTED', result)
                self.metrics['orders_rejected'] = self.metrics.get('orders_rejected', 0) + 1
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing order {order_id}: {e}")
            self.order_manager.update_order_status(order_id, 'ERROR', {'error': str(e)})
            return {'success': False, 'error': str(e)}
            
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """获取订单状态"""
        return self.order_manager.get_order_status(order_id)


class RiskModule(ModuleInterface):
    """风险模块 - 整合微服务的风险管理能力"""
    
    def __init__(self, system_config: SystemConfig):
        super().__init__("risk", system_config)
        self.risk_manager = RiskManager(system_config.__dict__)
        self.risk_limits = {
            'max_position_size': system_config.max_position_size,
            'max_drawdown_limit': system_config.max_drawdown_limit,
            'max_daily_loss': system_config.max_daily_loss
        }
        self.daily_pnl = 0.0
        self.current_drawdown = 0.0
        
    async def initialize(self):
        await super().initialize()
        self.logger.info("Risk module initialized successfully")
        
    async def check_signal_risk(self, signal: Dict[str, Any], 
                               current_positions: Dict[str, Any]) -> Dict[str, Any]:
        """检查信号风险"""
        try:
            # 使用现有风险管理器
            risk_check = self.risk_manager.check_signal_risk(signal, current_positions)
            
            # 额外的风险检查
            if self.daily_pnl < -self.risk_limits['max_daily_loss']:
                risk_check.allowed = False
                risk_check.reason = "Daily loss limit exceeded"
                
            if self.current_drawdown > self.risk_limits['max_drawdown_limit']:
                risk_check.allowed = False
                risk_check.reason = "Drawdown limit exceeded"
                
            self.metrics['risk_checks'] = self.metrics.get('risk_checks', 0) + 1
            if not risk_check.allowed:
                self.metrics['risk_blocks'] = self.metrics.get('risk_blocks', 0) + 1
                
            return risk_check.__dict__
            
        except Exception as e:
            self.logger.error(f"Error in risk check: {e}")
            return {'allowed': False, 'reason': f'Risk check error: {str(e)}'}
            
    async def update_pnl(self, pnl_change: float):
        """更新盈亏"""
        self.daily_pnl += pnl_change
        # 更新回撤计算逻辑
        if pnl_change < 0:
            self.current_drawdown += abs(pnl_change)
        else:
            self.current_drawdown = max(0, self.current_drawdown - pnl_change * 0.5)
            
    def get_risk_metrics(self) -> Dict[str, Any]:
        """获取风险指标"""
        return {
            'daily_pnl': self.daily_pnl,
            'current_drawdown': self.current_drawdown,
            'risk_limits': self.risk_limits,
            'risk_utilization': {
                'daily_loss_ratio': abs(self.daily_pnl) / self.risk_limits['max_daily_loss'],
                'drawdown_ratio': self.current_drawdown / self.risk_limits['max_drawdown_limit']
            }
        }


class PortfolioModule(ModuleInterface):
    """投资组合模块"""
    
    def __init__(self, system_config: SystemConfig):
        super().__init__("portfolio", system_config)
        self.portfolio_manager = PortfolioManager(system_config.__dict__)
        self.current_positions = {}
        self.portfolio_value_history = []
        
    async def initialize(self):
        await super().initialize()
        self.logger.info("Portfolio module initialized successfully")
        
    async def update_position(self, execution_result: Dict[str, Any]):
        """更新持仓"""
        try:
            self.portfolio_manager.update_position_from_execution(execution_result)
            self.current_positions = self.portfolio_manager.get_current_positions()
            
            # 记录投资组合价值
            current_value = self.portfolio_manager.calculate_total_value()
            self.portfolio_value_history.append({
                'timestamp': datetime.now(),
                'value': current_value
            })
            
            self.metrics['portfolio_value'] = current_value
            
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")
            raise
            
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """获取投资组合摘要"""
        return {
            'total_value': self.portfolio_manager.calculate_total_value(),
            'cash_balance': self.portfolio_manager.current_cash,
            'positions': self.current_positions,
            'unrealized_pnl': self.portfolio_manager.calculate_unrealized_pnl(),
            'positions_count': len(self.current_positions)
        }


class AnalyticsModule(ModuleInterface):
    """分析模块"""
    
    def __init__(self, system_config: SystemConfig):
        super().__init__("analytics", system_config)
        self.performance_analyzer = PerformanceAnalyzer()
        self.trade_history = []
        self.performance_metrics = {}
        
    async def initialize(self):
        await super().initialize()
        self.logger.info("Analytics module initialized successfully")
        
    async def record_trade(self, trade_data: Dict[str, Any]):
        """记录交易"""
        self.trade_history.append({
            **trade_data,
            'timestamp': datetime.now()
        })
        
        # 更新性能指标
        await self._update_performance_metrics()
        
    async def _update_performance_metrics(self):
        """更新性能指标"""
        if len(self.trade_history) > 0:
            # 计算基本指标
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
            
            self.performance_metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'total_pnl': sum(trade.get('pnl', 0) for trade in self.trade_history),
                'avg_trade_pnl': sum(trade.get('pnl', 0) for trade in self.trade_history) / total_trades if total_trades > 0 else 0
            }
            
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            'metrics': self.performance_metrics,
            'trade_history': self.trade_history[-100:],  # 最近100笔交易
            'generated_at': datetime.now().isoformat()
        }


class EnhancedTradingSystem:
    """增强版交易系统 - 模块化单体架构"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.event_bus = EventBus()
        
        # 初始化模块
        self.modules = {}
        self._initialize_modules()
        
        # 系统状态 - 默认未启动
        self.is_running = False
        self.start_time = None
        
    def _initialize_modules(self):
        """初始化所有模块"""
        # 根据配置启用模块
        if SystemModule.DATA in self.config.enabled_modules:
            self.modules['data'] = DataModule(self.config)
            
        if SystemModule.STRATEGY in self.config.enabled_modules:
            self.modules['strategy'] = StrategyModule(self.config)
            
        if SystemModule.EXECUTION in self.config.enabled_modules:
            self.modules['execution'] = ExecutionModule(self.config)
            
        if SystemModule.RISK in self.config.enabled_modules:
            self.modules['risk'] = RiskModule(self.config)
            
        if SystemModule.PORTFOLIO in self.config.enabled_modules:
            self.modules['portfolio'] = PortfolioModule(self.config)
            
        if SystemModule.ANALYTICS in self.config.enabled_modules:
            self.modules['analytics'] = AnalyticsModule(self.config)
            
    async def initialize(self):
        """初始化系统"""
        self.logger.info("Initializing Enhanced Trading System")
        
        # 初始化所有模块
        for module_name, module in self.modules.items():
            await module.initialize()
            
        # 设置模块间的数据流
        await self._setup_data_flow()
        
        self.logger.info("Enhanced Trading System initialized successfully")
        
    async def _setup_data_flow(self):
        """设置模块间数据流"""
        # 数据模块订阅市场数据
        if 'data' in self.modules and 'strategy' in self.modules:
            self.modules['data'].subscribe_to_data('*', self._on_market_data)
            
    async def _on_market_data(self, tick_data: Dict[str, Any]):
        """处理市场数据"""
        try:
            # 1. 策略处理
            signals = []
            if 'strategy' in self.modules:
                signals = await self.modules['strategy'].process_market_data(tick_data)
                
            # 2. 风险检查和订单创建
            for signal in signals:
                if 'risk' in self.modules:
                    current_positions = {}
                    if 'portfolio' in self.modules:
                        current_positions = self.modules['portfolio'].current_positions
                        
                    risk_check = await self.modules['risk'].check_signal_risk(signal, current_positions)
                    
                    if risk_check.get('allowed', False):
                        # 3. 创建订单
                        if 'execution' in self.modules:
                            order_id = await self.modules['execution'].create_order(signal)
                            
                            # 4. 记录交易
                            if 'analytics' in self.modules:
                                await self.modules['analytics'].record_trade({
                                    'signal': signal,
                                    'order_id': order_id,
                                    'timestamp': datetime.now()
                                })
                                
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            
    async def start(self):
        """启动系统"""
        if self.is_running:
            return
            
        self.logger.info("Starting Enhanced Trading System")
        
        # 初始化系统
        await self.initialize()
        
        # 启动系统
        self.is_running = True
        self.start_time = datetime.now()
        
        self.logger.info("Enhanced Trading System started successfully")
        
    async def stop(self):
        """停止系统"""
        if not self.is_running:
            return
            
        self.logger.info("Stopping Enhanced Trading System")
        
        # 关闭所有模块
        for module_name, module in self.modules.items():
            await module.shutdown()
            
        self.is_running = False
        self.logger.info("Enhanced Trading System stopped")
        
    def process_market_tick(self, tick_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理市场数据 - 同步版本用于演示"""
        try:
            # 检查市场时间
            tick_time = tick_data.get("datetime", datetime.now())
            market_status = market_time_manager.get_market_status(tick_time)
            
            signals_count = 0
            
            # 处理策略信号
            if 'strategy' in self.modules:
                for strategy_name, strategy in self.modules['strategy'].active_strategies.items():
                    try:
                        signals = strategy.on_bar(tick_data)
                        signals_count += len(signals)
                        
                        # 更新策略性能
                        if signals:
                            self.modules['strategy'].strategy_performance[strategy_name]['signals_generated'] += len(signals)
                            
                    except Exception as e:
                        self.logger.error(f"Error processing strategy {strategy_name}: {e}")
                        
            result = {
                'timestamp': tick_time,
                'symbol': tick_data.get('symbol'),
                'processed': True,
                'signals_count': signals_count,
                'market_status': market_status.value,
                'system_uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing market tick: {e}")
            return {'error': str(e), 'processed': False}
            
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        health_status = {
            'system_running': self.is_running,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'modules': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for module_name, module in self.modules.items():
            health_status['modules'][module_name] = module.get_health_status()
            
        return health_status
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        metrics = {
            'system': {
                'running': self.is_running,
                'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                'modules_count': len(self.modules)
            }
        }
        
        # 收集各模块指标
        for module_name, module in self.modules.items():
            metrics[module_name] = module.metrics
            
        # 特殊指标
        if 'portfolio' in self.modules:
            metrics['portfolio_summary'] = self.modules['portfolio'].get_portfolio_summary()
            
        if 'risk' in self.modules:
            metrics['risk_metrics'] = self.modules['risk'].get_risk_metrics()
            
        if 'analytics' in self.modules:
            metrics['performance'] = self.modules['analytics'].performance_metrics
            
        return metrics
    
    def add_strategy(self, strategy):
        """添加策略到系统"""
        if 'strategy' not in self.modules:
            raise RuntimeError("Strategy module not enabled")
        
        # 初始化策略
        if hasattr(strategy, 'initialize'):
            strategy.initialize()
        
        # 将策略添加到策略模块
        self.modules['strategy'].active_strategies[strategy.name] = strategy
        self.modules['strategy'].strategy_performance[strategy.name] = {
            'signals_generated': 0,
            'successful_trades': 0,
            'total_pnl': 0.0
        }
        self.logger.info(f"Strategy {strategy.name} added to system")
        
    @property
    def data_manager(self):
        """获取数据管理器"""
        if 'data' in self.modules:
            return self.modules['data'].data_manager
        return None
    
    def pre_market_setup(self):
        """开盘前准备"""
        self.logger.info("Pre-market setup started")
        # 基本的开盘前准备工作
        return True
    
    def post_market_summary(self) -> Dict[str, Any]:
        """收盘后总结"""
        summary = {
            'trades_count': 0,
            'portfolio_value': self.config.initial_capital,
            'pnl': 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        # 从各模块收集信息
        if 'portfolio' in self.modules:
            portfolio_summary = self.modules['portfolio'].get_portfolio_summary()
            summary.update({
                'portfolio_value': portfolio_summary.get('total_value', self.config.initial_capital),
                'cash_balance': portfolio_summary.get('cash_balance', self.config.initial_capital),
                'positions_count': portfolio_summary.get('positions_count', 0)
            })
        
        if 'analytics' in self.modules:
            performance = self.modules['analytics'].performance_metrics
            summary.update({
                'trades_count': performance.get('total_trades', 0),
                'total_pnl': performance.get('total_pnl', 0.0),
                'win_rate': performance.get('win_rate', 0.0)
            })
        
        self.logger.info(f"Post-market summary: {summary}")
        return summary
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            'is_running': self.is_running,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'strategies_count': 0,
            'portfolio_value': self.config.initial_capital,
            'orders_count': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # 从各模块收集信息
        if 'strategy' in self.modules:
            status['strategies_count'] = len(self.modules['strategy'].active_strategies)
        
        if 'portfolio' in self.modules:
            portfolio_summary = self.modules['portfolio'].get_portfolio_summary()
            status['portfolio_value'] = portfolio_summary.get('total_value', self.config.initial_capital)
        
        if 'execution' in self.modules:
            # 简单计算订单数量
            status['orders_count'] = self.modules['execution'].metrics.get('orders_created', 0)
        
        return status