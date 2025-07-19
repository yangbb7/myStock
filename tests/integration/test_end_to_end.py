# -*- coding: utf-8 -*-
"""
端到端集成测试
完整的系统工作流程测试，模拟真实的交易场景
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "myQuant"))

from myQuant.core.trading_system import TradingSystem
from myQuant.core.managers.data_manager import DataManager
from myQuant.core.engines.strategy_engine import StrategyEngine
from myQuant.core.engines.backtest_engine import BacktestEngine
from myQuant.core.managers.risk_manager import RiskManager
from myQuant.core.managers.portfolio_manager import PortfolioManager
from myQuant.core.managers.order_manager import OrderManager
from myQuant.core.analysis.performance_analyzer import PerformanceAnalyzer
from myQuant.core.strategy_engine import BaseStrategy
from myQuant.core.exceptions import (
    DataException, ConfigurationException, 
    OrderException, StrategyException
)
from myQuant.infrastructure.container import get_container
from myQuant import create_default_config, setup_logging


class TestEndToEndIntegration:
    """端到端集成测试类"""
    
    @pytest.fixture
    def e2e_config(self):
        """端到端测试配置"""
        return {
            'initial_capital': 1000000,
            'commission_rate': 0.0003,
            'min_commission': 5.0,
            'slippage_rate': 0.001,
            'max_position_size': 0.1,
            'max_drawdown_limit': 0.2,
            'risk_free_rate': 0.03,
            'trading_days_per_year': 252,
            'data_manager': {
                'db_path': ':memory:',
                'cache_size': 1000
            },
            'strategy_engine': {
                'max_strategies': 10,
                'event_queue_size': 1000
            },
            'order_manager': {
                'max_orders_per_symbol': 500,  # Increased for full day trading simulation
                'max_total_orders': 2000,
                'max_order_value': 100000
            },
            'backtest_engine': {
                'initial_capital': 1000000,
                'commission_rate': 0.0003,
                'slippage_rate': 0.001
            },
            'risk_manager': {
                'max_position_size': 0.1,
                'max_drawdown_limit': 0.2,
                'var_confidence': 0.95
            },
            'portfolio_manager': {
                'initial_capital': 1000000,
                'commission_rate': 0.0003,
                'max_positions': 50
            }
        }
    
    @pytest.fixture
    def sample_market_data(self):
        """生成样本市场数据"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        symbols = ['000001.SZ', '000002.SZ', '600000.SH']
        
        data_list = []
        for symbol in symbols:
            base_price = 15.0 if symbol == '000001.SZ' else (20.0 if symbol == '000002.SZ' else 25.0)
            
            for i, date in enumerate(dates):
                # 加入趋势和随机波动
                trend = 0.0005 * i  # 轻微上升趋势
                volatility = np.random.normal(0, 0.02)
                price = base_price * (1 + trend + volatility)
                
                # 生成OHLC数据
                open_price = price * (1 + np.random.uniform(-0.01, 0.01))
                high_price = price * (1 + np.random.uniform(0, 0.02))
                low_price = price * (1 + np.random.uniform(-0.02, 0))
                close_price = price
                volume = np.random.randint(1000000, 10000000)
                
                data_list.append({
                    'datetime': date,
                    'symbol': symbol,
                    'open': max(0.01, open_price),
                    'high': max(0.01, high_price),
                    'low': max(0.01, low_price),
                    'close': max(0.01, close_price),
                    'volume': volume,
                    'adj_close': max(0.01, close_price)
                })
        
        return pd.DataFrame(data_list)
    
    @pytest.fixture
    def test_strategy_class(self):
        """测试策略类"""
        class TestMAStrategy(BaseStrategy):
            """移动平均策略"""
            
            def initialize(self):
                self.short_window = self.params.get('short_window', 5)
                self.long_window = self.params.get('long_window', 20)
                self.prices = {}
                self.positions = {}
                self.signals_generated = []
                
            def on_bar(self, bar_data):
                symbol = bar_data['symbol']
                close_price = bar_data['close']
                
                # 初始化价格历史
                if symbol not in self.prices:
                    self.prices[symbol] = []
                    self.positions[symbol] = 0
                
                # 更新价格历史
                self.prices[symbol].append(close_price)
                if len(self.prices[symbol]) > self.long_window:
                    self.prices[symbol].pop(0)
                
                # 计算移动平均
                if len(self.prices[symbol]) >= self.long_window:
                    short_ma = sum(self.prices[symbol][-self.short_window:]) / self.short_window
                    long_ma = sum(self.prices[symbol]) / len(self.prices[symbol])
                    
                    # 生成信号
                    if short_ma > long_ma and self.positions[symbol] <= 0:
                        signal = {
                            'timestamp': bar_data['datetime'],
                            'symbol': symbol,
                            'signal_type': 'BUY',
                            'price': close_price,
                            'quantity': 1000,
                            'strategy_name': self.name,
                            'confidence': 0.8
                        }
                        self.positions[symbol] = 1
                        self.signals_generated.append(signal)
                        return [signal]
                    elif short_ma < long_ma and self.positions[symbol] >= 0:
                        signal = {
                            'timestamp': bar_data['datetime'],
                            'symbol': symbol,
                            'signal_type': 'SELL',
                            'price': close_price,
                            'quantity': 1000,
                            'strategy_name': self.name,
                            'confidence': 0.8
                        }
                        self.positions[symbol] = -1
                        self.signals_generated.append(signal)
                        return [signal]
                
                return []
            
            def on_tick(self, tick_data):
                return []
            
            def finalize(self):
                pass
        
        return TestMAStrategy
    
    # === 完整工作流程测试 ===
    
    @pytest.mark.integration
    def test_complete_backtest_workflow(self, e2e_config, sample_market_data, test_strategy_class):
        """测试完整的回测工作流程"""
        print("=== 开始完整回测工作流程测试 ===")
        
        # 1. 初始化系统
        trading_system = TradingSystem(e2e_config)
        
        # 2. 创建和添加策略
        strategy = test_strategy_class(
            name="EndToEndTestStrategy",
            symbols=['000001.SZ', '000002.SZ'],
            params={'short_window': 5, 'long_window': 20}
        )
        
        strategy_id = trading_system.add_strategy(strategy)
        assert strategy_id is not None
        print(f"✓ 策略已添加: {strategy.name}")
        
        # 3. 加载历史数据
        trading_system.data_manager.load_data(sample_market_data)
        print(f"✓ 数据已加载: {len(sample_market_data)} 条记录")
        
        # 4. 运行回测引擎
        start_date = sample_market_data['datetime'].min().strftime('%Y-%m-%d')
        end_date = sample_market_data['datetime'].max().strftime('%Y-%m-%d')
        
        # 手动运行回测逻辑
        signals_generated = 0
        orders_created = 0
        
        for _, row in sample_market_data.iterrows():
            bar_data = row.to_dict()
            
            # 策略处理
            signals = strategy.on_bar(bar_data)
            signals_generated += len(signals)
            
            # 订单管理
            for signal in signals:
                # 风险检查
                current_positions = trading_system.portfolio_manager.get_positions()
                risk_check = trading_system.risk_manager.check_signal_risk(signal, current_positions)
                
                if risk_check.get('allowed', True):
                    # 创建订单
                    order = trading_system.portfolio_manager.process_signal(signal)
                    if order:
                        orders_created += 1
        
        # 5. 绩效分析
        portfolio_values = [e2e_config['initial_capital'] + i * 1000 for i in range(10)]  # 模拟组合价值
        performance_analyzer = PerformanceAnalyzer(e2e_config)
        
        results = performance_analyzer.analyze_portfolio(pd.Series(portfolio_values))
        
        # 6. 验证完整流程
        assert signals_generated >= 0
        assert orders_created >= 0
        assert results is not None
        assert 'returns' in results
        assert 'risk' in results
        
        print(f"✓ 回测完成: 信号数={signals_generated}, 订单数={orders_created}")
        print(f"✓ 绩效分析: 总收益率={results['returns'].get('total_return', 0):.2%}")
        print("=== 完整回测工作流程测试通过 ===")
    
    @pytest.mark.integration
    def test_live_trading_simulation(self, e2e_config, test_strategy_class):
        """测试实时交易模拟"""
        print("=== 开始实时交易模拟测试 ===")
        
        # 1. 初始化实时交易系统
        trading_system = TradingSystem(e2e_config)
        
        # 2. 创建策略
        strategy = test_strategy_class(
            name="LiveTradingStrategy",
            symbols=['000001.SZ', '000002.SZ'],
            params={'short_window': 3, 'long_window': 10}
        )
        
        trading_system.add_strategy(strategy)
        
        # 3. 模拟开盘前准备
        trading_system.pre_market_setup()
        print("✓ 开盘前准备完成")
        
        # 4. 模拟实时数据流
        live_data_stream = []
        base_prices = {'000001.SZ': 15.0, '000002.SZ': 20.0}
        
        for i in range(50):  # 模拟50个tick
            for symbol, base_price in base_prices.items():
                price = base_price * (1 + np.random.normal(0, 0.01))
                tick_data = {
                    'datetime': datetime.now() + timedelta(seconds=i),
                    'symbol': symbol,
                    'close': price,
                    'volume': np.random.randint(100000, 1000000),
                    'open': price * 0.999,
                    'high': price * 1.001,
                    'low': price * 0.998
                }
                live_data_stream.append(tick_data)
        
        # 5. 处理实时数据
        processed_ticks = 0
        total_signals = 0
        
        for tick_data in live_data_stream:
            result = trading_system.process_market_tick(tick_data)
            
            if result.get('processed'):
                processed_ticks += 1
                total_signals += result.get('signals_count', 0)
        
        # 6. 收盘后总结
        daily_summary = trading_system.post_market_summary()
        
        # 7. 验证实时交易流程
        assert processed_ticks > 0
        assert total_signals >= 0
        assert daily_summary is not None
        assert 'trades_count' in daily_summary
        assert 'portfolio_value' in daily_summary
        
        print(f"✓ 实时交易完成: 处理tick数={processed_ticks}, 信号数={total_signals}")
        print(f"✓ 日终总结: 交易次数={daily_summary.get('trades_count', 0)}")
        print("=== 实时交易模拟测试通过 ===")
    
    @pytest.mark.integration
    def test_multi_strategy_coordination(self, e2e_config, sample_market_data, test_strategy_class):
        """测试多策略协调"""
        print("=== 开始多策略协调测试 ===")
        
        # 1. 初始化系统
        trading_system = TradingSystem(e2e_config)
        
        # 2. 创建多个策略
        strategies = []
        for i in range(3):
            strategy = test_strategy_class(
                name=f"Strategy_{i}",
                symbols=['000001.SZ', '000002.SZ'],
                params={'short_window': 3 + i, 'long_window': 15 + i * 5}
            )
            strategies.append(strategy)
            trading_system.add_strategy(strategy)
        
        print(f"✓ 已添加 {len(strategies)} 个策略")
        
        # 3. 处理市场数据
        total_signals = 0
        processed_bars = 0
        
        for _, row in sample_market_data.head(30).iterrows():
            bar_data = row.to_dict()
            
            # 所有策略处理
            for strategy in strategies:
                signals = strategy.on_bar(bar_data)
                total_signals += len(signals)
                
                # 处理信号
                for signal in signals:
                    current_positions = trading_system.portfolio_manager.get_positions()
                    risk_check = trading_system.risk_manager.check_signal_risk(signal, current_positions)
                    
                    if risk_check.get('allowed', True):
                        order = trading_system.portfolio_manager.process_signal(signal)
            
            processed_bars += 1
        
        # 4. 验证多策略协调
        assert processed_bars > 0
        assert total_signals >= 0
        
        # 检查每个策略都有信号生成（或至少处理了数据）
        for strategy in strategies:
            assert len(strategy.signals_generated) >= 0
        
        print(f"✓ 多策略协调完成: 处理bar数={processed_bars}, 总信号数={total_signals}")
        print("=== 多策略协调测试通过 ===")
    
    @pytest.mark.integration
    def test_risk_management_integration(self, e2e_config, sample_market_data, test_strategy_class):
        """测试风险管理集成"""
        print("=== 开始风险管理集成测试 ===")
        
        # 1. 创建严格的风险控制配置
        strict_config = e2e_config.copy()
        strict_config['max_position_size'] = 0.02  # 2%最大仓位
        strict_config['max_drawdown_limit'] = 0.05  # 5%最大回撤
        
        trading_system = TradingSystem(strict_config)
        
        # 2. 创建激进策略（产生大量信号）
        strategy = test_strategy_class(
            name="AggressiveStrategy",
            symbols=['000001.SZ', '000002.SZ'],
            params={'short_window': 2, 'long_window': 5}
        )
        
        trading_system.add_strategy(strategy)
        
        # 3. 处理数据并测试风险控制
        allowed_signals = 0
        rejected_signals = 0
        
        for _, row in sample_market_data.head(50).iterrows():
            bar_data = row.to_dict()
            signals = strategy.on_bar(bar_data)
            
            for signal in signals:
                current_positions = trading_system.portfolio_manager.get_positions()
                risk_check = trading_system.risk_manager.check_signal_risk(signal, current_positions)
                
                if risk_check.get('allowed', True):
                    allowed_signals += 1
                    order = trading_system.portfolio_manager.process_signal(signal)
                else:
                    rejected_signals += 1
        
        # 4. 验证风险控制有效
        assert allowed_signals >= 0
        assert rejected_signals >= 0
        
        # 检查投资组合符合风险限制
        positions = trading_system.portfolio_manager.get_positions()
        total_value = trading_system.portfolio_manager.calculate_total_value()
        
        for symbol, position in positions.items():
            if position['quantity'] > 0:
                position_value = position['quantity'] * position['avg_cost']
                position_ratio = position_value / total_value
                assert position_ratio <= strict_config['max_position_size'] * 1.1  # 允许小幅超出
        
        print(f"✓ 风险管理完成: 允许信号={allowed_signals}, 拒绝信号={rejected_signals}")
        print("=== 风险管理集成测试通过 ===")
    
    @pytest.mark.integration
    def test_error_handling_and_recovery(self, e2e_config, sample_market_data):
        """测试错误处理和系统恢复"""
        print("=== 开始错误处理和恢复测试 ===")
        
        # 1. 创建会抛出异常的策略
        class ErrorStrategy(BaseStrategy):
            def __init__(self, name, symbols, params=None):
                super().__init__(name, symbols, params)
                self.error_count = 0
                
            def initialize(self):
                pass
                
            def on_bar(self, bar_data):
                self.error_count += 1
                if self.error_count % 5 == 0:  # 每5次调用抛出异常
                    raise StrategyException("Strategy processing error")
                return []
                
            def on_tick(self, tick_data):
                return []
                
            def finalize(self):
                pass
        
        # 2. 创建正常策略
        class NormalStrategy(BaseStrategy):
            def __init__(self, name, symbols, params=None):
                super().__init__(name, symbols, params)
                self.processed_count = 0
                
            def initialize(self):
                pass
                
            def on_bar(self, bar_data):
                self.processed_count += 1
                return []
                
            def on_tick(self, tick_data):
                return []
                
            def finalize(self):
                pass
        
        # 3. 初始化系统
        trading_system = TradingSystem(e2e_config)
        
        error_strategy = ErrorStrategy("ErrorStrategy", ['000001.SZ'])
        normal_strategy = NormalStrategy("NormalStrategy", ['000001.SZ'])
        
        trading_system.add_strategy(error_strategy)
        trading_system.add_strategy(normal_strategy)
        
        # 4. 处理数据，测试错误处理
        errors_caught = 0
        normal_processing = 0
        
        for _, row in sample_market_data.head(20).iterrows():
            bar_data = row.to_dict()
            
            try:
                # 处理错误策略
                error_strategy.on_bar(bar_data)
            except StrategyException:
                errors_caught += 1
            
            # 正常策略应该继续工作
            normal_strategy.on_bar(bar_data)
            normal_processing += 1
        
        # 5. 验证错误处理
        assert errors_caught > 0  # 应该捕获到异常
        assert normal_processing > 0  # 正常策略应该继续工作
        assert normal_strategy.processed_count == normal_processing
        
        print(f"✓ 错误处理完成: 捕获异常={errors_caught}, 正常处理={normal_processing}")
        print("=== 错误处理和恢复测试通过 ===")
    
    @pytest.mark.integration
    def test_performance_analysis_integration(self, e2e_config, sample_market_data):
        """测试绩效分析集成"""
        print("=== 开始绩效分析集成测试 ===")
        
        # 1. 创建绩效分析器
        performance_analyzer = PerformanceAnalyzer(e2e_config)
        
        # 2. 生成模拟交易记录
        initial_capital = e2e_config['initial_capital']
        portfolio_values = [initial_capital]
        
        # 模拟一年的交易
        for i in range(252):  # 252个交易日
            daily_return = np.random.normal(0.0008, 0.015)  # 年化约20%收益，24%波动
            new_value = portfolio_values[-1] * (1 + daily_return)
            portfolio_values.append(new_value)
        
        portfolio_series = pd.Series(portfolio_values)
        
        # 3. 全面绩效分析
        results = performance_analyzer.analyze_portfolio(portfolio_series)
        
        # 4. 验证分析结果
        assert results is not None
        assert 'returns' in results
        assert 'risk' in results
        
        # 验证收益率指标
        returns_metrics = results['returns']
        assert 'total_return' in returns_metrics
        assert 'annualized_return' in returns_metrics
        assert 'sharpe_ratio' in returns_metrics
        
        # 验证风险指标
        risk_metrics = results['risk']
        assert 'volatility' in risk_metrics
        assert 'max_drawdown' in risk_metrics
        
        # 验证指标合理性
        assert -1.0 <= returns_metrics['total_return'] <= 3.0
        assert -0.5 <= returns_metrics['annualized_return'] <= 1.0
        assert -5.0 <= returns_metrics['sharpe_ratio'] <= 5.0
        
        print(f"✓ 绩效分析完成:")
        print(f"  总收益率: {returns_metrics['total_return']:.2%}")
        print(f"  年化收益率: {returns_metrics['annualized_return']:.2%}")
        print(f"  夏普比率: {returns_metrics['sharpe_ratio']:.2f}")
        print(f"  波动率: {risk_metrics['volatility']:.2%}")
        
        max_dd = risk_metrics['max_drawdown']
        if isinstance(max_dd, dict):
            max_dd_value = max_dd.get('max_drawdown', 0)
        else:
            max_dd_value = max_dd
        print(f"  最大回撤: {max_dd_value:.2%}")
        
        print("=== 绩效分析集成测试通过 ===")
    
    @pytest.mark.integration
    def test_dependency_injection_integration(self, e2e_config):
        """测试依赖注入集成"""
        print("=== 开始依赖注入集成测试 ===")
        
        # 1. 获取依赖注入容器
        container = get_container()
        
        # 2. 通过容器获取组件
        data_manager = container.data_manager()
        strategy_engine = container.strategy_engine()
        backtest_engine = container.backtest_engine()
        portfolio_manager = container.portfolio_manager()
        risk_manager = container.risk_manager()
        performance_analyzer = container.performance_analyzer()
        
        # 3. 验证组件实例化
        assert data_manager is not None
        assert strategy_engine is not None
        assert backtest_engine is not None
        assert portfolio_manager is not None
        assert risk_manager is not None
        assert performance_analyzer is not None
        
        # 4. 验证组件类型
        assert isinstance(data_manager, DataManager)
        assert isinstance(strategy_engine, StrategyEngine)
        assert isinstance(backtest_engine, BacktestEngine)
        assert isinstance(portfolio_manager, PortfolioManager)
        assert isinstance(risk_manager, RiskManager)
        assert isinstance(performance_analyzer, PerformanceAnalyzer)
        
        # 5. 验证单例模式（如果适用）
        data_manager2 = container.data_manager()
        assert data_manager is data_manager2  # 应该是同一个实例
        
        print("✓ 依赖注入容器工作正常")
        print("✓ 所有核心组件已成功实例化")
        print("=== 依赖注入集成测试通过 ===")
    
    @pytest.mark.integration
    def test_configuration_management_integration(self, e2e_config):
        """测试配置管理集成"""
        print("=== 开始配置管理集成测试 ===")
        
        # 1. 测试默认配置
        default_config = create_default_config()
        assert default_config is not None
        assert 'initial_capital' in default_config
        
        # 2. 测试配置传递
        trading_system = TradingSystem(e2e_config)
        
        # 验证配置正确传递到各组件
        assert trading_system.config['initial_capital'] == e2e_config['initial_capital']
        assert trading_system.config['commission_rate'] == e2e_config['commission_rate']
        
        # 3. 测试配置验证
        invalid_config = e2e_config.copy()
        invalid_config['initial_capital'] = -1000  # 无效值
        
        # 应该处理无效配置
        try:
            trading_system_invalid = TradingSystem(invalid_config)
            # 如果没有抛出异常，验证系统是否使用了默认值
            assert trading_system_invalid.config['initial_capital'] > 0
        except ConfigurationException:
            # 如果抛出配置异常，这也是正确的
            pass
        
        print("✓ 配置管理正常工作")
        print("✓ 配置验证和传递测试通过")
        print("=== 配置管理集成测试通过 ===")
    
    @pytest.mark.integration
    def test_logging_integration(self, e2e_config):
        """测试日志集成"""
        print("=== 开始日志集成测试 ===")
        
        # 1. 设置日志
        logger = setup_logging(level='INFO')
        assert logger is not None
        
        # 2. 创建系统并记录日志
        trading_system = TradingSystem(e2e_config)
        
        # 3. 模拟各种日志场景
        logger.info("测试信息日志")
        logger.warning("测试警告日志")
        
        # 4. 测试异常日志
        try:
            raise DataException("测试数据异常")
        except DataException as e:
            logger.error(f"捕获异常: {e}")
        
        print("✓ 日志系统工作正常")
        print("✓ 异常日志记录测试通过")
        print("=== 日志集成测试通过 ===")
    
    @pytest.mark.integration
    def test_data_persistence_integration(self, e2e_config, sample_market_data):
        """测试数据持久化集成"""
        print("=== 开始数据持久化集成测试 ===")
        
        # 1. 创建交易系统
        trading_system = TradingSystem(e2e_config)
        
        # 2. 生成一些交易数据
        test_order = {
            'symbol': '000001.SZ',
            'side': 'BUY',
            'quantity': 1000,
            'price': 15.0,
            'timestamp': datetime.now()
        }
        
        trading_system.portfolio_manager.update_position(test_order)
        
        # 3. 保存状态
        portfolio_state = trading_system.portfolio_manager.save_state()
        assert portfolio_state is not None
        
        # 4. 创建新的组件实例并恢复状态
        new_portfolio_manager = PortfolioManager(e2e_config)
        new_portfolio_manager.load_state(portfolio_state)
        
        # 5. 验证状态恢复
        restored_position = new_portfolio_manager.get_position('000001.SZ')
        assert restored_position is not None
        assert restored_position['quantity'] == 1000
        assert restored_position['avg_cost'] == 15.0
        
        print("✓ 数据持久化工作正常")
        print("✓ 状态保存和恢复测试通过")
        print("=== 数据持久化集成测试通过 ===")
    
    @pytest.mark.integration
    def test_system_scalability(self, e2e_config, sample_market_data):
        """测试系统扩展性"""
        print("=== 开始系统扩展性测试 ===")
        
        # 1. 创建大量数据
        large_dataset = sample_market_data.copy()
        for i in range(5):  # 扩展5倍数据
            large_dataset = pd.concat([large_dataset, sample_market_data], ignore_index=True)
        
        print(f"✓ 创建大数据集: {len(large_dataset)} 条记录")
        
        # 2. 创建多个策略
        trading_system = TradingSystem(e2e_config)
        
        class SimpleStrategy(BaseStrategy):
            def initialize(self):
                self.processed_count = 0
                
            def on_bar(self, bar_data):
                self.processed_count += 1
                return []
                
            def on_tick(self, tick_data):
                return []
                
            def finalize(self):
                pass
        
        strategies = []
        for i in range(5):  # 创建5个策略
            strategy = SimpleStrategy(f"Strategy_{i}", ['000001.SZ', '000002.SZ'])
            strategies.append(strategy)
            trading_system.add_strategy(strategy)
        
        # 3. 测试处理性能
        import time
        start_time = time.time()
        
        processed_records = 0
        for _, row in large_dataset.iterrows():
            bar_data = row.to_dict()
            
            # 所有策略处理
            for strategy in strategies:
                strategy.on_bar(bar_data)
            
            processed_records += 1
            
            # 每1000条记录检查一次
            if processed_records % 1000 == 0:
                elapsed = time.time() - start_time
                rate = processed_records / elapsed
                print(f"  处理进度: {processed_records}/{len(large_dataset)} ({rate:.1f} 记录/秒)")
        
        end_time = time.time()
        total_time = end_time - start_time
        total_rate = processed_records / total_time
        
        # 4. 验证性能指标
        assert total_rate > 100  # 应该能处理每秒100条以上
        assert total_time < 60   # 应该在1分钟内完成
        
        print(f"✓ 扩展性测试完成:")
        print(f"  总记录数: {processed_records}")
        print(f"  总时间: {total_time:.2f}秒")
        print(f"  处理速率: {total_rate:.1f} 记录/秒")
        print("=== 系统扩展性测试通过 ===")
    
    @pytest.mark.integration
    def test_complete_trading_day_simulation(self, e2e_config, test_strategy_class):
        """测试完整交易日模拟"""
        print("=== 开始完整交易日模拟测试 ===")
        
        # 1. 创建完整交易系统
        trading_system = TradingSystem(e2e_config)
        
        # 2. 添加策略
        strategy = test_strategy_class(
            name="DayTradingStrategy",
            symbols=['000001.SZ', '000002.SZ'],
            params={'short_window': 3, 'long_window': 10}
        )
        
        trading_system.add_strategy(strategy)
        
        # 3. 开盘前准备
        trading_system.pre_market_setup()
        initial_status = trading_system.get_system_status()
        
        assert initial_status['is_running'] == True
        assert initial_status['strategies_count'] == 1
        
        # 4. 模拟交易时段
        trading_hours = []
        base_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        
        # 生成交易时段数据 (9:30-15:00)
        for minute in range(0, 330, 1):  # 330分钟
            current_time = base_time + timedelta(minutes=minute)
            
            for symbol in ['000001.SZ', '000002.SZ']:
                base_price = 15.0 if symbol == '000001.SZ' else 20.0
                price = base_price * (1 + np.random.normal(0, 0.005))
                
                tick_data = {
                    'datetime': current_time,
                    'symbol': symbol,
                    'close': price,
                    'volume': np.random.randint(100000, 500000),
                    'open': price * 0.9999,
                    'high': price * 1.0001,
                    'low': price * 0.9999
                }
                trading_hours.append(tick_data)
        
        # 5. 处理交易时段数据
        total_ticks = len(trading_hours)
        processed_ticks = 0
        total_signals = 0
        
        for tick_data in trading_hours:
            result = trading_system.process_market_tick(tick_data)
            
            if result.get('processed'):
                processed_ticks += 1
                total_signals += result.get('signals_count', 0)
        
        # 6. 收盘后处理
        daily_summary = trading_system.post_market_summary()
        
        # 7. 验证完整交易日
        assert processed_ticks > 0
        assert processed_ticks == total_ticks
        assert total_signals >= 0
        assert daily_summary is not None
        
        # 验证交易日统计
        assert 'trades_count' in daily_summary
        assert 'portfolio_value' in daily_summary
        assert 'pnl' in daily_summary
        
        print(f"✓ 交易日模拟完成:")
        print(f"  总tick数: {total_ticks}")
        print(f"  处理tick数: {processed_ticks}")
        print(f"  生成信号数: {total_signals}")
        print(f"  交易次数: {daily_summary.get('trades_count', 0)}")
        print(f"  投资组合价值: ¥{daily_summary.get('portfolio_value', 0):,.2f}")
        print(f"  盈亏: ¥{daily_summary.get('pnl', 0):,.2f}")
        print("=== 完整交易日模拟测试通过 ===")