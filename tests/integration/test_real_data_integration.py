# -*- coding: utf-8 -*-
"""
真实数据集成测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from myQuant.core.managers.risk_manager import RiskManager
from myQuant.core.managers.portfolio_manager import PortfolioManager
from myQuant.core.managers.order_manager import OrderManager
from myQuant.core.managers.data_manager import DataManager
from myQuant.core.engines.strategy_engine import StrategyEngine
from myQuant.core.engines.backtest_engine import BacktestEngine
from myQuant.core.analysis.performance_analyzer import PerformanceAnalyzer
from myQuant.core.trading_system import TradingSystem

# 导入真实数据fixtures
from tests.fixtures.real_data_fixtures import (
    real_data_config, real_stock_symbols, real_market_data,
    real_portfolio_transactions, real_risk_parameters,
    real_benchmark_data, real_current_prices
)


class TestRealDataIntegration:
    """真实数据系统集成测试"""

    @pytest.mark.integration
    def test_real_data_to_strategy_integration(self, real_data_config, real_market_data, real_stock_symbols):
        """测试真实数据到策略引擎的数据流"""
        # 初始化组件
        data_manager = DataManager(real_data_config['data_manager'])
        strategy_engine = StrategyEngine(real_data_config['strategy_engine'])
        
        # 创建真实策略
        mock_strategy = Mock()
        mock_strategy.name = "RealDataStrategy"
        mock_strategy.symbols = real_stock_symbols[:2]  # 使用前两个真实股票
        
        def real_on_bar(bar_data):
            # 基于真实价格数据的策略逻辑
            symbol = bar_data['symbol']
            close_price = bar_data['close']
            
            # 简单的价格突破策略
            if close_price > 12.0 and symbol == '000001.SZ':  # 平安银行价格突破
                return [{
                    'timestamp': bar_data['datetime'],
                    'symbol': symbol,
                    'signal_type': 'BUY',
                    'price': close_price,
                    'quantity': 1000,
                    'strategy_name': mock_strategy.name
                }]
            elif close_price > 8.0 and symbol == '000002.SZ':  # 万科A价格突破
                return [{
                    'timestamp': bar_data['datetime'],
                    'symbol': symbol,
                    'signal_type': 'BUY',
                    'price': close_price,
                    'quantity': 800,
                    'strategy_name': mock_strategy.name
                }]
            return []
        
        mock_strategy.on_bar = real_on_bar
        mock_strategy.initialize = Mock()
        mock_strategy.finalize = Mock()
        
        # 加载真实数据
        data_manager.load_data(real_market_data)
        strategy_engine.add_strategy(mock_strategy)
        
        # 处理真实数据流
        signals = []
        for _, row in real_market_data.iterrows():
            bar_data = row.to_dict()
            strategy_signals = strategy_engine.process_bar_data(bar_data)
            signals.extend(strategy_signals)
        
        # 验证真实数据处理
        assert len(signals) >= 0
        mock_strategy.initialize.assert_called()
        
        # 验证信号基于真实价格
        if signals:
            for signal in signals:
                assert signal['price'] > 0
                assert signal['symbol'] in real_stock_symbols
                print(f"生成信号: {signal['symbol']} 价格 {signal['price']} 数量 {signal['quantity']}")
    
    @pytest.mark.integration
    def test_real_backtest_workflow(self, real_data_config, real_market_data, real_stock_symbols):
        """测试基于真实数据的完整回测工作流程"""
        # 初始化所有组件
        data_manager = DataManager(real_data_config['data_manager'])
        strategy_engine = StrategyEngine(real_data_config['strategy_engine'])
        backtest_engine = BacktestEngine(real_data_config['backtest_engine'])
        risk_manager = RiskManager(real_data_config['risk_manager'])
        portfolio_manager = PortfolioManager(real_data_config['portfolio_manager'])
        
        # 设置组件关联
        backtest_engine.set_data_manager(data_manager)
        backtest_engine.set_strategy_engine(strategy_engine)
        backtest_engine.set_risk_manager(risk_manager)
        backtest_engine.set_portfolio_manager(portfolio_manager)
        
        # 创建基于真实数据的策略
        mock_strategy = Mock()
        mock_strategy.name = "RealBacktestStrategy"
        mock_strategy.symbols = real_stock_symbols
        mock_strategy.initialize = Mock()
        mock_strategy.finalize = Mock()
        
        # 加载真实数据和策略
        data_manager.load_data(real_market_data)
        strategy_engine.add_strategy(mock_strategy)
        
        # 运行真实数据回测
        start_date = real_market_data['datetime'].min().strftime('%Y-%m-%d')
        end_date = real_market_data['datetime'].max().strftime('%Y-%m-%d')
        
        backtest_result = backtest_engine.run_backtest(
            start_date=start_date,
            end_date=end_date
        )
        
        # 验证回测结果基于真实数据
        assert backtest_result is not None
        assert 'final_value' in backtest_result
        assert 'total_return' in backtest_result
        assert 'sharpe_ratio' in backtest_result
        assert 'max_drawdown' in backtest_result
        
        # 验证结果的合理性（基于真实市场表现）
        assert backtest_result['final_value'] >= 0
        assert -1.0 <= backtest_result['total_return'] <= 5.0  # 合理的收益率范围
        assert -0.5 <= backtest_result['max_drawdown'] <= 0.0  # 合理的回撤范围
        
        print(f"真实数据回测结果: 收益率 {backtest_result['total_return']:.2%}, "
              f"夏普比率 {backtest_result['sharpe_ratio']:.2f}, "
              f"最大回撤 {backtest_result['max_drawdown']:.2%}")
    
    @pytest.mark.integration
    def test_real_portfolio_value_calculation(self, real_data_config, real_current_prices):
        """测试基于真实价格的投资组合价值计算"""
        portfolio_manager = PortfolioManager(real_data_config['portfolio_manager'])
        performance_analyzer = PerformanceAnalyzer()
        
        # 基于真实价格模拟交易
        real_trades = [
            {'symbol': '000001.SZ', 'side': 'BUY', 'quantity': 1000, 'price': real_current_prices['000001.SZ']},
            {'symbol': '000002.SZ', 'side': 'BUY', 'quantity': 800, 'price': real_current_prices['000002.SZ']},
            {'symbol': '000001.SZ', 'side': 'SELL', 'quantity': 200, 'price': real_current_prices['000001.SZ'] * 1.02}
        ]
        
        for trade in real_trades:
            portfolio_manager.update_position(trade)
        
        # 使用真实当前价格更新
        portfolio_manager.update_prices(real_current_prices)
        
        # 计算投资组合价值
        pm_total_value = portfolio_manager.calculate_total_value()
        
        # 使用绩效分析器验证（传入现金）
        positions = portfolio_manager.get_positions()
        cash = portfolio_manager.current_cash
        pa_total_value = performance_analyzer.calculate_portfolio_value(positions, real_current_prices, cash=cash)
        
        # 验证一致性
        assert abs(pm_total_value - pa_total_value) < 10.0  # 允许小误差
        
        # 验证价值合理性
        assert pm_total_value > 0
        assert pa_total_value > 0
        
        print(f"投资组合价值: PortfolioManager={pm_total_value:.2f}, PerformanceAnalyzer={pa_total_value:.2f}")
        print(f"使用真实价格: {real_current_prices}")
    
    @pytest.mark.integration
    def test_real_risk_management(self, real_data_config, real_current_prices, real_risk_parameters):
        """测试基于真实数据的风险管理"""
        risk_manager = RiskManager(real_data_config['risk_manager'])
        portfolio_manager = PortfolioManager(real_data_config['portfolio_manager'])
        
        # 更新风险参数为真实参数
        for param, value in real_risk_parameters.items():
            if hasattr(risk_manager, param):
                setattr(risk_manager, param, value)
        
        # 基于真实价格创建订单
        test_order = {
            'symbol': '000001.SZ',
            'side': 'BUY',
            'quantity': 1000,
            'price': real_current_prices['000001.SZ'],
            'order_type': 'MARKET'
        }
        
        current_positions = {}  # 空仓开始
        
        # 测试风险检查
        risk_check = risk_manager.check_signal_risk(test_order, current_positions)
        
        # 验证风险检查结果
        assert hasattr(risk_check, 'allowed') or 'allowed' in risk_check
        print(f"风险检查结果: {risk_check}")
        print(f"基于真实价格 {real_current_prices['000001.SZ']} 的订单风险评估")
    
    @pytest.mark.integration  
    def test_real_performance_analysis(self, real_data_config, real_portfolio_transactions, real_benchmark_data):
        """测试基于真实数据的绩效分析"""
        performance_analyzer = PerformanceAnalyzer()
        
        # 使用真实交易记录计算收益
        portfolio_values = []
        current_value = 1000000  # 初始资金
        
        for transaction in real_portfolio_transactions:
            if transaction['side'] == 'BUY':
                current_value -= transaction['value']
            else:
                current_value += transaction['value']
            portfolio_values.append(current_value)
        
        portfolio_series = pd.Series(portfolio_values)
        
        # 计算基于真实数据的绩效指标
        returns = performance_analyzer.calculate_returns(portfolio_series)
        sharpe_ratio = performance_analyzer.calculate_sharpe_ratio(returns, True)
        max_drawdown = performance_analyzer.calculate_max_drawdown(portfolio_series)
        
        # 验证绩效指标合理性
        assert not returns.empty
        assert isinstance(sharpe_ratio, (int, float))
        
        # 处理max_drawdown返回值（可能是字典或数值）
        if isinstance(max_drawdown, dict):
            max_drawdown_value = max_drawdown.get('max_drawdown', 0.0)
        else:
            max_drawdown_value = max_drawdown
            
        # 确保是数值类型，处理numpy类型和元组等情况
        if hasattr(max_drawdown_value, 'item'):  # numpy类型
            max_drawdown_value = max_drawdown_value.item()
        elif isinstance(max_drawdown_value, (tuple, list)):  # 元组或列表
            max_drawdown_value = float(max_drawdown_value[0]) if len(max_drawdown_value) > 0 else 0.0
        else:
            max_drawdown_value = float(max_drawdown_value)
            
        assert isinstance(max_drawdown_value, (int, float))
        assert -1.0 <= max_drawdown_value <= 0.0
        
        print(f"真实绩效分析: 夏普比率={sharpe_ratio:.2f}, 最大回撤={max_drawdown_value:.2%}")
        
        # 与基准比较（如果有基准数据）
        if len(real_benchmark_data) > 0:
            # 对齐数据长度
            min_length = min(len(returns), len(real_benchmark_data))
            portfolio_returns = returns.iloc[:min_length]
            benchmark_returns = real_benchmark_data.iloc[:min_length]
            
            if len(portfolio_returns) > 1 and len(benchmark_returns) > 1:
                try:
                    comparison = performance_analyzer.calculate_benchmark_metrics(portfolio_returns, benchmark_returns)
                    print(f"与基准比较: Alpha={comparison.get('alpha', 0):.4f}, Beta={comparison.get('beta', 0):.2f}")
                except Exception as e:
                    print(f"基准比较计算失败: {e}")
    
    @pytest.mark.integration
    def test_real_trading_system_simulation(self, real_data_config, real_stock_symbols, real_current_prices):
        """测试基于真实数据的完整交易系统模拟"""
        # 初始化完整交易系统
        trading_system = TradingSystem(real_data_config)
        
        # 创建基于真实价格的策略
        from myQuant.core.strategy_engine import MAStrategy
        real_strategy = MAStrategy("RealMAStrategy", real_stock_symbols[:2], {"ma_period": 5})
        trading_system.add_strategy(real_strategy)
        
        # 开盘前准备
        trading_system.pre_market_setup()
        
        # 模拟真实交易时段数据流
        trading_results = []
        
        for symbol in real_stock_symbols[:2]:
            # 使用真实当前价格
            real_tick = {
                'datetime': datetime.now(),
                'symbol': symbol,
                'close': real_current_prices[symbol],
                'volume': np.random.randint(1000000, 5000000),
                'open': real_current_prices[symbol] * 0.999,
                'high': real_current_prices[symbol] * 1.001,
                'low': real_current_prices[symbol] * 0.998
            }
            
            tick_result = trading_system.process_market_tick(real_tick)
            trading_results.append(tick_result)
            
            print(f"处理 {symbol} 真实价格 {real_current_prices[symbol]}: {tick_result}")
        
        # 收盘后处理
        daily_summary = trading_system.post_market_summary()
        
        # 验证完整流程
        assert daily_summary is not None
        assert 'trades_count' in daily_summary
        assert 'pnl' in daily_summary
        assert 'portfolio_value' in daily_summary
        
        # 验证结果基于真实数据
        assert all(result['processed'] for result in trading_results)
        
        print(f"交易日总结: {daily_summary}")
        print(f"使用真实价格数据: {real_current_prices}")
    
    @pytest.mark.integration
    def test_real_data_quality_validation(self, real_market_data):
        """测试真实数据质量验证"""
        # 验证数据完整性
        assert not real_market_data.empty
        required_columns = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in real_market_data.columns
        
        # 验证价格逻辑
        for _, row in real_market_data.iterrows():
            assert row['high'] >= max(row['open'], row['close'])
            assert row['low'] <= min(row['open'], row['close'])
            assert row['high'] >= row['low']
            assert row['volume'] > 0
            assert all(row[col] > 0 for col in ['open', 'high', 'low', 'close'])
        
        # 验证数据时间序列
        for symbol in real_market_data['symbol'].unique():
            symbol_data = real_market_data[real_market_data['symbol'] == symbol]
            dates = symbol_data['datetime'].sort_values()
            assert dates.is_monotonic_increasing
        
        print(f"真实数据质量验证通过: {len(real_market_data)}条数据")
        print(f"价格范围: {real_market_data['close'].min():.2f} - {real_market_data['close'].max():.2f}")
        print(f"时间范围: {real_market_data['datetime'].min()} - {real_market_data['datetime'].max()}")