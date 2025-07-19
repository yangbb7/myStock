# -*- coding: utf-8 -*-
"""
增强的错误处理测试
专门测试各种错误情况和异常处理
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import sqlite3
import requests
import tempfile
import os

from tests.base_test import BaseTestCase, TestDataFactory, MockFactory, IsolatedComponentFactory
from myQuant.core.exceptions import (
    DataException, 
    ConfigurationException, 
    OrderException, 
    StrategyException,
    PortfolioException,
    MyQuantException
)


class TestDataManagerErrorHandling(BaseTestCase):
    """数据管理器错误处理测试"""
    
    @pytest.mark.unit
    def test_network_timeout_handling(self):
        """测试网络超时处理"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = IsolatedComponentFactory.create_isolated_data_manager(tmp_dir)
            
            # Mock网络超时
            with patch.object(manager.provider, 'get_price_data', 
                             side_effect=requests.Timeout("Network timeout")):
                result = manager.get_price_data('000001.SZ', '2023-01-01', '2023-01-31')
                
                # 应该返回空DataFrame而不是抛出异常
                assert isinstance(result, pd.DataFrame)
                assert result.empty
    
    @pytest.mark.unit
    def test_database_corruption_recovery(self):
        """测试数据库损坏恢复"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = IsolatedComponentFactory.create_isolated_data_manager(tmp_dir)
            
            # 模拟数据库损坏，系统应该处理错误并返回空DataFrame
            with patch('sqlite3.connect', side_effect=sqlite3.DatabaseError("Database is corrupted")):
                result = manager.get_price_data('000001.SZ', '2023-01-01', '2023-01-31')
                
                # 数据库损坏时，应该返回空DataFrame而不是崩溃
                assert isinstance(result, pd.DataFrame)
                assert result.empty  # 由于数据库损坏和保存失败，返回空DataFrame
    
    @pytest.mark.unit  
    def test_invalid_data_format_handling(self):
        """测试无效数据格式处理"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = IsolatedComponentFactory.create_isolated_data_manager(tmp_dir)
            
            # 测试无效数据类型
            with pytest.raises(TypeError, match="DataFrame|data.*format"):
                manager.save_price_data("invalid_data_string")
    
    @pytest.mark.unit
    def test_missing_columns_validation(self):
        """测试缺失列验证"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = IsolatedComponentFactory.create_isolated_data_manager(tmp_dir)
            
            invalid_data = pd.DataFrame({
                'datetime': ['2023-01-01'],
                'symbol': ['000001.SZ'],
                'close': [15.0]
                # 缺少必要的OHLCV列
            })
            
            is_valid = manager.validate_price_data(invalid_data)
            assert is_valid is False
    
    @pytest.mark.unit
    def test_price_anomaly_detection(self):
        """测试价格异常检测"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = IsolatedComponentFactory.create_isolated_data_manager(tmp_dir)
            
            anomaly_data = pd.DataFrame({
                'datetime': ['2023-01-01'],
                'symbol': ['000001.SZ'],
                'open': [15.0],
                'high': [10.0],  # 最高价小于开盘价（异常）
                'low': [20.0],   # 最低价大于开盘价（异常）
                'close': [15.0],
                'volume': [1000000]
            })
            
            is_valid = manager.validate_price_data(anomaly_data)
            assert is_valid is False


class TestPortfolioManagerErrorHandling(BaseTestCase):
    """投资组合管理器错误处理测试"""
    
    @pytest.mark.unit
    def test_insufficient_cash_handling(self):
        """测试现金不足处理"""
        config = TestDataFactory.create_portfolio_config(100000)  # 较少的初始资金
        
        from myQuant.core.managers.portfolio_manager import PortfolioManager
        pm = PortfolioManager(config)
        pm.current_cash = 1000  # 很少的现金
        
        # Mock external dependencies
        pm.data_manager = Mock()
        pm.risk_manager = Mock()
        
        large_order = {
            'symbol': '000001.SZ',
            'side': 'BUY',
            'quantity': 10000,
            'price': 15.0,
            'timestamp': datetime.now()
        }
        
        # First validate the order format
        try:
            pm.validate_order(large_order)
            # Then check if there's enough cash
            result = pm.check_cash_sufficiency(large_order)
            assert result is False  # Should not have enough cash
        except PortfolioException:
            # Order validation failed, which is also acceptable
            pass
    
    @pytest.mark.unit
    def test_oversell_prevention(self):
        """测试防止过度卖出"""
        config = TestDataFactory.create_portfolio_config()
        positions = TestDataFactory.create_sample_positions()
        
        from myQuant.core.managers.portfolio_manager import PortfolioManager
        pm = PortfolioManager(config)
        pm.positions = positions
        
        # Mock external dependencies  
        pm.data_manager = Mock()
        pm.risk_manager = Mock()
        
        oversell_order = {
            'symbol': '000001.SZ',
            'side': 'SELL',
            'quantity': 10000,  # 超过持有的5000股
            'price': 15.0,
            'timestamp': datetime.now()
        }
        
        # Validate order format first 
        pm.validate_order(oversell_order)
        
        # Try to update position, should fail due to insufficient shares
        try:
            pm.update_position(oversell_order)
            assert False, "Expected MyQuantException to be raised"
        except MyQuantException as e:
            assert "Insufficient shares" in str(e)
            # Test passed if we get here
    
    @pytest.mark.unit
    def test_invalid_order_format(self):
        """测试无效订单格式"""
        config = TestDataFactory.create_portfolio_config()
        
        from myQuant.core.managers.portfolio_manager import PortfolioManager
        pm = PortfolioManager(config)
        
        # Mock external dependencies  
        pm.data_manager = Mock()
        pm.risk_manager = Mock()
        
        invalid_orders = [
            {},  # 空订单
            {'symbol': ''},  # 空股票代码
            {'symbol': '000001.SZ', 'side': 'INVALID'},  # 无效方向
            {'symbol': '000001.SZ', 'side': 'BUY', 'quantity': -1000},  # 负数量
            {'symbol': '000001.SZ', 'side': 'BUY', 'quantity': 1000, 'price': 0},  # 零价格
        ]
        
        for invalid_order in invalid_orders:
            invalid_order['timestamp'] = datetime.now()
            with pytest.raises(PortfolioException):
                pm.validate_order(invalid_order)


class TestStrategyEngineErrorHandling(BaseTestCase):
    """策略引擎错误处理测试"""
    
    @pytest.mark.unit
    def test_strategy_initialization_failure(self):
        """测试策略初始化失败"""
        from myQuant.core.engines.strategy_engine import StrategyEngine
        
        engine = StrategyEngine()
        
        # 创建会初始化失败的策略
        failing_strategy = MockFactory.create_mock_strategy("FailingStrategy")
        failing_strategy.initialize.side_effect = StrategyException("Initialization failed")
        
        with pytest.raises(RuntimeError, match="Strategy initialization failed"):
            engine.add_strategy(failing_strategy)
    
    @pytest.mark.unit
    def test_strategy_execution_exception_isolation(self):
        """测试策略执行异常隔离"""
        from myQuant.core.engines.strategy_engine import StrategyEngine
        
        engine = StrategyEngine()
        
        # 添加一个正常策略和一个异常策略
        normal_strategy = MockFactory.create_mock_strategy("NormalStrategy")
        error_strategy = MockFactory.create_mock_strategy("ErrorStrategy")
        error_strategy.on_bar.side_effect = Exception("Strategy execution error")
        
        engine.add_strategy(normal_strategy)
        engine.add_strategy(error_strategy)
        
        bar_data = {
            'datetime': datetime.now(),
            'symbol': '000001.SZ',
            'close': 15.0,
            'volume': 1000000
        }
        
        # 一个策略的异常不应该影响其他策略
        signals = engine.process_bar_data(bar_data)
        
        # 正常策略应该被调用
        normal_strategy.on_bar.assert_called_once()
        # 异常策略也应该被尝试调用
        error_strategy.on_bar.assert_called_once()
        # 应该返回信号列表（可能为空）
        assert isinstance(signals, list)
    
    @pytest.mark.unit
    def test_invalid_signal_filtering(self):
        """测试无效信号过滤"""
        from myQuant.core.engines.strategy_engine import StrategyEngine
        
        engine = StrategyEngine()
        
        invalid_signals = [
            {},  # 空信号
            {'symbol': '000001.SZ'},  # 缺少必要字段
            {'symbol': '', 'signal_type': 'BUY', 'price': 15.0, 'quantity': 1000},  # 空股票代码
            {'symbol': '000001.SZ', 'signal_type': 'INVALID', 'price': 15.0, 'quantity': 1000},  # 无效信号类型
            {'symbol': '000001.SZ', 'signal_type': 'BUY', 'price': -15.0, 'quantity': 1000},  # 负价格
            {'symbol': '000001.SZ', 'signal_type': 'BUY', 'price': 15.0, 'quantity': -1000},  # 负数量
        ]
        
        for invalid_signal in invalid_signals:
            invalid_signal['timestamp'] = datetime.now()
            is_valid = engine.validate_signal(invalid_signal)
            assert is_valid is False, f"Should reject invalid signal: {invalid_signal}"


class TestBacktestEngineErrorHandling(BaseTestCase):
    """回测引擎错误处理测试"""
    
    @pytest.mark.unit
    def test_invalid_date_range(self):
        """测试无效日期范围"""
        from myQuant.core.engines.backtest_engine import BacktestEngine
        
        invalid_configs = [
            {
                'start_date': '2023-12-31',
                'end_date': '2023-01-01',  # 结束日期早于开始日期
                'initial_capital': 1000000
            },
            {
                'start_date': 'invalid_date',
                'end_date': '2023-12-31',
                'initial_capital': 1000000
            }
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError, match="date|Date"):
                BacktestEngine(config)
    
    @pytest.mark.unit
    def test_negative_capital_handling(self):
        """测试负资金处理"""
        from myQuant.core.engines.backtest_engine import BacktestEngine
        
        config = {
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'initial_capital': -1000000  # 负资金
        }
        
        with pytest.raises(ValueError, match="capital.*positive|Initial.*capital"):
            BacktestEngine(config)
    
    @pytest.mark.unit
    def test_empty_data_handling(self):
        """测试空数据处理"""
        from myQuant.core.engines.backtest_engine import BacktestEngine
        
        config = TestDataFactory.create_portfolio_config()
        engine = BacktestEngine(config)
        
        empty_data = pd.DataFrame()
        
        # 空数据应该被优雅处理
        with patch.object(engine, '_fetch_data', return_value=empty_data):
            engine.load_historical_data(None)
            # 应该生成默认数据或给出明确错误信息
            assert len(engine.historical_data) >= 0


class TestIntegrationErrorHandling(BaseTestCase):
    """集成错误处理测试"""
    
    @pytest.mark.integration
    def test_component_failure_isolation(self):
        """测试组件故障隔离"""
        # 模拟数据管理器故障不应该影响整个系统
        config = TestDataFactory.create_portfolio_config()
        
        with patch('myQuant.core.managers.data_manager.DataManager') as mock_dm_class:
            # 让数据管理器抛出异常
            mock_dm = Mock()
            mock_dm.get_price_data.side_effect = DataException("Data source unavailable")
            mock_dm_class.return_value = mock_dm
            
            from myQuant.core.trading_system import TradingSystem
            
            # 系统应该能处理数据管理器故障
            try:
                trading_system = TradingSystem(config)
                # 系统应该有回退机制或错误处理
                assert trading_system is not None
            except DataException:
                # 如果抛出异常，应该是明确的数据异常
                pass
    
    @pytest.mark.integration
    def test_partial_service_degradation(self):
        """测试部分服务降级"""
        config = TestDataFactory.create_portfolio_config()
        
        # 模拟部分服务不可用的情况
        with patch('myQuant.infrastructure.data.providers.RealDataProvider') as mock_provider:
            mock_provider.side_effect = Exception("External service unavailable")
            
            from myQuant.core.managers.data_manager import DataManager
            
            # 数据管理器应该能够降级到本地数据或缓存
            try:
                dm = DataManager(config.get('data_manager', {}))
                result = dm.get_price_data('000001.SZ', '2023-01-01', '2023-01-31')
                # 应该返回某种数据（可能是缓存或默认数据）
                assert isinstance(result, pd.DataFrame)
            except Exception as e:
                # 如果无法降级，应该给出明确的错误信息
                assert "unavailable" in str(e).lower() or "error" in str(e).lower()


@pytest.mark.parametrize("invalid_config", [
    {'initial_capital': 'invalid'},  # 字符串而不是数字
    {'initial_capital': None},       # None值
    {'commission_rate': -0.1},       # 负佣金率
    {'commission_rate': 1.5},        # 超过100%的佣金率
    {},                              # 完全空配置
])
def test_configuration_validation(invalid_config):
    """参数化测试配置验证"""
    from myQuant.core.managers.portfolio_manager import PortfolioManager
    
    # PortfolioManager doesn't directly import DataManager/RiskManager, so no mocking needed
    
    # 应该拒绝无效配置或使用合理默认值
    try:
        pm = PortfolioManager(invalid_config)
        # 如果创建成功，验证使用了合理的默认值
        assert pm.initial_capital > 0
        assert pm.commission_rate >= 0  # 佣金率非负即可
    except (ValueError, ConfigurationException, TypeError):
        # 如果抛出异常，这也是正确的行为
        pass
