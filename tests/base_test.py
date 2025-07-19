# -*- coding: utf-8 -*-
"""
测试基类和通用工具
提供测试中的通用功能和断言方法
"""

import time
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sqlite3
import tempfile
import os
from pathlib import Path


class BaseTestCase:
    """测试基类，提供通用功能"""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """自动设置测试环境"""
        self.test_start_time = time.time()
        # 设置测试隔离环境
        yield
        # 清理测试环境
        test_duration = time.time() - self.test_start_time
        if test_duration > 10:  # 测试时间超过10秒的警告
            print(f"⚠️  Test took {test_duration:.2f}s, consider optimization")
    
    def assert_price_data_valid(self, data):
        """验证价格数据的通用断言"""
        assert not data.empty, "Price data should not be empty"
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        assert len(missing_columns) == 0, f"Missing required columns: {missing_columns}"
        
        # 价格逻辑验证
        assert (data['high'] >= data['low']).all(), "High price should be >= low price"
        assert (data['high'] >= data['open']).all(), "High price should be >= open price"
        assert (data['high'] >= data['close']).all(), "High price should be >= close price"
        assert (data['low'] <= data['open']).all(), "Low price should be <= open price"
        assert (data['low'] <= data['close']).all(), "Low price should be <= close price"
        
        # 数值有效性验证
        assert (data['volume'] >= 0).all(), "Volume should be non-negative"
        assert (data['open'] > 0).all(), "Open price should be positive"
        assert (data['high'] > 0).all(), "High price should be positive"
        assert (data['low'] > 0).all(), "Low price should be positive"
        assert (data['close'] > 0).all(), "Close price should be positive"
    
    def assert_position_valid(self, position):
        """验证持仓数据的通用断言"""
        assert position is not None, "Position should not be None"
        assert 'quantity' in position, "Position should have quantity"
        assert 'avg_cost' in position, "Position should have avg_cost"
        assert position['quantity'] >= 0, "Position quantity should be non-negative"
        assert position['avg_cost'] > 0, "Position avg_cost should be positive"
    
    def assert_order_valid(self, order):
        """验证订单数据的通用断言"""
        assert order is not None, "Order should not be None"
        required_fields = ['symbol', 'side', 'quantity', 'price']
        for field in required_fields:
            assert field in order, f"Order should have {field}"
        
        assert order['side'] in ['BUY', 'SELL'], "Order side should be BUY or SELL"
        assert order['quantity'] > 0, "Order quantity should be positive"
        assert order['price'] > 0, "Order price should be positive"
    
    def assert_performance_metrics_valid(self, metrics):
        """验证绩效指标的通用断言"""
        assert metrics is not None, "Performance metrics should not be None"
        
        if 'total_return' in metrics:
            assert isinstance(metrics['total_return'], (int, float)), "Total return should be numeric"
            assert -1.0 <= metrics['total_return'] <= 10.0, "Total return should be reasonable"
        
        if 'sharpe_ratio' in metrics:
            assert isinstance(metrics['sharpe_ratio'], (int, float)), "Sharpe ratio should be numeric"
            assert not np.isnan(metrics['sharpe_ratio']), "Sharpe ratio should not be NaN"
        
        if 'max_drawdown' in metrics:
            drawdown = metrics['max_drawdown']
            if isinstance(drawdown, dict):
                drawdown = drawdown.get('max_drawdown', 0)
            assert drawdown <= 0, "Max drawdown should be non-positive"


class TestDataFactory:
    """测试数据工厂，提供确定性测试数据"""
    
    @staticmethod
    def create_deterministic_price_data(symbol='000001.SZ', days=100, base_price=15.0):
        """创建确定性价格数据，避免随机性"""
        dates = pd.date_range('2023-01-01', periods=days, freq='D')
        data = []
        
        for i, date in enumerate(dates):
            # 使用确定性价格计算，而不是随机数
            trend = base_price * 0.001 * i  # 线性趋势
            daily_vol = base_price * 0.01 * np.sin(i * 0.1)  # 周期性波动
            
            price = base_price + trend + daily_vol
            
            # 确保价格逻辑正确
            open_price = price * 0.999
            close_price = price * 1.001
            high_price = max(open_price, close_price) * 1.002
            low_price = min(open_price, close_price) * 0.998
            
            data.append({
                'datetime': date,
                'symbol': symbol,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': 1000000 + (i % 1000) * 1000,  # 确定性成交量
                'adj_close': round(close_price, 2)
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_portfolio_config(initial_capital=1000000):
        """创建标准投资组合配置"""
        return {
            'initial_capital': initial_capital,
            'base_currency': 'CNY',
            'rebalance_frequency': 'monthly',
            'commission_rate': 0.0003,
            'min_position_value': 1000,
            'max_positions': 50,
            'cash_buffer': 0.05,
            'target_weights': {
                'Finance': 0.3,
                'Technology': 0.25,
                'Consumer': 0.2,
                'Healthcare': 0.15,
                'Others': 0.1
            }
        }
    
    @staticmethod
    def create_sample_positions():
        """创建样本持仓数据"""
        return {
            '000001.SZ': {
                'quantity': 5000,
                'avg_cost': 14.5,
                'current_price': 15.0,
                'market_value': 75000,
                'unrealized_pnl': 2500,
                'sector': 'Finance',
                'weight': 0.075,
                'last_updated': datetime.now()
            },
            '000002.SZ': {
                'quantity': 3000,
                'avg_cost': 19.8,
                'current_price': 20.0,
                'market_value': 60000,
                'unrealized_pnl': 600,
                'sector': 'Finance',
                'weight': 0.06,
                'last_updated': datetime.now()
            }
        }
    
    @staticmethod
    def create_sample_orders():
        """创建样本订单数据"""
        return [
            {
                'symbol': '000001.SZ',
                'side': 'BUY',
                'quantity': 1000,
                'price': 15.0,
                'order_type': 'MARKET',
                'timestamp': datetime.now()
            },
            {
                'symbol': '000002.SZ',
                'side': 'SELL',
                'quantity': 500,
                'price': 20.0,
                'order_type': 'LIMIT',
                'timestamp': datetime.now()
            }
        ]


class MockFactory:
    """Mock对象工厂"""
    
    @staticmethod
    def create_mock_strategy(name="MockStrategy", symbols=None):
        """创建模拟策略"""
        if symbols is None:
            symbols = ["000001.SZ", "000002.SZ"]
        
        strategy = Mock()
        strategy.name = name
        strategy.symbols = symbols
        strategy.params = {"period": 20, "threshold": 0.02}
        strategy.active = True  # Make sure strategy is active
        strategy.initialize = Mock()
        strategy.finalize = Mock()
        strategy.on_bar = Mock(return_value=[])
        strategy.on_tick = Mock(return_value=[])
        return strategy
    
    @staticmethod
    def create_mock_data_provider():
        """创建模拟数据提供者"""
        provider = Mock()
        provider.get_stock_data = Mock(return_value=TestDataFactory.create_deterministic_price_data())
        provider.get_price_data = Mock(return_value=TestDataFactory.create_deterministic_price_data())
        provider.get_current_price = Mock(return_value=15.0)
        provider.get_financial_data = Mock(return_value=pd.Series({
            'symbol': '000001.SZ',
            'report_date': '2023-12-31',
            'eps': 1.2,
            'revenue': 1000000000
        }))
        return provider


class TestFixtures:
    """通用测试Fixtures"""
    
    @staticmethod
    @pytest.fixture
    def temp_db_path():
        """临时数据库路径fixture"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        yield temp_file.name
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
    
    @staticmethod
    @pytest.fixture
    def deterministic_price_data():
        """确定性价格数据fixture"""
        return TestDataFactory.create_deterministic_price_data()
    
    @staticmethod
    @pytest.fixture
    def portfolio_config():
        """投资组合配置fixture"""
        return TestDataFactory.create_portfolio_config()
    
    @staticmethod
    @pytest.fixture
    def sample_positions():
        """样本持仓fixture"""
        return TestDataFactory.create_sample_positions()
    
    @staticmethod
    @pytest.fixture
    def sample_orders():
        """样本订单fixture"""
        return TestDataFactory.create_sample_orders()
    
    @staticmethod
    @pytest.fixture
    def mock_strategy():
        """模拟策略fixture"""
        return MockFactory.create_mock_strategy()
    
    @staticmethod
    @pytest.fixture
    def mock_data_provider():
        """模拟数据提供者fixture"""
        return MockFactory.create_mock_data_provider()


class IsolatedComponentFactory:
    """隔离组件工厂，用于创建完全隔离的测试组件"""
    
    @staticmethod
    def create_isolated_data_manager(tmp_path):
        """创建完全隔离的数据管理器"""
        from myQuant.core.managers.data_manager import DataManager
        
        db_path = os.path.join(tmp_path, "test.db")
        config = {
            'db_path': str(db_path),
            'cache_size': 100,
            'use_real_data': False
        }
        
        # Mock外部数据提供者
        with patch('myQuant.infrastructure.data.providers.RealDataProvider') as mock_provider_class:
            mock_provider = MockFactory.create_mock_data_provider()
            mock_provider_class.return_value = mock_provider
            
            manager = DataManager(config)
            manager.provider = mock_provider  # 确保使用mock provider
            
            return manager
    
    @staticmethod
    def create_isolated_portfolio_manager(config=None):
        """创建隔离的投资组合管理器"""
        from myQuant.core.managers.portfolio_manager import PortfolioManager
        
        if config is None:
            config = TestDataFactory.create_portfolio_config()
        
        # Mock外部依赖
        with patch('myQuant.core.managers.portfolio_manager.DataManager'), \
             patch('myQuant.core.managers.portfolio_manager.RiskManager'):
            
            return PortfolioManager(config)


# 导出常用的测试工具
__all__ = [
    'BaseTestCase', 
    'TestDataFactory', 
    'MockFactory', 
    'TestFixtures',
    'IsolatedComponentFactory'
]