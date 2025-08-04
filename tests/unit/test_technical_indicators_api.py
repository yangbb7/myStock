"""
技术指标计算API的TDD测试

按照TDD原则，先编写完整的测试确保测试全部失败，然后实现功能代码
测试各种技术指标的计算，包括移动平均线、MACD、RSI、布林带等
"""

import pytest
import pytest_asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Optional
from unittest.mock import AsyncMock, patch, MagicMock

# 待实现的模块
from myQuant.core.analysis.technical_indicators import TechnicalIndicatorCalculator
from myQuant.core.analysis.indicator_factory import IndicatorFactory
from myQuant.interfaces.api.technical_indicators_api import TechnicalIndicatorsAPI
from myQuant.core.models.market_data import MarketData, KlineData


class TestTechnicalIndicatorCalculator:
    """技术指标计算器测试"""

    @pytest.fixture
    def sample_price_data(self):
        """样本价格数据"""
        dates = [datetime.now() - timedelta(days=i) for i in range(50, 0, -1)]
        prices = [10.0 + i * 0.1 + np.sin(i * 0.1) * 2 for i in range(50)]
        volumes = [1000000 + i * 10000 for i in range(50)]
        
        return [
            KlineData(
                symbol="000001.SZ",
                timestamp=dates[i],
                open=prices[i] - 0.1,
                high=prices[i] + 0.2,
                low=prices[i] - 0.2,
                close=prices[i],
                volume=volumes[i],
                turnover=prices[i] * volumes[i]
            ) for i in range(50)
        ]

    @pytest.fixture
    def calculator(self):
        """技术指标计算器实例"""
        return TechnicalIndicatorCalculator()

    @pytest.mark.asyncio
    async def test_calculate_simple_moving_average(self, calculator, sample_price_data):
        """测试简单移动平均线计算"""
        # Arrange
        prices = [data.close for data in sample_price_data]
        period = 5
        
        # Act
        ma_values = await calculator.calculate_sma(prices, period)
        
        # Assert
        assert len(ma_values) == len(prices)
        assert ma_values[:period-1] == [None] * (period-1)  # 前几个值应该是None
        assert ma_values[period-1] is not None
        
        # 验证计算正确性
        expected_first_ma = float(sum(prices[:period]) / period)
        assert abs(ma_values[period-1] - expected_first_ma) < 0.0001

    @pytest.mark.asyncio
    async def test_calculate_exponential_moving_average(self, calculator, sample_price_data):
        """测试指数移动平均线计算"""
        # Arrange
        prices = [data.close for data in sample_price_data]
        period = 12
        
        # Act
        ema_values = await calculator.calculate_ema(prices, period)
        
        # Assert
        assert len(ema_values) == len(prices)
        assert ema_values[0] == prices[0]  # 第一个EMA值应该等于第一个价格
        assert all(val is not None for val in ema_values)
        
        # EMA应该对最近的价格反应更敏感
        assert ema_values[-1] != ema_values[-2]

    @pytest.mark.asyncio
    async def test_calculate_macd(self, calculator, sample_price_data):
        """测试MACD计算"""
        # Arrange
        prices = [data.close for data in sample_price_data]
        fast_period = 12
        slow_period = 26
        signal_period = 9
        
        # Act
        macd_result = await calculator.calculate_macd(
            prices, fast_period, slow_period, signal_period
        )
        
        # Assert
        assert 'macd' in macd_result
        assert 'signal' in macd_result
        assert 'histogram' in macd_result
        
        macd_line = macd_result['macd']
        signal_line = macd_result['signal']
        histogram = macd_result['histogram']
        
        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)
        assert len(histogram) == len(prices)
        
        # 验证MACD线前几个值为None（需要足够数据）
        assert macd_line[slow_period-2] is None
        assert macd_line[slow_period-1] is not None

    @pytest.mark.asyncio
    async def test_calculate_rsi(self, calculator, sample_price_data):
        """测试RSI计算"""
        # Arrange
        prices = [data.close for data in sample_price_data]
        period = 14
        
        # Act
        rsi_values = await calculator.calculate_rsi(prices, period)
        
        # Assert
        assert len(rsi_values) == len(prices)
        assert rsi_values[period-1] is None  # 第一个RSI前应该是None
        assert rsi_values[period] is not None
        
        # RSI值应该在0-100之间
        valid_rsi_values = [v for v in rsi_values if v is not None]
        assert all(0 <= v <= 100 for v in valid_rsi_values)

    @pytest.mark.asyncio
    async def test_calculate_bollinger_bands(self, calculator, sample_price_data):
        """测试布林带计算"""
        # Arrange
        prices = [data.close for data in sample_price_data]
        period = 20
        std_dev = 2.0
        
        # Act
        bb_result = await calculator.calculate_bollinger_bands(prices, period, std_dev)
        
        # Assert
        assert 'upper' in bb_result
        assert 'middle' in bb_result
        assert 'lower' in bb_result
        
        upper_band = bb_result['upper']
        middle_band = bb_result['middle']
        lower_band = bb_result['lower']
        
        assert len(upper_band) == len(prices)
        assert len(middle_band) == len(prices)
        assert len(lower_band) == len(prices)
        
        # 验证关系：上轨 > 中轨 > 下轨
        for i in range(period, len(prices)):
            if all(band[i] is not None for band in [upper_band, middle_band, lower_band]):
                assert upper_band[i] > middle_band[i] > lower_band[i]

    @pytest.mark.asyncio
    async def test_calculate_stochastic_oscillator(self, calculator, sample_price_data):
        """测试随机震荡指标计算"""
        # Arrange
        period = 14
        k_smoothing = 3
        d_smoothing = 3
        
        # Act
        stoch_result = await calculator.calculate_stochastic(
            sample_price_data, period, k_smoothing, d_smoothing
        )
        
        # Assert
        assert 'k_percent' in stoch_result
        assert 'd_percent' in stoch_result
        
        k_values = stoch_result['k_percent']
        d_values = stoch_result['d_percent']
        
        assert len(k_values) == len(sample_price_data)
        assert len(d_values) == len(sample_price_data)
        
        # %K和%D值应该在0-100之间
        valid_k_values = [v for v in k_values if v is not None]
        valid_d_values = [v for v in d_values if v is not None]
        
        assert all(0 <= v <= 100 for v in valid_k_values)
        assert all(0 <= v <= 100 for v in valid_d_values)

    @pytest.mark.asyncio
    async def test_calculate_atr(self, calculator, sample_price_data):
        """测试平均真实波幅(ATR)计算"""
        # Arrange
        period = 14
        
        # Act
        atr_values = await calculator.calculate_atr(sample_price_data, period)
        
        # Assert
        assert len(atr_values) == len(sample_price_data)
        assert atr_values[period-1] is None  # 前面的值应该是None
        assert atr_values[period] is not None
        
        # ATR值应该都是正数
        valid_atr_values = [v for v in atr_values if v is not None]
        assert all(v > 0 for v in valid_atr_values)

    @pytest.mark.asyncio
    async def test_calculate_volume_indicators(self, calculator, sample_price_data):
        """测试成交量指标计算"""
        # Act
        volume_result = await calculator.calculate_volume_indicators(sample_price_data)
        
        # Assert
        assert 'obv' in volume_result  # On Balance Volume
        assert 'vwap' in volume_result  # Volume Weighted Average Price
        assert 'volume_ma' in volume_result  # Volume Moving Average
        
        obv_values = volume_result['obv']
        vwap_values = volume_result['vwap']
        volume_ma_values = volume_result['volume_ma']
        
        assert len(obv_values) == len(sample_price_data)
        assert len(vwap_values) == len(sample_price_data)
        assert len(volume_ma_values) == len(sample_price_data)

    @pytest.mark.asyncio
    async def test_batch_calculate_indicators(self, calculator, sample_price_data):
        """测试批量计算指标"""
        # Arrange
        indicators_config = {
            'sma_5': {'type': 'sma', 'period': 5},
            'sma_20': {'type': 'sma', 'period': 20},
            'ema_12': {'type': 'ema', 'period': 12},
            'rsi_14': {'type': 'rsi', 'period': 14},
            'macd': {'type': 'macd', 'fast': 12, 'slow': 26, 'signal': 9},
            'bb_20': {'type': 'bollinger_bands', 'period': 20, 'std_dev': 2}
        }
        
        # Act
        results = await calculator.batch_calculate(sample_price_data, indicators_config)
        
        # Assert
        assert len(results) == len(indicators_config)
        for indicator_name in indicators_config.keys():
            assert indicator_name in results
            
        # 验证每个指标都有正确的数据长度
        for indicator_name, values in results.items():
            if isinstance(values, dict):  # 对于MACD、布林带等复合指标
                for key, data in values.items():
                    assert len(data) == len(sample_price_data)
            else:  # 对于简单指标
                assert len(values) == len(sample_price_data)


class TestIndicatorFactory:
    """指标工厂测试"""

    @pytest.fixture
    def factory(self):
        """指标工厂实例"""
        return IndicatorFactory()

    def test_create_sma_indicator(self, factory):
        """测试创建SMA指标"""
        # Act
        sma_indicator = factory.create_indicator('sma', period=5)
        
        # Assert
        assert sma_indicator is not None
        assert sma_indicator.name == 'sma'
        assert sma_indicator.period == 5

    def test_create_ema_indicator(self, factory):
        """测试创建EMA指标"""
        # Act
        ema_indicator = factory.create_indicator('ema', period=12)
        
        # Assert
        assert ema_indicator is not None
        assert ema_indicator.name == 'ema'
        assert ema_indicator.period == 12

    def test_create_macd_indicator(self, factory):
        """测试创建MACD指标"""
        # Act
        macd_indicator = factory.create_indicator(
            'macd', fast_period=12, slow_period=26, signal_period=9
        )
        
        # Assert
        assert macd_indicator is not None
        assert macd_indicator.name == 'macd'
        assert macd_indicator.fast_period == 12
        assert macd_indicator.slow_period == 26
        assert macd_indicator.signal_period == 9

    def test_create_custom_indicator(self, factory):
        """测试创建自定义指标"""
        # Arrange
        def custom_calculation(data, period):
            return [sum(data[max(0, i-period+1):i+1]) / min(i+1, period) for i in range(len(data))]
        
        # Act
        custom_indicator = factory.register_custom_indicator(
            'custom_ma', custom_calculation, period=10
        )
        
        # Assert
        assert custom_indicator is not None
        assert custom_indicator.name == 'custom_ma'

    def test_list_available_indicators(self, factory):
        """测试列出可用指标"""
        # Act
        available = factory.list_available_indicators()
        
        # Assert
        assert 'sma' in available
        assert 'ema' in available
        assert 'macd' in available
        assert 'rsi' in available
        assert 'bollinger_bands' in available
        assert 'stochastic' in available
        assert 'atr' in available

    def test_get_indicator_parameters(self, factory):
        """测试获取指标参数"""
        # Act
        sma_params = factory.get_indicator_parameters('sma')
        macd_params = factory.get_indicator_parameters('macd')
        
        # Assert
        assert 'period' in sma_params
        assert 'fast_period' in macd_params
        assert 'slow_period' in macd_params
        assert 'signal_period' in macd_params


class TestTechnicalIndicatorsAPI:
    """技术指标API测试"""

    @pytest.fixture
    def mock_database_manager(self):
        """模拟数据库管理器"""
        return AsyncMock()

    @pytest.fixture
    def api(self, mock_database_manager):
        """技术指标API实例"""
        return TechnicalIndicatorsAPI(mock_database_manager)

    @pytest.fixture
    def sample_market_data(self):
        """样本市场数据"""
        return [
            {
                "symbol": "000001.SZ",
                "timestamp": "2024-01-01T09:30:00",
                "open": 12.50,
                "high": 12.80,
                "low": 12.30,
                "close": 12.75,
                "volume": 1000000,
                "turnover": 12650000.00
            },
            {
                "symbol": "000001.SZ",
                "timestamp": "2024-01-01T09:31:00",
                "open": 12.75,
                "high": 12.90,
                "low": 12.60,
                "close": 12.85,
                "volume": 1200000,
                "turnover": 15420000.00
            }
        ] * 25  # 50个数据点

    @pytest.mark.asyncio
    async def test_calculate_single_indicator_endpoint(self, api, sample_market_data):
        """测试单个指标计算端点"""
        # Arrange
        request_data = {
            "symbol": "000001.SZ",
            "indicator": "sma",
            "parameters": {"period": 5},
            "data": sample_market_data
        }
        
        # Act
        result = await api.calculate_indicator(request_data)
        
        # Assert
        assert result['success'] is True
        assert 'data' in result
        assert result['data']['indicator'] == 'sma'
        assert result['data']['parameters']['period'] == 5
        assert len(result['data']['values']) == len(sample_market_data)

    @pytest.mark.asyncio
    async def test_calculate_multiple_indicators_endpoint(self, api, sample_market_data):
        """测试多个指标计算端点"""
        # Arrange
        request_data = {
            "symbol": "000001.SZ",
            "indicators": {
                "sma_5": {"type": "sma", "period": 5},
                "sma_20": {"type": "sma", "period": 20},
                "rsi_14": {"type": "rsi", "period": 14}
            },
            "data": sample_market_data
        }
        
        # Act
        result = await api.calculate_multiple_indicators(request_data)
        
        # Assert
        assert result['success'] is True
        assert 'data' in result
        assert len(result['data']['indicators']) == 3
        assert 'sma_5' in result['data']['indicators']
        assert 'sma_20' in result['data']['indicators']
        assert 'rsi_14' in result['data']['indicators']

    @pytest.mark.asyncio
    async def test_get_indicator_from_database(self, api, mock_database_manager):
        """测试从数据库获取指标数据"""
        # Arrange
        mock_database_manager.fetch_all.return_value = [
            {"symbol": "000001.SZ", "trade_date": "2024-01-01", "open_price": 12.50, "high_price": 12.80, "low_price": 12.30, "close_price": 12.75, "volume": 1000000, "turnover": 12650000.00},
            {"symbol": "000001.SZ", "trade_date": "2024-01-02", "open_price": 12.75, "high_price": 12.90, "low_price": 12.60, "close_price": 12.85, "volume": 1200000, "turnover": 15420000.00}
        ]
        
        request_data = {
            "symbol": "000001.SZ",
            "indicator": "sma",
            "parameters": {"period": 5},
            "start_date": "2024-01-01",
            "end_date": "2024-01-31"
        }
        
        # Act
        result = await api.get_indicator_from_database(request_data)
        
        # Assert
        assert result['success'] is True
        mock_database_manager.fetch_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_indicator_request(self, api):
        """测试指标请求验证"""
        # Arrange - 有效请求
        valid_request = {
            "symbol": "000001.SZ",
            "indicator": "sma",
            "parameters": {"period": 5}
        }
        
        # Act
        is_valid, errors = await api.validate_request(valid_request)
        
        # Assert
        assert is_valid is True
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_validate_invalid_indicator_request(self, api):
        """测试无效指标请求验证"""
        # Arrange - 无效请求
        invalid_request = {
            "symbol": "",  # 空符号
            "indicator": "invalid_indicator",  # 无效指标
            "parameters": {}  # 缺少参数
        }
        
        # Act
        is_valid, errors = await api.validate_request(invalid_request)
        
        # Assert
        assert is_valid is False
        assert len(errors) > 0
        assert any("symbol" in error for error in errors)
        assert any("indicator" in error for error in errors)

    @pytest.mark.asyncio
    async def test_cache_indicator_results(self, api):
        """测试指标结果缓存"""
        # Arrange
        cache_key = "000001.SZ_sma_5_20240101_20240131"
        indicator_data = {
            "indicator": "sma",
            "values": [12.5, 12.6, 12.7, 12.8, 12.9]
        }
        
        # Act
        await api.cache_indicator_result(cache_key, indicator_data)
        cached_result = await api.get_cached_indicator_result(cache_key)
        
        # Assert
        assert cached_result is not None
        assert cached_result['indicator'] == 'sma'
        assert len(cached_result['values']) == 5

    @pytest.mark.asyncio
    async def test_indicator_calculation_performance(self, api, sample_market_data):
        """测试指标计算性能"""
        # Arrange
        large_dataset = sample_market_data * 20  # 1000个数据点
        request_data = {
            "symbol": "000001.SZ",
            "indicator": "sma",
            "parameters": {"period": 20},
            "data": large_dataset
        }
        
        # Act
        start_time = datetime.now()
        result = await api.calculate_indicator(request_data)
        end_time = datetime.now()
        
        # Assert
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 1.0  # 计算应该在1秒内完成
        assert result['success'] is True

    @pytest.mark.asyncio
    async def test_error_handling_for_insufficient_data(self, api):
        """测试数据不足时的错误处理"""
        # Arrange - 数据不足
        insufficient_data = [
            {"close": 12.5}, {"close": 12.6}  # 只有2个数据点
        ]
        request_data = {
            "symbol": "000001.SZ",
            "indicator": "sma",
            "parameters": {"period": 20},  # 需要20个数据点
            "data": insufficient_data
        }
        
        # Act
        result = await api.calculate_indicator(request_data)
        
        # Assert
        assert result['success'] is False
        assert 'error' in result
        assert '无法获取市场数据' in result['error'] or 'insufficient data' in result['error'].lower()


if __name__ == "__main__":
    # 运行测试确保全部失败（TDD第一步）
    pytest.main([__file__, "-v", "--tb=short"])