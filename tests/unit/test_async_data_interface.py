# -*- coding: utf-8 -*-
"""
测试异步数据接口模块
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from myQuant.core.interfaces.async_data_interface import IAsyncDataProvider


class TestAsyncDataInterface:
    """测试异步数据接口"""
    
    class MockAsyncDataProvider(IAsyncDataProvider):
        """模拟异步数据提供者"""
        
        async def get_price_data(self, symbol, start_date=None, end_date=None):
            """模拟获取价格数据"""
            raise NotImplementedError("This is an abstract method")
        
        async def get_price_data_batch(self, symbols, start_date=None, end_date=None):
            """模拟批量获取价格数据"""
            raise NotImplementedError("This is an abstract method")
            yield  # unreachable but needed for async generator
        
        async def get_financial_data(self, symbol, report_date=None):
            """模拟获取财务数据"""
            raise NotImplementedError("This is an abstract method")
        
        async def get_real_time_quote(self, symbol):
            """模拟获取实时报价"""
            raise NotImplementedError("This is an abstract method")
        
        async def get_real_time_quotes_batch(self, symbols):
            """模拟批量获取实时报价"""
            raise NotImplementedError("This is an abstract method")
            yield  # unreachable but needed for async generator
        
        async def get_fundamental_data(self, symbol, report_date=None):
            """模拟获取基本面数据"""
            raise NotImplementedError("This is an abstract method")
        
        async def get_technical_indicators(self, symbol, indicators, period=20):
            """模拟获取技术指标"""
            raise NotImplementedError("This is an abstract method")
    
    @pytest.fixture
    def interface(self):
        """创建异步数据接口实例"""
        return TestAsyncDataInterface.MockAsyncDataProvider()
    
    @pytest.mark.asyncio
    async def test_get_price_data_abstract(self, interface):
        """测试获取价格数据抽象方法"""
        with pytest.raises(NotImplementedError):
            await interface.get_price_data("AAPL", "2023-01-01", "2023-12-31")
    
    @pytest.mark.asyncio
    async def test_get_price_data_batch_abstract(self, interface):
        """测试批量获取价格数据抽象方法"""
        with pytest.raises(NotImplementedError):
            generator = interface.get_price_data_batch(["AAPL", "MSFT"], "2023-01-01", "2023-12-31")
            await generator.__anext__()
    
    @pytest.mark.asyncio
    async def test_get_financial_data_abstract(self, interface):
        """测试获取财务数据抽象方法"""
        with pytest.raises(NotImplementedError):
            await interface.get_financial_data("AAPL", "2023-12-31")
    
    @pytest.mark.asyncio
    async def test_get_real_time_quote_abstract(self, interface):
        """测试获取实时报价抽象方法"""
        with pytest.raises(NotImplementedError):
            await interface.get_real_time_quote("AAPL")
    
    @pytest.mark.asyncio
    async def test_get_real_time_quotes_batch_abstract(self, interface):
        """测试批量获取实时报价抽象方法"""
        with pytest.raises(NotImplementedError):
            generator = interface.get_real_time_quotes_batch(["AAPL", "MSFT"])
            await generator.__anext__()
    
    @pytest.mark.asyncio
    async def test_get_fundamental_data_abstract(self, interface):
        """测试获取基本面数据抽象方法"""
        with pytest.raises(NotImplementedError):
            await interface.get_fundamental_data("AAPL", "2023-12-31")
    
    @pytest.mark.asyncio
    async def test_get_technical_indicators_abstract(self, interface):
        """测试获取技术指标抽象方法"""
        with pytest.raises(NotImplementedError):
            await interface.get_technical_indicators("AAPL", ["MA", "RSI"], 20)


class TestConcreteAsyncDataInterface:
    """测试具体的异步数据接口实现"""
    
    class ConcreteAsyncDataProvider(IAsyncDataProvider):
        """具体的异步数据提供者实现"""
        
        async def get_price_data(self, symbol, start_date=None, end_date=None):
            """获取价格数据"""
            if start_date is None:
                start_date = "2023-01-01"
            if end_date is None:
                end_date = "2023-01-10"
            
            # 创建模拟数据
            dates = pd.date_range(start_date, end_date, freq='D')
            np.random.seed(42)  # 确保可重复性
            
            data = {
                'date': dates,
                'symbol': [symbol] * len(dates),
                'open': np.random.uniform(100, 200, len(dates)),
                'high': np.random.uniform(150, 250, len(dates)),
                'low': np.random.uniform(50, 150, len(dates)),
                'close': np.random.uniform(100, 200, len(dates)),
                'volume': np.random.randint(1000000, 10000000, len(dates))
            }
            
            return pd.DataFrame(data)
        
        async def get_price_data_batch(self, symbols, start_date=None, end_date=None):
            """批量获取价格数据"""
            for symbol in symbols:
                data = await self.get_price_data(symbol, start_date, end_date)
                yield {
                    'symbol': symbol,
                    'data': data,
                    'status': 'success'
                }
        
        async def get_financial_data(self, symbol, report_date=None):
            """获取财务数据"""
            await asyncio.sleep(0.001)  # 模拟异步操作
            return {
                'symbol': symbol,
                'date': report_date,
                'revenue': 100000000,
                'net_income': 20000000,
                'eps': 2.5,
                'roe': 0.15
            }
        
        async def get_real_time_quote(self, symbol):
            """获取实时报价"""
            await asyncio.sleep(0.001)
            return {
                'symbol': symbol,
                'price': 150.25,
                'bid': 150.20,
                'ask': 150.30,
                'volume': 1000000,
                'timestamp': datetime.now().isoformat()
            }
        
        async def get_real_time_quotes_batch(self, symbols):
            """批量获取实时报价"""
            for symbol in symbols:
                quote = await self.get_real_time_quote(symbol)
                yield quote
        
        async def get_fundamental_data(self, symbol, report_date=None):
            """获取基本面数据"""
            await asyncio.sleep(0.001)
            return {
                'symbol': symbol,
                'report_date': report_date,
                'pe_ratio': 25.5,
                'pb_ratio': 3.2,
                'roe': 0.15,
                'debt_to_equity': 0.8
            }
        
        async def get_technical_indicators(self, symbol, indicators, period=20):
            """获取技术指标"""
            await asyncio.sleep(0.001)
            result = {'symbol': symbol, 'period': period}
            
            for indicator in indicators:
                if indicator == 'MA':
                    result['MA'] = 150.0
                elif indicator == 'RSI':
                    result['RSI'] = 65.5
                elif indicator == 'MACD':
                    result['MACD'] = {'macd': 1.2, 'signal': 0.8, 'histogram': 0.4}
                else:
                    result[indicator] = 50.0
            
            return result
    
    @pytest.fixture
    def provider(self):
        """创建具体的异步数据提供者"""
        return TestConcreteAsyncDataInterface.ConcreteAsyncDataProvider()
    
    @pytest.mark.asyncio
    async def test_get_price_data_implementation(self, provider):
        """测试价格数据获取实现"""
        result = await provider.get_price_data("AAPL", "2023-01-01", "2023-01-10")
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'symbol' in result.columns
        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns
        assert 'volume' in result.columns
        assert result['symbol'].iloc[0] == "AAPL"
        assert len(result) == 10  # 10 days of data
        
        # 验证数据完整性
        assert all(result['high'] >= result['low'])
        assert all(result['volume'] > 0)
    
    @pytest.mark.asyncio
    async def test_get_price_data_with_defaults(self, provider):
        """测试使用默认参数获取价格数据"""
        result = await provider.get_price_data("AAPL")
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert result['symbol'].iloc[0] == "AAPL"
        assert len(result) == 10  # 默认10天数据
    
    @pytest.mark.asyncio
    async def test_get_price_data_batch_implementation(self, provider):
        """测试批量价格数据获取实现"""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        results = []
        
        async for data in provider.get_price_data_batch(symbols, "2023-01-01", "2023-01-05"):
            results.append(data)
        
        assert len(results) == len(symbols)
        
        for i, result in enumerate(results):
            assert 'symbol' in result
            assert 'data' in result
            assert 'status' in result
            assert result['symbol'] == symbols[i]
            assert result['status'] == 'success'
            assert isinstance(result['data'], pd.DataFrame)
            assert not result['data'].empty
    
    @pytest.mark.asyncio
    async def test_get_financial_data_implementation(self, provider):
        """测试财务数据获取实现"""
        result = await provider.get_financial_data("AAPL", "2023-12-31")
        
        assert isinstance(result, dict)
        assert 'symbol' in result
        assert 'date' in result
        assert 'revenue' in result
        assert 'net_income' in result
        assert 'eps' in result
        assert 'roe' in result
        assert result['symbol'] == "AAPL"
        assert result['date'] == "2023-12-31"
        assert isinstance(result['revenue'], int)
        assert isinstance(result['eps'], float)
    
    @pytest.mark.asyncio
    async def test_get_real_time_quote_implementation(self, provider):
        """测试实时报价获取实现"""
        result = await provider.get_real_time_quote("AAPL")
        
        assert isinstance(result, dict)
        assert 'symbol' in result
        assert 'price' in result
        assert 'bid' in result
        assert 'ask' in result
        assert 'volume' in result
        assert 'timestamp' in result
        assert result['symbol'] == "AAPL"
        assert isinstance(result['price'], float)
        assert isinstance(result['volume'], int)
    
    @pytest.mark.asyncio
    async def test_get_real_time_quotes_batch_implementation(self, provider):
        """测试批量实时报价获取实现"""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        results = []
        
        async for quote in provider.get_real_time_quotes_batch(symbols):
            results.append(quote)
        
        assert len(results) == len(symbols)
        for i, result in enumerate(results):
            assert 'symbol' in result
            assert 'price' in result
            assert result['symbol'] == symbols[i]
    
    @pytest.mark.asyncio
    async def test_get_fundamental_data_implementation(self, provider):
        """测试基本面数据获取实现"""
        result = await provider.get_fundamental_data("AAPL", "2023-12-31")
        
        assert isinstance(result, dict)
        assert 'symbol' in result
        assert 'report_date' in result
        assert 'pe_ratio' in result
        assert 'pb_ratio' in result
        assert 'roe' in result
        assert 'debt_to_equity' in result
        assert result['symbol'] == "AAPL"
        assert result['report_date'] == "2023-12-31"
    
    @pytest.mark.asyncio
    async def test_get_technical_indicators_implementation(self, provider):
        """测试技术指标获取实现"""
        indicators = ["MA", "RSI", "MACD"]
        result = await provider.get_technical_indicators("AAPL", indicators, 20)
        
        assert isinstance(result, dict)
        assert 'symbol' in result
        assert 'period' in result
        assert result['symbol'] == "AAPL"
        assert result['period'] == 20
        
        for indicator in indicators:
            assert indicator in result
        
        assert isinstance(result['MA'], float)
        assert isinstance(result['RSI'], float)
        assert isinstance(result['MACD'], dict)
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, provider):
        """测试并发请求"""
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        # 并发获取多个股票的数据
        tasks = [
            provider.get_price_data(symbol, "2023-01-01", "2023-01-05") 
            for symbol in symbols
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == len(symbols)
        for i, result in enumerate(results):
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert result['symbol'].iloc[0] == symbols[i]
    
    @pytest.mark.asyncio
    async def test_concurrent_financial_data(self, provider):
        """测试并发财务数据请求"""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        # 并发获取多个股票的财务数据
        tasks = [
            provider.get_financial_data(symbol, "2023-12-31") 
            for symbol in symbols
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == len(symbols)
        for i, result in enumerate(results):
            assert isinstance(result, dict)
            assert result['symbol'] == symbols[i]
            assert result['date'] == "2023-12-31"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, provider):
        """测试错误处理"""
        # 创建一个会抛出异常的提供者
        class FailingProvider(IAsyncDataProvider):
            async def get_price_data(self, symbol, start_date=None, end_date=None):
                raise ValueError("API Error")
            
            async def get_price_data_batch(self, symbols, start_date=None, end_date=None):
                raise ValueError("API Error")
                
            async def get_financial_data(self, symbol, report_date=None):
                raise ValueError("API Error")
                
            async def get_real_time_quote(self, symbol):
                raise ValueError("API Error")
                
            async def get_real_time_quotes_batch(self, symbols):
                raise ValueError("API Error")
                
            async def get_fundamental_data(self, symbol, report_date=None):
                raise ValueError("API Error")
                
            async def get_technical_indicators(self, symbol, indicators, period=20):
                raise ValueError("API Error")
        
        failing_provider = FailingProvider()
        
        with pytest.raises(ValueError, match="API Error"):
            await failing_provider.get_price_data("AAPL")
        
        with pytest.raises(ValueError, match="API Error"):
            await failing_provider.get_financial_data("AAPL", "2023-12-31")
        
        with pytest.raises(ValueError, match="API Error"):
            await failing_provider.get_real_time_quote("AAPL")
        
        with pytest.raises(ValueError, match="API Error"):
            await failing_provider.get_fundamental_data("AAPL", "2023-12-31")
        
        with pytest.raises(ValueError, match="API Error"):
            await failing_provider.get_technical_indicators("AAPL", ["MA"], 20)
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, provider):
        """测试超时处理"""
        # 创建一个慢速提供者
        class SlowProvider(IAsyncDataProvider):
            async def get_price_data(self, symbol, start_date=None, end_date=None):
                await asyncio.sleep(2)  # 模拟慢速操作
                return pd.DataFrame()
            
            async def get_price_data_batch(self, symbols, start_date=None, end_date=None):
                await asyncio.sleep(2)
                for symbol in symbols:
                    yield {'symbol': symbol, 'data': pd.DataFrame()}
                
            async def get_financial_data(self, symbol, report_date=None):
                await asyncio.sleep(2)
                return {}
                
            async def get_real_time_quote(self, symbol):
                await asyncio.sleep(2)
                return {}
                
            async def get_real_time_quotes_batch(self, symbols):
                await asyncio.sleep(2)
                for symbol in symbols:
                    yield {'symbol': symbol}
                    
            async def get_fundamental_data(self, symbol, report_date=None):
                await asyncio.sleep(2)
                return {}
                
            async def get_technical_indicators(self, symbol, indicators, period=20):
                await asyncio.sleep(2)
                return {}
        
        slow_provider = SlowProvider()
        
        # 使用timeout处理
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                slow_provider.get_price_data("AAPL"),
                timeout=0.5
            )
    
    @pytest.mark.asyncio
    async def test_large_data_handling(self, provider):
        """测试大数据处理"""
        # 获取一年的数据
        result = await provider.get_price_data("AAPL", "2023-01-01", "2023-12-31")
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) == 365  # 一年的数据
        
        # 验证数据完整性
        assert 'symbol' in result.columns
        assert all(result['symbol'] == "AAPL")
        assert all(result['high'] >= result['low'])
        assert all(result['volume'] > 0)
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, provider):
        """测试批处理性能"""
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        # 测试批量处理
        start_time = datetime.now()
        results = []
        
        async for data in provider.get_price_data_batch(symbols, "2023-01-01", "2023-01-05"):
            results.append(data)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        assert len(results) == len(symbols)
        assert processing_time < 2.0  # 批处理应该相对较快
        
        # 验证结果
        for i, result in enumerate(results):
            assert result['symbol'] == symbols[i]
            assert result['status'] == 'success'
            assert isinstance(result['data'], pd.DataFrame)
    
    @pytest.mark.asyncio
    async def test_data_consistency(self, provider):
        """测试数据一致性"""
        # 多次获取相同数据，应该保持一致
        symbol = "AAPL"
        start_date = "2023-01-01"
        end_date = "2023-01-05"
        
        result1 = await provider.get_price_data(symbol, start_date, end_date)
        result2 = await provider.get_price_data(symbol, start_date, end_date)
        
        # 由于使用了随机种子，结果应该一致
        assert result1.equals(result2)
    
    @pytest.mark.asyncio
    async def test_edge_cases(self, provider):
        """测试边界情况"""
        # 测试空股票代码
        result = await provider.get_price_data("", "2023-01-01", "2023-01-05")
        assert isinstance(result, pd.DataFrame)
        assert result['symbol'].iloc[0] == ""
        
        # 测试单日数据
        result = await provider.get_price_data("AAPL", "2023-01-01", "2023-01-01")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        
        # 测试财务数据边界情况
        result = await provider.get_financial_data("TEST", "2023-12-31")
        assert result['symbol'] == "TEST"
        assert result['date'] == "2023-12-31"
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, provider):
        """测试内存效率"""
        # 获取多个大数据集，确保内存使用合理
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        # 并发获取大数据集
        tasks = [
            provider.get_price_data(symbol, "2023-01-01", "2023-12-31")
            for symbol in symbols
        ]
        
        results = await asyncio.gather(*tasks)
        
        # 验证结果
        assert len(results) == len(symbols)
        for i, result in enumerate(results):
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert all(result['symbol'] == symbols[i])
            assert len(result) == 365  # 一年的数据
    
    @pytest.mark.asyncio
    async def test_async_generator_behavior(self, provider):
        """测试异步生成器行为"""
        symbols = ["AAPL", "MSFT"]
        
        # 测试异步生成器的逐项处理
        processed_symbols = []
        
        async for data in provider.get_price_data_batch(symbols, "2023-01-01", "2023-01-05"):
            processed_symbols.append(data['symbol'])
            
            # 验证每个项目
            assert 'symbol' in data
            assert 'data' in data
            assert 'status' in data
            assert data['status'] == 'success'
            assert isinstance(data['data'], pd.DataFrame)
        
        # 确保所有符号都被处理
        assert processed_symbols == symbols
    
    @pytest.mark.asyncio
    async def test_context_manager_compatibility(self, provider):
        """测试上下文管理器兼容性"""
        # 虽然当前接口不是上下文管理器，但测试在上下文中使用
        symbol = "AAPL"
        
        async def get_data_in_context():
            return await provider.get_price_data(symbol, "2023-01-01", "2023-01-05")
        
        result = await get_data_in_context()
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert result['symbol'].iloc[0] == symbol
