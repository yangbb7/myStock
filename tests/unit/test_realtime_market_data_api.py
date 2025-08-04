"""
实时市场数据API的TDD测试

按照TDD原则，先编写完整的测试确保测试全部失败，然后实现功能代码
测试实时行情获取、K线数据查询、WebSocket数据推送等功能
"""

import pytest
import pytest_asyncio
import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Optional
from unittest.mock import AsyncMock, patch, MagicMock, call

# 待实现的模块
from myQuant.interfaces.api.market_data_api import MarketDataAPI
from myQuant.core.data.market_data_provider import MarketDataProvider
from myQuant.core.data.realtime_data_manager import RealTimeDataManager
from myQuant.interfaces.api.websocket_manager import WebSocketManager
from myQuant.core.models.market_data import RealTimeQuote, KlineData


class TestMarketDataAPI:
    """市场数据API测试"""

    @pytest.fixture
    def mock_database_manager(self):
        """模拟数据库管理器"""
        return AsyncMock()

    @pytest.fixture
    def mock_data_provider(self):
        """模拟数据提供者"""
        # Use AsyncMock without spec to allow any method calls
        mock = AsyncMock()
        # Set up the methods that might be called
        mock.get_realtime_quote = AsyncMock()
        mock.get_batch_realtime_quotes = AsyncMock()
        mock.get_market_depth = AsyncMock()
        mock.get_trade_ticks = AsyncMock()
        return mock

    @pytest.fixture
    def api(self, mock_database_manager, mock_data_provider):
        """市场数据API实例"""
        api = MarketDataAPI(mock_database_manager)
        # Inject the mock data provider into the data manager
        api.data_manager.data_provider = mock_data_provider
        return api

    @pytest.fixture
    def sample_kline_data(self):
        """样本K线数据"""
        base_time = datetime.now() - timedelta(days=30)
        return [
            {
                "symbol": "000001.SZ",
                "timestamp": (base_time + timedelta(days=i)).isoformat(),
                "open": 12.50 + i * 0.01,
                "high": 12.80 + i * 0.01,
                "low": 12.30 + i * 0.01,
                "close": 12.75 + i * 0.01,
                "volume": 1000000 + i * 10000,
                "turnover": (12.75 + i * 0.01) * (1000000 + i * 10000)
            }
            for i in range(30)
        ]

    @pytest.mark.asyncio
    async def test_get_kline_data_endpoint(self, api, mock_database_manager, sample_kline_data):
        """测试获取K线数据端点"""
        # Arrange
        mock_database_manager.fetch_all.return_value = sample_kline_data
        
        # Act
        result = await api.get_kline_data(
            symbol="000001.SZ",
            period="1d",
            start_date="2024-01-01",
            end_date="2024-01-31",
            limit=1000
        )
        
        # Assert
        assert result['success'] is True
        assert 'data' in result
        assert result['data']['symbol'] == "000001.SZ"
        assert result['data']['period'] == "1d"
        assert len(result['data']['records']) == 30
        
        # 验证数据库查询
        mock_database_manager.fetch_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_realtime_quote_endpoint(self, api, mock_data_provider):
        """测试获取实时行情端点"""
        # Arrange
        mock_quote = RealTimeQuote(
            symbol="000001.SZ",
            current_price=Decimal("12.75"),
            change_amount=Decimal("0.25"),
            change_percent=Decimal("2.00"),
            volume=15000000,
            turnover=Decimal("191250000.00"),
            bid_price_1=Decimal("12.74"),
            bid_volume_1=50000,
            ask_price_1=Decimal("12.75"),
            ask_volume_1=30000,
            last_updated=datetime.now()
        )
        mock_data_provider.get_realtime_quote.return_value = mock_quote
        
        # Act
        result = await api.get_realtime_quote("000001.SZ")
        
        # Assert
        assert result['success'] is True
        assert result['data']['symbol'] == "000001.SZ"
        assert result['data']['current_price'] == 12.75
        assert result['data']['change_amount'] == 0.25
        assert result['data']['change_percent'] == 2.00
        assert 'bid_ask' in result['data']
        
        mock_data_provider.get_realtime_quote.assert_called_once_with("000001.SZ")

    @pytest.mark.asyncio
    async def test_get_multiple_realtime_quotes(self, api, mock_data_provider):
        """测试获取多只股票实时行情"""
        # Arrange
        symbols = ["000001.SZ", "000002.SZ", "600000.SH"]
        mock_quotes = [
            RealTimeQuote(
                symbol=symbol,
                current_price=Decimal("12.75"),
                change_amount=Decimal("0.25"),
                change_percent=Decimal("2.00"),
                volume=15000000,
                turnover=Decimal("191250000.00"),
                last_updated=datetime.now()
            )
            for symbol in symbols
        ]
        mock_data_provider.get_batch_realtime_quotes.return_value = mock_quotes
        
        # Act
        result = await api.get_batch_realtime_quotes(symbols)
        
        # Assert
        assert result['success'] is True
        assert len(result['data']['quotes']) == 3
        for i, quote in enumerate(result['data']['quotes']):
            assert quote['symbol'] == symbols[i]
        
        mock_data_provider.get_batch_realtime_quotes.assert_called_once_with(symbols)

    @pytest.mark.asyncio
    async def test_get_kline_data_with_different_periods(self, api, mock_database_manager):
        """测试获取不同周期的K线数据"""
        # Arrange
        periods = ["1min", "5min", "15min", "30min", "1h", "1d", "1w", "1M"]
        mock_database_manager.fetch_all.return_value = []
        
        for period in periods:
            # Act
            result = await api.get_kline_data(
                symbol="000001.SZ",
                period=period,
                limit=100
            )
            
            # Assert
            assert result['success'] is True
            assert result['data']['period'] == period

    @pytest.mark.asyncio
    async def test_get_kline_data_with_validation(self, api):
        """测试K线数据请求验证"""
        # Act & Assert - 无效股票代码
        result = await api.get_kline_data(symbol="", period="1d")
        assert result['success'] is False
        assert 'error' in result
        
        # Act & Assert - 无效周期
        result = await api.get_kline_data(symbol="000001.SZ", period="invalid")
        assert result['success'] is False
        assert 'error' in result
        
        # Act & Assert - 无效日期范围
        result = await api.get_kline_data(
            symbol="000001.SZ", 
            period="1d",
            start_date="2024-01-31",
            end_date="2024-01-01"  # 结束日期早于开始日期
        )
        assert result['success'] is False
        assert 'error' in result

    @pytest.mark.asyncio
    async def test_get_market_depth(self, api, mock_data_provider):
        """测试获取市场深度数据"""
        # Arrange
        mock_depth = {
            "symbol": "000001.SZ",
            "bids": [
                {"price": 12.74, "volume": 50000},
                {"price": 12.73, "volume": 30000},
                {"price": 12.72, "volume": 40000}
            ],
            "asks": [
                {"price": 12.75, "volume": 30000},
                {"price": 12.76, "volume": 25000},
                {"price": 12.77, "volume": 35000}
            ],
            "timestamp": datetime.now().isoformat()
        }
        mock_data_provider.get_market_depth.return_value = mock_depth
        
        # Act
        result = await api.get_market_depth("000001.SZ", level=5)
        
        # Assert
        assert result['success'] is True
        assert result['data']['symbol'] == "000001.SZ"
        assert len(result['data']['bids']) == 3
        assert len(result['data']['asks']) == 3
        assert 'timestamp' in result['data']

    @pytest.mark.asyncio
    async def test_get_trade_ticks(self, api, mock_data_provider):
        """测试获取逐笔交易数据"""
        # Arrange
        mock_ticks = [
            {
                "symbol": "000001.SZ",
                "timestamp": datetime.now().isoformat(),
                "price": 12.75,
                "volume": 1000,
                "direction": "BUY"
            },
            {
                "symbol": "000001.SZ",
                "timestamp": datetime.now().isoformat(),
                "price": 12.74,
                "volume": 500,
                "direction": "SELL"
            }
        ]
        mock_data_provider.get_trade_ticks.return_value = mock_ticks
        
        # Act
        result = await api.get_trade_ticks("000001.SZ", limit=100)
        
        # Assert
        assert result['success'] is True
        assert len(result['data']['ticks']) == 2
        assert result['data']['ticks'][0]['symbol'] == "000001.SZ"

    @pytest.mark.asyncio
    async def test_search_stocks(self, api, mock_database_manager):
        """测试股票搜索功能"""
        # Arrange
        mock_stocks = [
            {"symbol": "000001.SZ", "name": "平安银行", "sector": "金融"},
            {"symbol": "000002.SZ", "name": "万科A", "sector": "房地产"}
        ]
        mock_database_manager.fetch_all.return_value = mock_stocks
        
        # Act
        result = await api.search_stocks("平安", limit=10)
        
        # Assert
        assert result['success'] is True
        assert len(result['data']['stocks']) == 2
        assert "平安银行" in [stock['name'] for stock in result['data']['stocks']]

    @pytest.mark.asyncio
    async def test_get_stock_info(self, api, mock_database_manager):
        """测试获取股票基本信息"""
        # Arrange
        mock_stock_info = {
            "symbol": "000001.SZ",
            "name": "平安银行",
            "sector": "金融",
            "industry": "银行",
            "market": "SZ",
            "listing_date": "1991-04-03",
            "total_shares": 19405918198,
            "float_shares": 19405918198
        }
        mock_database_manager.fetch_one.return_value = mock_stock_info
        
        # Act
        result = await api.get_stock_info("000001.SZ")
        
        # Assert
        assert result['success'] is True
        assert result['data']['symbol'] == "000001.SZ"
        assert result['data']['name'] == "平安银行"
        assert result['data']['sector'] == "金融"

    @pytest.mark.asyncio
    async def test_cache_market_data(self, api):
        """测试市场数据缓存"""
        # Arrange
        cache_key = "realtime_000001.SZ"
        quote_data = {
            "symbol": "000001.SZ",
            "current_price": 12.75,
            "timestamp": datetime.now().isoformat()
        }
        
        # Act
        await api.cache_market_data(cache_key, quote_data, ttl=60)
        cached_data = await api.get_cached_market_data(cache_key)
        
        # Assert
        assert cached_data is not None
        assert cached_data['symbol'] == "000001.SZ"
        assert cached_data['current_price'] == 12.75

    @pytest.mark.asyncio
    async def test_rate_limiting(self, api):
        """测试API请求限流"""
        # Arrange - 配置限流 (例如: 每秒最多100次请求)
        api.set_rate_limit(requests_per_second=2)
        
        # Act - 快速连续请求
        start_time = datetime.now()
        results = []
        for i in range(5):
            result = await api.get_realtime_quote("000001.SZ")
            results.append(result)
        end_time = datetime.now()
        
        # Assert
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time >= 2.0  # 由于限流，应该至少需要2秒
        
        # All requests should succeed with this implementation since we sleep instead of rejecting
        successful_results = [r for r in results if r.get('success', True)]
        assert len(successful_results) == 5


class TestRealTimeDataManager:
    """实时数据管理器测试"""

    @pytest.fixture
    def mock_data_provider(self):
        return AsyncMock(spec=MarketDataProvider)

    @pytest.fixture
    def data_manager(self, mock_data_provider):
        return RealTimeDataManager(mock_data_provider)

    @pytest.mark.asyncio
    async def test_start_realtime_data_stream(self, data_manager, mock_data_provider):
        """测试启动实时数据流"""
        # Arrange
        symbols = ["000001.SZ", "000002.SZ"]
        
        # Act
        await data_manager.start_stream(symbols)
        
        # Assert
        assert data_manager.is_streaming() is True
        assert set(data_manager.get_subscribed_symbols()) == set(symbols)

    @pytest.mark.asyncio
    async def test_add_subscription(self, data_manager):
        """测试添加订阅"""
        # Arrange
        initial_symbols = ["000001.SZ"]
        await data_manager.start_stream(initial_symbols)
        
        # Act
        await data_manager.add_subscription("000002.SZ")
        
        # Assert
        subscribed = data_manager.get_subscribed_symbols()
        assert "000001.SZ" in subscribed
        assert "000002.SZ" in subscribed

    @pytest.mark.asyncio
    async def test_remove_subscription(self, data_manager):
        """测试移除订阅"""
        # Arrange
        symbols = ["000001.SZ", "000002.SZ"]
        await data_manager.start_stream(symbols)
        
        # Act
        await data_manager.remove_subscription("000001.SZ")
        
        # Assert
        subscribed = data_manager.get_subscribed_symbols()
        assert "000001.SZ" not in subscribed
        assert "000002.SZ" in subscribed

    @pytest.mark.asyncio
    async def test_data_callback_handling(self, data_manager):
        """测试数据回调处理"""
        # Arrange
        received_data = []
        
        def data_callback(symbol, data):
            received_data.append((symbol, data))
        
        data_manager.add_data_callback(data_callback)
        await data_manager.start_stream(["000001.SZ"])
        
        # Act - 模拟接收数据
        mock_data = RealTimeQuote(
            symbol="000001.SZ",
            current_price=Decimal("12.75"),
            last_updated=datetime.now()
        )
        await data_manager._handle_data_update("000001.SZ", mock_data)
        
        # Assert
        assert len(received_data) == 1
        assert received_data[0][0] == "000001.SZ"

    @pytest.mark.asyncio
    async def test_connection_recovery(self, data_manager, mock_data_provider):
        """测试连接恢复机制"""
        # Arrange
        await data_manager.start_stream(["000001.SZ"])
        
        # Act - 模拟连接断开
        await data_manager._handle_connection_lost()
        
        # Assert - 应该尝试重连
        assert data_manager.is_reconnecting() is True
        
        # 模拟重连成功
        await data_manager._handle_connection_restored()
        assert data_manager.is_streaming() is True

    @pytest.mark.asyncio
    async def test_data_quality_monitoring(self, data_manager):
        """测试数据质量监控"""
        # Arrange
        await data_manager.start_stream(["000001.SZ"])
        
        # Act - 发送一些数据并检查质量指标
        for i in range(10):
            mock_data = RealTimeQuote(
                symbol="000001.SZ",
                current_price=Decimal("12.75"),
                last_updated=datetime.now()
            )
            await data_manager._handle_data_update("000001.SZ", mock_data)
        
        # Assert
        quality_metrics = data_manager.get_data_quality_metrics("000001.SZ")
        assert quality_metrics['messages_received'] == 10
        assert quality_metrics['last_update_time'] is not None


class TestWebSocketManager:
    """WebSocket管理器测试"""

    @pytest.fixture
    def websocket_manager(self):
        from unittest.mock import Mock
        mock_trading_system = Mock()
        return WebSocketManager(mock_trading_system)

    @pytest.mark.asyncio
    async def test_client_connection_management(self, websocket_manager):
        """测试客户端连接管理"""
        # Arrange
        mock_websocket = AsyncMock()
        mock_websocket.client_state.name = "CONNECTED"
        
        # Act
        client_id = await websocket_manager.add_client(mock_websocket)
        
        # Assert
        assert client_id is not None
        assert len(websocket_manager.get_connected_clients()) == 1

    @pytest.mark.asyncio
    async def test_client_subscription_management(self, websocket_manager):
        """测试客户端订阅管理"""
        # Arrange
        mock_websocket = AsyncMock()
        client_id = await websocket_manager.add_client(mock_websocket)
        
        # Act
        await websocket_manager.subscribe_client(client_id, "market_data", ["000001.SZ"])
        
        # Assert
        subscriptions = websocket_manager.get_client_subscriptions(client_id)
        assert "market_data" in subscriptions
        assert "000001.SZ" in subscriptions["market_data"]

    @pytest.mark.asyncio
    async def test_broadcast_market_data(self, websocket_manager):
        """测试广播市场数据"""
        # Arrange
        mock_websockets = [AsyncMock() for _ in range(3)]
        client_ids = []
        for ws in mock_websockets:
            client_id = await websocket_manager.add_client(ws)
            await websocket_manager.subscribe_client(client_id, "market_data", ["000001.SZ"])
            client_ids.append(client_id)
        
        market_data = {
            "type": "market_data",
            "data": {
                "symbol": "000001.SZ",
                "price": 12.75,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Act
        await websocket_manager.broadcast_to_channel("market_data", market_data)
        
        # Assert
        for ws in mock_websockets:
            ws.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_authentication(self, websocket_manager):
        """测试客户端认证"""
        # Arrange
        mock_websocket = AsyncMock()
        auth_message = {
            "type": "auth",
            "token": "valid_jwt_token"
        }
        
        # Act
        with patch('myQuant.core.auth.jwt_manager.verify_token') as mock_verify:
            mock_verify.return_value = {"user_id": "user123"}
            result = await websocket_manager.authenticate_client(mock_websocket, auth_message)
        
        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_message_routing(self, websocket_manager):
        """测试消息路由"""
        # Arrange
        mock_websocket = AsyncMock()
        client_id = await websocket_manager.add_client(mock_websocket)
        
        subscribe_message = {
            "type": "subscribe",
            "channel": "market_data",
            "symbols": ["000001.SZ", "000002.SZ"]
        }
        
        # Act
        await websocket_manager.handle_client_message(client_id, subscribe_message)
        
        # Assert
        subscriptions = websocket_manager.get_client_subscriptions(client_id)
        assert len(subscriptions["market_data"]) == 2

    @pytest.mark.asyncio
    async def test_connection_cleanup(self, websocket_manager):
        """测试连接清理"""
        # Arrange
        mock_websocket = AsyncMock()
        client_id = await websocket_manager.add_client(mock_websocket)
        await websocket_manager.subscribe_client(client_id, "market_data", ["000001.SZ"])
        
        # Act
        await websocket_manager.remove_client(client_id)
        
        # Assert
        assert len(websocket_manager.get_connected_clients()) == 0
        subscriptions = websocket_manager.get_client_subscriptions(client_id)
        assert len(subscriptions) == 0


if __name__ == "__main__":
    # 运行测试确保全部失败（TDD第一步）
    pytest.main([__file__, "-v", "--tb=short"])