import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import json
from dataclasses import dataclass

from ..base_provider import BaseDataProvider, DataRequest, DataResponse, DataType, DataQualityMetrics

@dataclass
class OrderBookLevel:
    price: float
    volume: int
    order_count: int
    timestamp: datetime

@dataclass
class Level2Data:
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    last_price: float
    last_volume: int
    total_bid_volume: int
    total_ask_volume: int
    spread: float
    mid_price: float
    source: str

class Level2DataProvider(BaseDataProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.websocket_url = config.get('websocket_url', 'wss://api.level2.com/ws')
        self.api_key = config.get('api_key')
        self.depth_levels = config.get('depth_levels', 10)
        self.update_frequency = config.get('update_frequency', 100)  # ms
        self.session = None
        self.websocket = None
        self.subscriptions = {}
        self.supported_data_types = [DataType.ORDERBOOK, DataType.TICK, DataType.TRADE]
        self.supported_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
        
    async def connect(self) -> bool:
        try:
            self.logger.info("Connecting to Level-2 Market Data API...")
            
            # 模拟Level-2数据连接
            await asyncio.sleep(0.2)
            
            self.is_connected = True
            self.connection_stats['success_count'] += 1
            self.logger.info("Successfully connected to Level-2 Market Data API")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Level-2 API: {e}")
            self.connection_stats['error_count'] += 1
            self.connection_stats['last_error'] = str(e)
            return False
    
    async def disconnect(self) -> bool:
        try:
            if self.websocket:
                self.websocket = None
            
            if self.session:
                self.session = None
            
            self.subscriptions.clear()
            self.is_connected = False
            self.logger.info("Disconnected from Level-2 Market Data API")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during Level-2 API disconnection: {e}")
            return False
    
    async def get_data(self, request: DataRequest) -> DataResponse:
        if not self.is_connected:
            raise Exception("Not connected to Level-2 Market Data API")
        
        start_time = datetime.now()
        
        try:
            if request.data_type == DataType.ORDERBOOK:
                data = await self._get_orderbook_data(request)
            elif request.data_type == DataType.TICK:
                data = await self._get_tick_data(request)
            elif request.data_type == DataType.TRADE:
                data = await self._get_trade_data(request)
            else:
                raise ValueError(f"Unsupported data type: {request.data_type}")
            
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            quality_metrics = self._calculate_quality_metrics(data)
            
            return DataResponse(
                data=data,
                metadata={
                    'data_type': request.data_type.value,
                    'symbol': request.symbol,
                    'depth_levels': self.depth_levels,
                    'update_frequency': self.update_frequency
                },
                quality_metrics=quality_metrics,
                source='Level2',
                timestamp=end_time,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching Level-2 data: {e}")
            self.connection_stats['error_count'] += 1
            self.connection_stats['last_error'] = str(e)
            raise
    
    async def _get_orderbook_data(self, request: DataRequest) -> Dict[str, Any]:
        await asyncio.sleep(0.01)
        
        base_price = 150.0 + (hash(request.symbol) % 50)
        
        # 生成买盘数据
        bids = []
        for i in range(self.depth_levels):
            price = base_price - (i + 1) * 0.01
            volume = 1000 + (hash(f"{request.symbol}_bid_{i}") % 5000)
            order_count = 5 + (hash(f"{request.symbol}_bid_{i}") % 25)
            
            bids.append({
                'price': price,
                'volume': volume,
                'order_count': order_count,
                'timestamp': datetime.now()
            })
        
        # 生成卖盘数据
        asks = []
        for i in range(self.depth_levels):
            price = base_price + (i + 1) * 0.01
            volume = 1000 + (hash(f"{request.symbol}_ask_{i}") % 5000)
            order_count = 5 + (hash(f"{request.symbol}_ask_{i}") % 25)
            
            asks.append({
                'price': price,
                'volume': volume,
                'order_count': order_count,
                'timestamp': datetime.now()
            })
        
        total_bid_volume = sum(bid['volume'] for bid in bids)
        total_ask_volume = sum(ask['volume'] for ask in asks)
        
        spread = asks[0]['price'] - bids[0]['price']
        mid_price = (bids[0]['price'] + asks[0]['price']) / 2
        
        return {
            'symbol': request.symbol,
            'timestamp': datetime.now(),
            'bids': bids,
            'asks': asks,
            'last_price': base_price,
            'last_volume': 500,
            'total_bid_volume': total_bid_volume,
            'total_ask_volume': total_ask_volume,
            'spread': spread,
            'mid_price': mid_price,
            'depth_levels': self.depth_levels,
            'source': 'Level2'
        }
    
    async def _get_tick_data(self, request: DataRequest) -> Dict[str, Any]:
        await asyncio.sleep(0.005)
        
        base_price = 150.0 + (hash(request.symbol) % 50)
        
        return {
            'symbol': request.symbol,
            'timestamp': datetime.now(),
            'price': base_price + (hash(f"{request.symbol}_tick") % 200 - 100) / 1000,
            'volume': 100 + (hash(f"{request.symbol}_tick") % 900),
            'bid_price': base_price - 0.01,
            'ask_price': base_price + 0.01,
            'bid_volume': 500 + (hash(f"{request.symbol}_bid") % 2000),
            'ask_volume': 500 + (hash(f"{request.symbol}_ask") % 2000),
            'trade_type': 'buy' if hash(request.symbol) % 2 == 0 else 'sell',
            'exchange': 'NASDAQ',
            'source': 'Level2'
        }
    
    async def _get_trade_data(self, request: DataRequest) -> List[Dict[str, Any]]:
        await asyncio.sleep(0.02)
        
        trades = []
        base_price = 150.0 + (hash(request.symbol) % 50)
        
        for i in range(10):
            price = base_price + (hash(f"{request.symbol}_trade_{i}") % 100 - 50) / 1000
            volume = 100 + (hash(f"{request.symbol}_trade_{i}") % 1900)
            
            trades.append({
                'symbol': request.symbol,
                'timestamp': datetime.now() - timedelta(milliseconds=i * 100),
                'price': price,
                'volume': volume,
                'trade_id': f"T{hash(f'{request.symbol}_{i}') % 1000000:06d}",
                'trade_type': 'buy' if i % 2 == 0 else 'sell',
                'exchange': 'NASDAQ',
                'source': 'Level2'
            })
        
        return trades
    
    async def subscribe_realtime(self, symbols: List[str], data_type: DataType, callback: Callable) -> bool:
        if not self.is_connected:
            return False
        
        try:
            self.logger.info(f"Subscribing to Level-2 {data_type.value} data for symbols: {symbols}")
            
            subscription_key = f"{data_type.value}_{','.join(symbols)}"
            self.subscriptions[subscription_key] = {
                'symbols': symbols,
                'data_type': data_type,
                'callback': callback,
                'active': True
            }
            
            # 启动实时数据流
            asyncio.create_task(self._stream_realtime_data(subscription_key))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to Level-2 realtime data: {e}")
            return False
    
    async def unsubscribe_realtime(self, symbols: List[str], data_type: DataType) -> bool:
        try:
            subscription_key = f"{data_type.value}_{','.join(symbols)}"
            
            if subscription_key in self.subscriptions:
                self.subscriptions[subscription_key]['active'] = False
                del self.subscriptions[subscription_key]
                self.logger.info(f"Unsubscribed from Level-2 {data_type.value} data for symbols: {symbols}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from Level-2 realtime data: {e}")
            return False
    
    async def _stream_realtime_data(self, subscription_key: str):
        subscription = self.subscriptions.get(subscription_key)
        if not subscription:
            return
        
        symbols = subscription['symbols']
        data_type = subscription['data_type']
        callback = subscription['callback']
        
        while self.is_connected and subscription.get('active', False):
            try:
                for symbol in symbols:
                    if not subscription.get('active', False):
                        break
                    
                    request = DataRequest(symbol=symbol, data_type=data_type)
                    
                    if data_type == DataType.ORDERBOOK:
                        data = await self._get_orderbook_data(request)
                    elif data_type == DataType.TICK:
                        data = await self._get_tick_data(request)
                    elif data_type == DataType.TRADE:
                        data = await self._get_trade_data(request)
                        # 对于交易数据，发送最新的交易
                        data = data[0] if data else {}
                    else:
                        continue
                    
                    quality_metrics = self._calculate_quality_metrics(data)
                    
                    response = DataResponse(
                        data=data,
                        metadata={
                            'data_type': data_type.value,
                            'symbol': symbol,
                            'realtime': True,
                            'subscription_key': subscription_key
                        },
                        quality_metrics=quality_metrics,
                        source='Level2',
                        timestamp=datetime.now(),
                        latency_ms=10.0  # 极低延迟
                    )
                    
                    await callback(response)
                
                # 高频更新
                await asyncio.sleep(self.update_frequency / 1000.0)
                
            except Exception as e:
                self.logger.error(f"Error in Level-2 realtime data stream: {e}")
                await asyncio.sleep(1)
    
    async def get_market_depth(self, symbol: str, levels: int = None) -> Dict[str, Any]:
        if levels:
            original_levels = self.depth_levels
            self.depth_levels = levels
        
        try:
            request = DataRequest(symbol=symbol, data_type=DataType.ORDERBOOK)
            response = await self.get_data(request)
            return response.data
        finally:
            if levels:
                self.depth_levels = original_levels
    
    async def get_trade_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        request = DataRequest(symbol=symbol, data_type=DataType.TRADE)
        response = await self.get_data(request)
        
        trades = response.data if isinstance(response.data, list) else [response.data]
        return trades[:limit]
    
    def get_supported_data_types(self) -> List[DataType]:
        return self.supported_data_types
    
    def get_supported_symbols(self) -> List[str]:
        return self.supported_symbols
    
    def _calculate_quality_metrics(self, data: Any) -> DataQualityMetrics:
        if not data:
            return DataQualityMetrics(0, 0, 0, 0, 0, 0, 1, datetime.now())
        
        # Level-2数据质量极高，延迟极低
        completeness = 0.99
        accuracy = 0.995
        timeliness = 0.98
        consistency = 0.99
        validity = 0.99
        overall_score = (completeness + accuracy + timeliness + consistency + validity) / 5
        
        return DataQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            timeliness=timeliness,
            consistency=consistency,
            validity=validity,
            overall_score=overall_score,
            anomaly_count=0,
            last_updated=datetime.now()
        )