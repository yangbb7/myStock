# -*- coding: utf-8 -*-
"""
市场数据提供者 - 提供实时和历史市场数据
"""

import asyncio
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod


@dataclass
class MarketQuote:
    """市场报价数据"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None


@dataclass  
class KlineData:
    """K线数据"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    amount: Optional[float] = None


class MarketDataProvider(ABC):
    """市场数据提供者基类"""
    
    def __init__(self):
        self.is_connected = False
        self.subscriptions = set()
    
    @abstractmethod
    async def connect(self) -> bool:
        """连接到数据源"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    async def get_realtime_quote(self, symbol: str) -> Optional[MarketQuote]:
        """获取实时报价"""
        pass
    
    @abstractmethod
    async def get_kline_data(self, symbol: str, start_time: datetime, 
                           end_time: datetime, interval: str = '1m') -> List[KlineData]:
        """获取K线数据"""
        pass
    
    @abstractmethod
    async def subscribe_quote(self, symbol: str):
        """订阅实时报价"""
        pass
    
    @abstractmethod
    async def unsubscribe_quote(self, symbol: str):
        """取消订阅实时报价"""
        pass
    
    async def subscribe_multiple(self, symbols: List[str]):
        """批量订阅"""
        for symbol in symbols:
            await self.subscribe_quote(symbol)
    
    async def unsubscribe_multiple(self, symbols: List[str]):
        """批量取消订阅"""
        for symbol in symbols:
            await self.unsubscribe_quote(symbol)


class MockMarketDataProvider(MarketDataProvider):
    """模拟市场数据提供者 - 用于测试"""
    
    def __init__(self):
        super().__init__()
        self.mock_data = {}
    
    async def connect(self) -> bool:
        """模拟连接"""
        await asyncio.sleep(0.1)  # 模拟网络延迟
        self.is_connected = True
        return True
    
    async def disconnect(self):
        """模拟断开连接"""
        self.is_connected = False
        self.subscriptions.clear()
    
    async def get_realtime_quote(self, symbol: str) -> Optional[MarketQuote]:
        """获取模拟实时报价"""
        if not self.is_connected:
            return None
        
        # 生成模拟数据
        import random
        price = random.uniform(10, 100)
        volume = random.randint(100, 10000)
        
        return MarketQuote(
            symbol=symbol,
            timestamp=datetime.now(),
            price=price,
            volume=volume,
            bid=price - 0.01,
            ask=price + 0.01,
            bid_size=random.randint(100, 1000),
            ask_size=random.randint(100, 1000)
        )
    
    async def get_kline_data(self, symbol: str, start_time: datetime, 
                           end_time: datetime, interval: str = '1m') -> List[KlineData]:
        """获取模拟K线数据"""
        if not self.is_connected:
            return []
        
        # 生成模拟K线数据
        import random
        klines = []
        current_time = start_time
        base_price = random.uniform(50, 100)
        
        while current_time <= end_time:
            open_price = base_price + random.uniform(-2, 2)
            close_price = open_price + random.uniform(-1, 1)
            high_price = max(open_price, close_price) + random.uniform(0, 0.5)
            low_price = min(open_price, close_price) - random.uniform(0, 0.5)
            
            klines.append(KlineData(
                symbol=symbol,
                timestamp=current_time,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=random.randint(1000, 50000)
            ))
            
            # 下一个时间点
            if interval == '1m':
                current_time += timedelta(minutes=1)
            elif interval == '5m':
                current_time += timedelta(minutes=5)
            elif interval == '1h':
                current_time += timedelta(hours=1)
            elif interval == '1d':
                current_time += timedelta(days=1)
            else:
                current_time += timedelta(minutes=1)
            
            base_price = close_price  # 下一个K线的基准价格
        
        return klines
    
    async def subscribe_quote(self, symbol: str):
        """订阅模拟报价"""
        if self.is_connected:
            self.subscriptions.add(symbol)
    
    async def unsubscribe_quote(self, symbol: str):
        """取消订阅模拟报价"""
        self.subscriptions.discard(symbol)
