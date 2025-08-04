# -*- coding: utf-8 -*-
"""
实时数据管理器 - 管理实时市场数据的接收、处理和分发
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

from .market_data_provider import MarketDataProvider, MarketQuote, KlineData


@dataclass
class DataSubscription:
    """数据订阅信息"""
    symbol: str
    data_type: str  # 'quote', 'kline', 'trade'
    callback: Callable[[Any], None]
    subscription_id: str


class RealTimeDataManager:
    """实时数据管理器"""
    
    def __init__(self, market_data_provider: MarketDataProvider):
        self.logger = logging.getLogger(__name__)
        self.provider = market_data_provider
        self.is_running = False
        
        # 订阅管理
        self.subscriptions: Dict[str, List[DataSubscription]] = defaultdict(list)
        self.subscription_counter = 0
        
        # 数据缓存
        self.latest_quotes: Dict[str, MarketQuote] = {}
        self.latest_klines: Dict[str, KlineData] = {}
        
        # 数据更新间隔（秒）
        self.update_interval = 1.0
        
        # 流状态管理
        self._streaming = False
        self._reconnecting = False
        self._subscribed_symbols: set = set()
        
        # 数据回调
        self._data_callbacks: List[Callable] = []
        
        # 数据质量监控
        self._quality_metrics: Dict[str, Dict] = defaultdict(lambda: {
            'messages_received': 0,
            'last_update_time': None
        })
        
    async def start(self):
        """启动实时数据管理器"""
        if self.is_running:
            return
        
        # 连接数据提供者
        if not await self.provider.connect():
            raise RuntimeError("无法连接到数据提供者")
        
        self.is_running = True
        self.logger.info("实时数据管理器已启动")
        
        # 启动数据更新循环
        asyncio.create_task(self._data_update_loop())
    
    async def stop(self):
        """停止实时数据管理器"""
        self.is_running = False
        await self.provider.disconnect()
        self.logger.info("实时数据管理器已停止")
    
    async def _data_update_loop(self):
        """数据更新循环"""
        while self.is_running:
            try:
                await self._update_subscribed_data()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"数据更新循环出错: {e}")
                await asyncio.sleep(1)
    
    async def _update_subscribed_data(self):
        """更新订阅的数据"""
        # 获取所有订阅的股票代码
        symbols = set(self.subscriptions.keys())
        
        # 并发获取数据
        tasks = []
        for symbol in symbols:
            tasks.append(self._update_symbol_data(symbol))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _update_symbol_data(self, symbol: str):
        """更新单个股票的数据"""
        try:
            # 获取实时报价
            quote = await self.provider.get_realtime_quote(symbol)
            if quote:
                self.latest_quotes[symbol] = quote
                await self._notify_subscribers(symbol, 'quote', quote)
                
        except Exception as e:
            self.logger.error(f"更新{symbol}数据失败: {e}")
    
    async def _notify_subscribers(self, symbol: str, data_type: str, data: Any):
        """通知订阅者"""
        for subscription in self.subscriptions[symbol]:
            if subscription.data_type == data_type:
                try:
                    subscription.callback(data)
                except Exception as e:
                    self.logger.error(f"订阅回调出错: {e}")
    
    def subscribe_quote(self, symbol: str, callback: Callable[[MarketQuote], None]) -> str:
        """订阅实时报价"""
        subscription_id = f"quote_{symbol}_{self.subscription_counter}"
        self.subscription_counter += 1
        
        subscription = DataSubscription(
            symbol=symbol,
            data_type='quote',
            callback=callback,
            subscription_id=subscription_id
        )
        
        self.subscriptions[symbol].append(subscription)
        
        # 如果是第一次订阅这个股票，告知数据提供者
        if len(self.subscriptions[symbol]) == 1:
            asyncio.create_task(self.provider.subscribe_quote(symbol))
        
        self.logger.info(f"订阅报价: {symbol}")
        return subscription_id
    
    def unsubscribe(self, subscription_id: str):
        """取消订阅"""
        for symbol, subs in self.subscriptions.items():
            for i, sub in enumerate(subs):
                if sub.subscription_id == subscription_id:
                    del subs[i]
                    
                    # 如果这个股票没有订阅者了，取消数据提供者的订阅
                    if len(subs) == 0:
                        asyncio.create_task(self.provider.unsubscribe_quote(symbol))
                        del self.subscriptions[symbol]
                    
                    self.logger.info(f"取消订阅: {subscription_id}")
                    return
    
    def get_latest_quote(self, symbol: str) -> Optional[MarketQuote]:
        """获取最新报价"""
        return self.latest_quotes.get(symbol)
    
    def get_latest_quotes(self, symbols: List[str]) -> Dict[str, MarketQuote]:
        """批量获取最新报价"""
        return {symbol: self.latest_quotes[symbol] 
                for symbol in symbols if symbol in self.latest_quotes}
    
    def get_all_latest_quotes(self) -> Dict[str, MarketQuote]:
        """获取所有最新报价"""
        return self.latest_quotes.copy()
    
    def get_subscription_status(self) -> Dict[str, Any]:
        """获取订阅状态"""
        return {
            'is_running': self.is_running,
            'provider_connected': self.provider.is_connected,
            'subscribed_symbols': list(self.subscriptions.keys()),
            'total_subscriptions': sum(len(subs) for subs in self.subscriptions.values()),
            'latest_quotes_count': len(self.latest_quotes)
        }
    
    async def start_stream(self, symbols: List[str]):
        """启动实时数据流"""
        self._streaming = True
        self._subscribed_symbols.update(symbols)
        
        # 为每个符号订阅数据
        for symbol in symbols:
            if symbol not in self.subscriptions:
                # 使用默认回调订阅
                self.subscribe_quote(symbol, lambda data: None)
        
        if not self.is_running:
            await self.start()
        
        self.logger.info(f"启动数据流，订阅股票: {symbols}")
    
    def is_streaming(self) -> bool:
        """检查是否正在流式传输数据"""
        return self._streaming
    
    def is_reconnecting(self) -> bool:
        """检查是否正在重连"""
        return self._reconnecting
    
    def get_subscribed_symbols(self) -> List[str]:
        """获取已订阅的股票代码列表"""
        return list(self._subscribed_symbols)
    
    async def add_subscription(self, symbol: str):
        """添加订阅"""
        if symbol not in self._subscribed_symbols:
            self._subscribed_symbols.add(symbol)
            self.subscribe_quote(symbol, lambda data: None)
            self.logger.info(f"添加订阅: {symbol}")
    
    async def remove_subscription(self, symbol: str):
        """移除订阅"""
        if symbol in self._subscribed_symbols:
            self._subscribed_symbols.remove(symbol)
            
            # 移除所有相关订阅
            if symbol in self.subscriptions:
                for sub in self.subscriptions[symbol].copy():
                    self.unsubscribe(sub.subscription_id)
            
            self.logger.info(f"移除订阅: {symbol}")
    
    def add_data_callback(self, callback: Callable):
        """添加数据回调函数"""
        self._data_callbacks.append(callback)
    
    async def _handle_data_update(self, symbol: str, data: Any):
        """处理数据更新"""
        # 更新质量指标
        self._quality_metrics[symbol]['messages_received'] += 1
        self._quality_metrics[symbol]['last_update_time'] = datetime.now()
        
        # 调用所有数据回调
        for callback in self._data_callbacks:
            try:
                callback(symbol, data)
            except Exception as e:
                self.logger.error(f"数据回调出错: {e}")
    
    async def _handle_connection_lost(self):
        """处理连接丢失"""
        self._reconnecting = True
        self._streaming = False
        self.logger.warning("连接丢失，准备重连")
    
    async def _handle_connection_restored(self):
        """处理连接恢复"""
        self._reconnecting = False
        self._streaming = True
        self.logger.info("连接已恢复")
    
    def get_data_quality_metrics(self, symbol: str) -> Dict[str, Any]:
        """获取数据质量指标"""
        return self._quality_metrics.get(symbol, {
            'messages_received': 0,
            'last_update_time': None
        })
