# -*- coding: utf-8 -*-
"""
对象池 - 减少对象创建开销
"""

import threading
from typing import Any, Callable, Generic, List, Optional, TypeVar
from queue import Queue, Empty
from datetime import datetime

T = TypeVar('T')


class ObjectPool(Generic[T]):
    """通用对象池"""
    
    def __init__(self, factory: Callable[[], T], max_size: int = 100):
        """
        初始化对象池
        
        Args:
            factory: 对象创建工厂函数
            max_size: 最大对象数量
        """
        self._factory = factory
        self._max_size = max_size
        self._pool = Queue(maxsize=max_size)
        self._created_count = 0
        self._lock = threading.Lock()
        
        # 统计信息
        self._borrowed_count = 0
        self._returned_count = 0
        self._created_total = 0
    
    def borrow(self) -> T:
        """借用对象"""
        with self._lock:
            self._borrowed_count += 1
        
        try:
            # 尝试从池中获取对象
            obj = self._pool.get_nowait()
            return obj
        except Empty:
            # 池中没有对象，创建新对象
            with self._lock:
                self._created_count += 1
                self._created_total += 1
            return self._factory()
    
    def return_object(self, obj: T) -> None:
        """归还对象"""
        with self._lock:
            self._returned_count += 1
        
        try:
            # 重置对象状态（如果需要）
            if hasattr(obj, 'reset'):
                obj.reset()
            
            # 如果池未满，归还对象
            if self._pool.qsize() < self._max_size:
                self._pool.put_nowait(obj)
                with self._lock:
                    self._created_count -= 1
        except:
            # 池已满，丢弃对象
            pass
    
    def get_stats(self) -> dict:
        """获取池统计信息"""
        with self._lock:
            return {
                'pool_size': self._pool.qsize(),
                'max_size': self._max_size,
                'active_objects': self._created_count,
                'borrowed_count': self._borrowed_count,
                'returned_count': self._returned_count,
                'created_total': self._created_total,
                'utilization': self._borrowed_count / max(self._created_total, 1)
            }
    
    def clear(self) -> None:
        """清空对象池"""
        with self._lock:
            while not self._pool.empty():
                try:
                    self._pool.get_nowait()
                except Empty:
                    break
            self._created_count = 0


class TickDataPool(ObjectPool):
    """Tick数据对象池"""
    
    def __init__(self, max_size: int = 1000):
        def create_tick_data():
            return {
                'symbol': '',
                'timestamp': None,
                'open': 0.0,
                'high': 0.0,
                'low': 0.0,
                'close': 0.0,
                'volume': 0
            }
        
        super().__init__(create_tick_data, max_size)
    
    def get_tick_data(self, symbol: str, timestamp: datetime, 
                     open_price: float, high: float, low: float, 
                     close: float, volume: int) -> dict:
        """获取配置好的tick数据对象"""
        tick = self.borrow()
        tick['symbol'] = symbol
        tick['timestamp'] = timestamp
        tick['open'] = open_price
        tick['high'] = high
        tick['low'] = low
        tick['close'] = close
        tick['volume'] = volume
        return tick


class SignalPool(ObjectPool):
    """交易信号对象池"""
    
    def __init__(self, max_size: int = 500):
        def create_signal():
            return {
                'symbol': '',
                'signal_type': '',
                'timestamp': None,
                'price': 0.0,
                'quantity': 0,
                'strategy_name': '',
                'confidence': 0.0,
                'metadata': {}
            }
        
        super().__init__(create_signal, max_size)
    
    def get_signal(self, symbol: str, signal_type: str, timestamp: datetime,
                  price: float, quantity: int, strategy_name: str,
                  confidence: float = 1.0, metadata: dict = None) -> dict:
        """获取配置好的交易信号对象"""
        signal = self.borrow()
        signal['symbol'] = symbol
        signal['signal_type'] = signal_type
        signal['timestamp'] = timestamp
        signal['price'] = price
        signal['quantity'] = quantity
        signal['strategy_name'] = strategy_name
        signal['confidence'] = confidence
        signal['metadata'] = metadata or {}
        return signal


class OrderPool(ObjectPool):
    """订单对象池"""
    
    def __init__(self, max_size: int = 1000):
        def create_order():
            return {
                'order_id': '',
                'symbol': '',
                'side': '',
                'quantity': 0,
                'price': 0.0,
                'order_type': 'MARKET',
                'timestamp': None,
                'status': 'CREATED',
                'filled_quantity': 0,
                'avg_fill_price': 0.0,
                'commission': 0.0
            }
        
        super().__init__(create_order, max_size)
    
    def get_order(self, order_id: str, symbol: str, side: str,
                 quantity: int, price: float = 0.0, 
                 order_type: str = 'MARKET') -> dict:
        """获取配置好的订单对象"""
        order = self.borrow()
        order['order_id'] = order_id
        order['symbol'] = symbol
        order['side'] = side
        order['quantity'] = quantity
        order['price'] = price
        order['order_type'] = order_type
        order['timestamp'] = datetime.now()
        order['status'] = 'CREATED'
        order['filled_quantity'] = 0
        order['avg_fill_price'] = 0.0
        order['commission'] = 0.0
        return order


# 全局对象池实例
tick_pool = TickDataPool()
signal_pool = SignalPool()
order_pool = OrderPool()