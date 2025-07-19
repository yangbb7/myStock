# -*- coding: utf-8 -*-
"""
缓存管理器 - 提供高性能缓存功能
"""

import time
import threading
from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict


class CacheManager:
    """高性能缓存管理器"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        """
        初始化缓存管理器
        
        Args:
            max_size: 最大缓存条目数
            ttl: 生存时间（秒）
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache = OrderedDict()
        self._timestamps = {}
        self._lock = threading.RLock()
        
        # 统计信息
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            # 检查TTL
            if self._is_expired(key):
                self._remove(key)
                self._misses += 1
                return None
            
            # 移到末尾（LRU）
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """设置缓存值"""
        with self._lock:
            current_time = time.time()
            
            if key in self._cache:
                # 更新现有值
                self._cache[key] = value
                self._timestamps[key] = current_time
                self._cache.move_to_end(key)
            else:
                # 添加新值
                if len(self._cache) >= self.max_size:
                    self._evict_oldest()
                
                self._cache[key] = value
                self._timestamps[key] = current_time
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        with self._lock:
            if key in self._cache:
                self._remove(key)
                return True
            return False
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def cleanup_expired(self) -> int:
        """清理过期缓存"""
        with self._lock:
            expired_keys = []
            for key in self._cache:
                if self._is_expired(key):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove(key)
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'ttl': self.ttl
            }
    
    def _is_expired(self, key: str) -> bool:
        """检查键是否过期"""
        if key not in self._timestamps:
            return True
        
        return time.time() - self._timestamps[key] > self.ttl
    
    def _remove(self, key: str) -> None:
        """移除键值对"""
        if key in self._cache:
            del self._cache[key]
            del self._timestamps[key]
    
    def _evict_oldest(self) -> None:
        """淘汰最旧的缓存项"""
        if self._cache:
            oldest_key = next(iter(self._cache))
            self._remove(oldest_key)
            self._evictions += 1


class PriceCache(CacheManager):
    """价格缓存专用类"""
    
    def __init__(self, max_size: int = 10000, ttl: int = 60):
        """价格缓存初始化，TTL设置为60秒"""
        super().__init__(max_size, ttl)
    
    def get_price(self, symbol: str) -> Optional[float]:
        """获取股票价格"""
        price_data = self.get(f"price:{symbol}")
        if price_data:
            return price_data.get('price')
        return None
    
    def set_price(self, symbol: str, price: float, volume: int = 0) -> None:
        """设置股票价格"""
        self.put(f"price:{symbol}", {
            'price': price,
            'volume': volume,
            'timestamp': time.time()
        })
    
    def get_ohlcv(self, symbol: str) -> Optional[Dict[str, float]]:
        """获取OHLCV数据"""
        return self.get(f"ohlcv:{symbol}")
    
    def set_ohlcv(self, symbol: str, open_price: float, high: float, 
                  low: float, close: float, volume: int) -> None:
        """设置OHLCV数据"""
        self.put(f"ohlcv:{symbol}", {
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'timestamp': time.time()
        })


class IndicatorCache(CacheManager):
    """技术指标缓存"""
    
    def __init__(self, max_size: int = 5000, ttl: int = 300):
        """技术指标缓存初始化，TTL设置为5分钟"""
        super().__init__(max_size, ttl)
    
    def get_indicator(self, symbol: str, indicator_name: str, 
                     params: str = "") -> Optional[Any]:
        """获取技术指标值"""
        key = f"indicator:{symbol}:{indicator_name}:{params}"
        return self.get(key)
    
    def set_indicator(self, symbol: str, indicator_name: str, 
                     value: Any, params: str = "") -> None:
        """设置技术指标值"""
        key = f"indicator:{symbol}:{indicator_name}:{params}"
        self.put(key, value)
    
    def get_moving_average(self, symbol: str, period: int) -> Optional[float]:
        """获取移动平均线值"""
        return self.get_indicator(symbol, "ma", str(period))
    
    def set_moving_average(self, symbol: str, period: int, value: float) -> None:
        """设置移动平均线值"""
        self.set_indicator(symbol, "ma", value, str(period))


# 全局缓存实例
price_cache = PriceCache()
indicator_cache = IndicatorCache()