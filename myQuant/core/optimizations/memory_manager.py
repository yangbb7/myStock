# -*- coding: utf-8 -*-
"""
内存管理器 - 内存优化和监控
"""

import gc
import threading
import time
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import weakref


class MemoryManager:
    """内存管理器"""
    
    def __init__(self, cleanup_interval: int = 300):
        """
        初始化内存管理器
        
        Args:
            cleanup_interval: 清理间隔（秒）
        """
        self.cleanup_interval = cleanup_interval
        self._running = False
        self._cleanup_thread = None
        self._weak_refs = weakref.WeakSet()
        self._last_cleanup = time.time()
        
        # 内存统计
        self._memory_stats = {
            'peak_memory': 0,
            'current_memory': 0,
            'gc_collections': 0,
            'objects_cleaned': 0
        }
        
        # 自动清理规则
        self._cleanup_rules = []
    
    def start(self) -> None:
        """启动内存管理器"""
        if self._running:
            return
        
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop)
        self._cleanup_thread.daemon = True
        self._cleanup_thread.start()
    
    def stop(self) -> None:
        """停止内存管理器"""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
    
    def register_object(self, obj: Any) -> None:
        """注册对象到弱引用集合"""
        self._weak_refs.add(obj)
    
    def add_cleanup_rule(self, rule_func, interval: int = 300) -> None:
        """添加清理规则"""
        self._cleanup_rules.append({
            'func': rule_func,
            'interval': interval,
            'last_run': 0
        })
    
    def force_cleanup(self) -> Dict[str, int]:
        """强制执行清理"""
        return self._perform_cleanup()
    
    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            current_memory = memory_info.rss / 1024 / 1024  # MB
            self._memory_stats['current_memory'] = current_memory
            
            if current_memory > self._memory_stats['peak_memory']:
                self._memory_stats['peak_memory'] = current_memory
            
            return {
                'current_memory_mb': current_memory,
                'peak_memory_mb': self._memory_stats['peak_memory'],
                'memory_percent': process.memory_percent(),
                'gc_collections': self._memory_stats['gc_collections'],
                'objects_cleaned': self._memory_stats['objects_cleaned'],
                'weak_refs_count': len(self._weak_refs)
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    def _cleanup_loop(self) -> None:
        """清理循环"""
        while self._running:
            try:
                current_time = time.time()
                
                # 检查是否需要清理
                if current_time - self._last_cleanup >= self.cleanup_interval:
                    self._perform_cleanup()
                    self._last_cleanup = current_time
                
                # 执行自定义清理规则
                self._run_cleanup_rules(current_time)
                
                time.sleep(10)  # 每10秒检查一次
                
            except Exception as e:
                print(f"Memory cleanup error: {e}")
    
    def _perform_cleanup(self) -> Dict[str, int]:
        """执行清理"""
        stats = {'gc_collected': 0, 'objects_cleaned': 0}
        
        # 强制垃圾回收
        before_gc = len(gc.get_objects())
        collected = gc.collect()
        after_gc = len(gc.get_objects())
        
        stats['gc_collected'] = collected
        stats['objects_cleaned'] = before_gc - after_gc
        
        self._memory_stats['gc_collections'] += 1
        self._memory_stats['objects_cleaned'] += stats['objects_cleaned']
        
        return stats
    
    def _run_cleanup_rules(self, current_time: float) -> None:
        """运行清理规则"""
        for rule in self._cleanup_rules:
            if current_time - rule['last_run'] >= rule['interval']:
                try:
                    rule['func']()
                    rule['last_run'] = current_time
                except Exception as e:
                    print(f"Cleanup rule error: {e}")


class DataRetentionManager:
    """数据保留管理器"""
    
    def __init__(self):
        self.retention_policies = {}
        self._data_stores = {}
    
    def set_retention_policy(self, data_type: str, retention_days: int) -> None:
        """设置数据保留策略"""
        self.retention_policies[data_type] = retention_days
    
    def register_data_store(self, name: str, store: Any) -> None:
        """注册数据存储"""
        self._data_stores[name] = store
    
    def cleanup_old_data(self) -> Dict[str, int]:
        """清理旧数据"""
        cleanup_stats = {}
        
        for data_type, retention_days in self.retention_policies.items():
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # 清理各种数据类型
            if data_type == 'tick_data':
                cleaned = self._cleanup_tick_data(cutoff_date)
            elif data_type == 'order_history':
                cleaned = self._cleanup_order_history(cutoff_date)
            elif data_type == 'trade_history':
                cleaned = self._cleanup_trade_history(cutoff_date)
            else:
                cleaned = 0
            
            cleanup_stats[data_type] = cleaned
        
        return cleanup_stats
    
    def _cleanup_tick_data(self, cutoff_date: datetime) -> int:
        """清理tick数据"""
        # 这里实现具体的tick数据清理逻辑
        return 0
    
    def _cleanup_order_history(self, cutoff_date: datetime) -> int:
        """清理订单历史"""
        # 这里实现具体的订单历史清理逻辑
        return 0
    
    def _cleanup_trade_history(self, cutoff_date: datetime) -> int:
        """清理交易历史"""
        # 这里实现具体的交易历史清理逻辑
        return 0


class CircularBuffer:
    """循环缓冲区 - 固定大小，自动覆盖旧数据"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._buffer = [None] * max_size
        self._head = 0
        self._tail = 0
        self._size = 0
        self._lock = threading.Lock()
    
    def append(self, item: Any) -> None:
        """添加项目到缓冲区"""
        with self._lock:
            self._buffer[self._tail] = item
            self._tail = (self._tail + 1) % self.max_size
            
            if self._size < self.max_size:
                self._size += 1
            else:
                # 缓冲区已满，移动head
                self._head = (self._head + 1) % self.max_size
    
    def get_all(self) -> List[Any]:
        """获取所有项目"""
        with self._lock:
            if self._size == 0:
                return []
            
            if self._size < self.max_size:
                return [item for item in self._buffer[:self._size] if item is not None]
            else:
                # 缓冲区已满
                return ([item for item in self._buffer[self._head:] if item is not None] +
                       [item for item in self._buffer[:self._head] if item is not None])
    
    def get_latest(self, n: int) -> List[Any]:
        """获取最新的n个项目"""
        all_items = self.get_all()
        return all_items[-n:] if len(all_items) >= n else all_items
    
    def clear(self) -> None:
        """清空缓冲区"""
        with self._lock:
            self._buffer = [None] * self.max_size
            self._head = 0
            self._tail = 0
            self._size = 0
    
    def size(self) -> int:
        """获取当前大小"""
        return self._size
    
    def is_full(self) -> bool:
        """检查是否已满"""
        return self._size == self.max_size


# 全局实例
memory_manager = MemoryManager()
data_retention_manager = DataRetentionManager()

# 设置默认数据保留策略
data_retention_manager.set_retention_policy('tick_data', 30)  # 30天
data_retention_manager.set_retention_policy('order_history', 90)  # 90天
data_retention_manager.set_retention_policy('trade_history', 365)  # 365天