# -*- coding: utf-8 -*-
"""
批处理器 - 批量处理提高效率
"""

import threading
import time
from typing import Any, Callable, List, Optional
from queue import Queue, Empty
from datetime import datetime


class BatchProcessor:
    """批处理器基类"""
    
    def __init__(self, batch_size: int = 100, max_wait_time: float = 0.1,
                 processor_func: Optional[Callable] = None):
        """
        初始化批处理器
        
        Args:
            batch_size: 批处理大小
            max_wait_time: 最大等待时间（秒）
            processor_func: 批处理函数
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.processor_func = processor_func
        
        self._queue = Queue()
        self._batch_buffer = []
        self._last_process_time = time.time()
        self._running = False
        self._worker_thread = None
        self._lock = threading.Lock()
        
        # 统计信息
        self._items_processed = 0
        self._batches_processed = 0
        self._average_batch_size = 0
    
    def start(self) -> None:
        """启动批处理器"""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop)
        self._worker_thread.daemon = True
        self._worker_thread.start()
    
    def stop(self) -> None:
        """停止批处理器"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        
        # 处理剩余的项目
        self._process_remaining()
    
    def submit(self, item: Any) -> None:
        """提交项目到批处理器"""
        if not self._running:
            self.start()
        
        self._queue.put(item)
    
    def _worker_loop(self) -> None:
        """工作线程循环"""
        while self._running:
            try:
                # 尝试获取项目
                try:
                    item = self._queue.get(timeout=0.01)
                    self._batch_buffer.append(item)
                except Empty:
                    # 队列为空，检查是否需要处理现有批次
                    pass
                
                # 检查是否需要处理批次
                current_time = time.time()
                should_process = (
                    len(self._batch_buffer) >= self.batch_size or
                    (self._batch_buffer and 
                     current_time - self._last_process_time >= self.max_wait_time)
                )
                
                if should_process:
                    self._process_batch()
                
            except Exception as e:
                print(f"BatchProcessor worker error: {e}")
    
    def _process_batch(self) -> None:
        """处理批次"""
        if not self._batch_buffer:
            return
        
        with self._lock:
            batch = self._batch_buffer.copy()
            self._batch_buffer.clear()
            self._last_process_time = time.time()
        
        if self.processor_func:
            try:
                self.processor_func(batch)
                
                # 更新统计信息
                self._items_processed += len(batch)
                self._batches_processed += 1
                self._average_batch_size = (
                    self._items_processed / self._batches_processed
                )
                
            except Exception as e:
                print(f"Batch processing error: {e}")
    
    def _process_remaining(self) -> None:
        """处理剩余项目"""
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                self._batch_buffer.append(item)
            except Empty:
                break
        
        if self._batch_buffer:
            self._process_batch()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'items_processed': self._items_processed,
            'batches_processed': self._batches_processed,
            'average_batch_size': self._average_batch_size,
            'current_buffer_size': len(self._batch_buffer),
            'queue_size': self._queue.qsize(),
            'running': self._running
        }


class TickBatchProcessor(BatchProcessor):
    """Tick数据批处理器"""
    
    def __init__(self, batch_size: int = 50, max_wait_time: float = 0.05):
        def process_tick_batch(ticks: List[dict]):
            """批量处理tick数据"""
            # 按symbol分组
            by_symbol = {}
            for tick in ticks:
                symbol = tick.get('symbol', '')
                if symbol not in by_symbol:
                    by_symbol[symbol] = []
                by_symbol[symbol].append(tick)
            
            # 处理每个symbol的tick数据
            for symbol, symbol_ticks in by_symbol.items():
                # 计算统计信息
                prices = [tick['close'] for tick in symbol_ticks]
                volumes = [tick['volume'] for tick in symbol_ticks]
                
                # 更新缓存或数据库
                # 这里可以添加具体的处理逻辑
                pass
        
        super().__init__(batch_size, max_wait_time, process_tick_batch)


class SignalBatchProcessor(BatchProcessor):
    """交易信号批处理器"""
    
    def __init__(self, batch_size: int = 20, max_wait_time: float = 0.1):
        def process_signal_batch(signals: List[dict]):
            """批量处理交易信号"""
            # 按策略分组
            by_strategy = {}
            for signal in signals:
                strategy = signal.get('strategy_name', '')
                if strategy not in by_strategy:
                    by_strategy[strategy] = []
                by_strategy[strategy].append(signal)
            
            # 处理每个策略的信号
            for strategy, strategy_signals in by_strategy.items():
                # 信号去重和合并
                unique_signals = self._deduplicate_signals(strategy_signals)
                
                # 风险检查
                validated_signals = self._validate_signals(unique_signals)
                
                # 生成订单
                for signal in validated_signals:
                    self._create_order_from_signal(signal)
        
        super().__init__(batch_size, max_wait_time, process_signal_batch)
    
    def _deduplicate_signals(self, signals: List[dict]) -> List[dict]:
        """信号去重"""
        seen = set()
        unique_signals = []
        
        for signal in signals:
            key = (signal.get('symbol'), signal.get('signal_type'))
            if key not in seen:
                seen.add(key)
                unique_signals.append(signal)
        
        return unique_signals
    
    def _validate_signals(self, signals: List[dict]) -> List[dict]:
        """验证信号"""
        validated = []
        for signal in signals:
            # 基本验证
            if (signal.get('symbol') and 
                signal.get('signal_type') and 
                signal.get('price', 0) > 0):
                validated.append(signal)
        
        return validated
    
    def _create_order_from_signal(self, signal: dict) -> None:
        """从信号创建订单"""
        # 这里可以调用订单管理器创建订单
        pass


class OrderBatchProcessor(BatchProcessor):
    """订单批处理器"""
    
    def __init__(self, batch_size: int = 30, max_wait_time: float = 0.2):
        def process_order_batch(orders: List[dict]):
            """批量处理订单"""
            # 按券商分组
            by_broker = {}
            for order in orders:
                broker = order.get('broker', 'default')
                if broker not in by_broker:
                    by_broker[broker] = []
                by_broker[broker].append(order)
            
            # 向每个券商批量提交订单
            for broker, broker_orders in by_broker.items():
                try:
                    # 这里可以调用券商API批量提交
                    self._submit_orders_to_broker(broker, broker_orders)
                except Exception as e:
                    print(f"Failed to submit orders to {broker}: {e}")
        
        super().__init__(batch_size, max_wait_time, process_order_batch)
    
    def _submit_orders_to_broker(self, broker: str, orders: List[dict]) -> None:
        """向券商提交订单"""
        # 模拟券商API调用
        print(f"Submitting {len(orders)} orders to {broker}")
        for order in orders:
            # 更新订单状态
            order['status'] = 'SUBMITTED'
            order['submit_time'] = datetime.now()


# 全局批处理器实例
tick_batch_processor = TickBatchProcessor()
signal_batch_processor = SignalBatchProcessor() 
order_batch_processor = OrderBatchProcessor()