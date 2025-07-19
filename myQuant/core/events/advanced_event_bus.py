# -*- coding: utf-8 -*-
"""
高级事件总线 - 支持复杂交易逻辑的增强事件处理系统
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from queue import PriorityQueue, Empty
import heapq
from contextlib import contextmanager

from .enhanced_event_types import (
    BaseEvent, EventPriority, EventCategory, EventStatus,
    CompositeEvent, EventFactory
)


class EventFilter:
    """事件过滤器"""
    
    def __init__(self, 
                 categories: Optional[List[EventCategory]] = None,
                 priorities: Optional[List[EventPriority]] = None,
                 sources: Optional[List[str]] = None,
                 custom_filter: Optional[Callable[[BaseEvent], bool]] = None):
        self.categories = set(categories) if categories else None
        self.priorities = set(priorities) if priorities else None
        self.sources = set(sources) if sources else None
        self.custom_filter = custom_filter
    
    def match(self, event: BaseEvent) -> bool:
        """检查事件是否匹配过滤条件"""
        if self.categories and event.category not in self.categories:
            return False
        if self.priorities and event.priority not in self.priorities:
            return False
        if self.sources and event.source not in self.sources:
            return False
        if self.custom_filter and not self.custom_filter(event):
            return False
        return True


class EventSubscription:
    """事件订阅"""
    
    def __init__(self,
                 handler: Callable[[BaseEvent], Any],
                 event_filter: Optional[EventFilter] = None,
                 async_handler: bool = False,
                 max_retries: int = 3,
                 timeout: Optional[float] = None):
        self.handler = handler
        self.event_filter = event_filter or EventFilter()
        self.async_handler = async_handler
        self.max_retries = max_retries
        self.timeout = timeout
        self.subscription_id = f"sub_{id(self)}"
        self.created_time = datetime.now()
        
        # 统计信息
        self.processed_count = 0
        self.error_count = 0
        self.last_processed_time = None
        self.avg_processing_time = 0.0


class EventMetrics:
    """事件指标统计"""
    
    def __init__(self):
        self.total_events = 0
        self.events_by_category = defaultdict(int)
        self.events_by_priority = defaultdict(int)
        self.processing_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.queue_sizes = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=100)
        
        self.start_time = datetime.now()
        self.last_reset_time = datetime.now()
    
    def record_event(self, event: BaseEvent) -> None:
        """记录事件"""
        self.total_events += 1
        self.events_by_category[event.category] += 1
        self.events_by_priority[event.priority] += 1
    
    def record_processing_time(self, event: BaseEvent, processing_time: float) -> None:
        """记录处理时间"""
        self.processing_times[event.category].append(processing_time)
        
        # 保持最近1000条记录
        if len(self.processing_times[event.category]) > 1000:
            self.processing_times[event.category] = self.processing_times[event.category][-1000:]
    
    def record_error(self, event: BaseEvent) -> None:
        """记录错误"""
        self.error_counts[event.category] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        current_time = datetime.now()
        uptime = (current_time - self.start_time).total_seconds()
        
        avg_processing_times = {}
        for category, times in self.processing_times.items():
            if times:
                avg_processing_times[category.value] = sum(times) / len(times)
        
        return {
            'total_events': self.total_events,
            'uptime_seconds': uptime,
            'events_per_second': self.total_events / uptime if uptime > 0 else 0,
            'events_by_category': {k.value: v for k, v in self.events_by_category.items()},
            'events_by_priority': {k.value: v for k, v in self.events_by_priority.items()},
            'avg_processing_times': avg_processing_times,
            'error_counts': {k.value: v for k, v in self.error_counts.items()},
            'current_queue_size': self.queue_sizes[-1] if self.queue_sizes else 0
        }


class AdvancedEventBus:
    """
    高级事件总线 - 支持优先级队列、异步处理、事件过滤、重试机制等高级功能
    """
    
    def __init__(self,
                 max_workers: int = 10,
                 max_queue_size: int = 10000,
                 enable_persistence: bool = False,
                 enable_metrics: bool = True):
        """
        初始化高级事件总线
        
        Args:
            max_workers: 最大工作线程数
            max_queue_size: 最大队列大小
            enable_persistence: 是否启用事件持久化
            enable_metrics: 是否启用指标统计
        """
        # 核心组件
        self.event_queue = PriorityQueue(maxsize=max_queue_size)
        self.subscriptions: Dict[str, List[EventSubscription]] = defaultdict(list)
        self.global_subscriptions: List[EventSubscription] = []
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.async_executor = ThreadPoolExecutor(max_workers=max_workers//2 + 1)
        
        # 控制标志
        self.running = False
        self.paused = False
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        
        # 线程同步
        self._lock = threading.RLock()
        self._subscription_lock = threading.RLock()
        
        # 事件存储和追踪
        self.event_history: deque = deque(maxlen=10000)
        self.active_events: Dict[str, BaseEvent] = {}
        self.composite_events: Dict[str, CompositeEvent] = {}
        
        # 指标统计
        self.metrics = EventMetrics() if enable_metrics else None
        
        # 持久化配置
        self.enable_persistence = enable_persistence
        self.persistence_handler = None
        
        # 重试机制
        self.retry_queue = PriorityQueue()
        self.failed_events: Dict[str, BaseEvent] = {}
        
        # 事件处理工作线程
        self._worker_threads: List[threading.Thread] = []
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Advanced EventBus initialized with {max_workers} workers")
    
    def start(self) -> None:
        """启动事件总线"""
        if self.running:
            return
            
        self.running = True
        self._stop_event.clear()
        
        # 启动工作线程
        main_worker = threading.Thread(target=self._main_event_loop, daemon=True)
        main_worker.start()
        self._worker_threads.append(main_worker)
        
        retry_worker = threading.Thread(target=self._retry_event_loop, daemon=True)
        retry_worker.start()
        self._worker_threads.append(retry_worker)
        
        if self.metrics:
            metrics_worker = threading.Thread(target=self._metrics_loop, daemon=True)
            metrics_worker.start()
            self._worker_threads.append(metrics_worker)
        
        self.logger.info("EventBus started")
    
    def stop(self, timeout: float = 30.0) -> None:
        """停止事件总线"""
        if not self.running:
            return
            
        self.logger.info("Stopping EventBus...")
        
        self.running = False
        self._stop_event.set()
        
        # 等待工作线程完成
        for thread in self._worker_threads:
            thread.join(timeout=timeout)
        
        # 关闭线程池
        self.executor.shutdown(wait=True, timeout=timeout)
        self.async_executor.shutdown(wait=True, timeout=timeout)
        
        self._worker_threads.clear()
        self.logger.info("EventBus stopped")
    
    def pause(self) -> None:
        """暂停事件处理"""
        self.paused = True
        self._pause_event.set()
        self.logger.info("EventBus paused")
    
    def resume(self) -> None:
        """恢复事件处理"""
        self.paused = False
        self._pause_event.clear()
        self.logger.info("EventBus resumed")
    
    def publish(self, event: BaseEvent, immediate: bool = False) -> str:
        """
        发布事件
        
        Args:
            event: 要发布的事件
            immediate: 是否立即处理
            
        Returns:
            str: 事件ID
        """
        if not event.validate():
            raise ValueError(f"Invalid event: {event}")
        
        # 记录事件
        with self._lock:
            self.active_events[event.event_id] = event
            self.event_history.append(event)
            
            if self.metrics:
                self.metrics.record_event(event)
        
        # 处理复合事件
        if isinstance(event, CompositeEvent):
            self.composite_events[event.event_id] = event
        
        # 立即处理或加入队列
        if immediate:
            self._process_event_immediate(event)
        else:
            # 使用优先级队列
            priority = (event.priority.value, event.timestamp.timestamp())
            try:
                self.event_queue.put((priority, event), block=False)
            except:
                self.logger.error(f"Event queue full, dropping event {event.event_id}")
                return event.event_id
        
        self.logger.debug(f"Published event {event.event_id} of type {type(event).__name__}")
        return event.event_id
    
    def subscribe(self,
                  handler: Callable[[BaseEvent], Any],
                  event_types: Optional[List[type]] = None,
                  event_filter: Optional[EventFilter] = None,
                  async_handler: bool = False) -> str:
        """
        订阅事件
        
        Args:
            handler: 事件处理器
            event_types: 订阅的事件类型列表
            event_filter: 事件过滤器
            async_handler: 是否为异步处理器
            
        Returns:
            str: 订阅ID
        """
        subscription = EventSubscription(
            handler=handler,
            event_filter=event_filter,
            async_handler=async_handler
        )
        
        with self._subscription_lock:
            if event_types:
                for event_type in event_types:
                    type_name = event_type.__name__
                    self.subscriptions[type_name].append(subscription)
            else:
                # 全局订阅
                self.global_subscriptions.append(subscription)
        
        self.logger.info(f"Added subscription {subscription.subscription_id}")
        return subscription.subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """取消订阅"""
        with self._subscription_lock:
            # 从类型订阅中移除
            for type_subscriptions in self.subscriptions.values():
                type_subscriptions[:] = [
                    sub for sub in type_subscriptions 
                    if sub.subscription_id != subscription_id
                ]
            
            # 从全局订阅中移除
            self.global_subscriptions[:] = [
                sub for sub in self.global_subscriptions
                if sub.subscription_id != subscription_id
            ]
        
        self.logger.info(f"Removed subscription {subscription_id}")
        return True
    
    def get_event(self, event_id: str) -> Optional[BaseEvent]:
        """获取事件"""
        return self.active_events.get(event_id)
    
    def get_event_status(self, event_id: str) -> Optional[EventStatus]:
        """获取事件状态"""
        event = self.active_events.get(event_id)
        return event.status if event else None
    
    def cancel_event(self, event_id: str) -> bool:
        """取消事件"""
        event = self.active_events.get(event_id)
        if event and event.status in [EventStatus.CREATED, EventStatus.PENDING]:
            event.status = EventStatus.CANCELLED
            return True
        return False
    
    def _main_event_loop(self) -> None:
        """主事件处理循环"""
        while self.running:
            try:
                if self.paused:
                    self._pause_event.wait(timeout=1.0)
                    continue
                
                # 获取事件
                try:
                    priority, event = self.event_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # 处理事件
                self._process_event(event)
                self.event_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in main event loop: {e}")
    
    def _retry_event_loop(self) -> None:
        """重试事件处理循环"""
        while self.running:
            try:
                if self.paused:
                    self._pause_event.wait(timeout=1.0)
                    continue
                
                # 获取重试事件
                try:
                    retry_time, event = self.retry_queue.get(timeout=5.0)
                except Empty:
                    continue
                
                # 检查是否到了重试时间
                if time.time() < retry_time:
                    # 重新放回队列
                    self.retry_queue.put((retry_time, event))
                    time.sleep(1.0)
                    continue
                
                # 重新处理事件
                if event.can_retry():
                    self._process_event(event)
                else:
                    # 超过最大重试次数，标记为失败
                    event.mark_failed("Exceeded maximum retry attempts")
                    self.failed_events[event.event_id] = event
                    self.logger.error(f"Event {event.event_id} failed permanently")
                
                self.retry_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in retry event loop: {e}")
    
    def _metrics_loop(self) -> None:
        """指标统计循环"""
        while self.running:
            try:
                if self.metrics:
                    # 记录队列大小
                    self.metrics.queue_sizes.append(self.event_queue.qsize())
                
                time.sleep(10.0)  # 每10秒记录一次
                
            except Exception as e:
                self.logger.error(f"Error in metrics loop: {e}")
    
    def _process_event(self, event: BaseEvent) -> None:
        """处理单个事件"""
        start_time = time.time()
        
        try:
            event.mark_processing("EventBus")
            
            # 获取订阅者
            subscriptions = self._get_subscriptions_for_event(event)
            
            if not subscriptions:
                event.mark_completed()
                return
            
            # 处理订阅者
            futures = []
            for subscription in subscriptions:
                try:
                    if subscription.async_handler:
                        future = self.async_executor.submit(
                            self._handle_subscription_async, subscription, event
                        )
                    else:
                        future = self.executor.submit(
                            self._handle_subscription, subscription, event
                        )
                    futures.append(future)
                    
                except Exception as e:
                    self.logger.error(f"Error submitting handler: {e}")
            
            # 等待所有处理器完成
            for future in futures:
                try:
                    future.result(timeout=subscription.timeout)
                except Exception as e:
                    self.logger.error(f"Handler execution failed: {e}")
            
            # 处理复合事件
            if isinstance(event, CompositeEvent):
                self._process_composite_event(event)
            
            event.mark_completed()
            
        except Exception as e:
            event.mark_failed(str(e))
            
            # 加入重试队列
            if event.can_retry():
                retry_time = time.time() + (2 ** event.retry_count)  # 指数退避
                self.retry_queue.put((retry_time, event))
            else:
                self.failed_events[event.event_id] = event
            
            if self.metrics:
                self.metrics.record_error(event)
        
        finally:
            processing_time = time.time() - start_time
            event.processing_time = processing_time
            
            if self.metrics:
                self.metrics.record_processing_time(event, processing_time)
    
    def _process_event_immediate(self, event: BaseEvent) -> None:
        """立即处理事件"""
        subscriptions = self._get_subscriptions_for_event(event)
        
        for subscription in subscriptions:
            try:
                self._handle_subscription(subscription, event)
            except Exception as e:
                self.logger.error(f"Immediate handler failed: {e}")
    
    def _get_subscriptions_for_event(self, event: BaseEvent) -> List[EventSubscription]:
        """获取事件的订阅者"""
        matching_subscriptions = []
        
        with self._subscription_lock:
            # 类型特定订阅
            event_type_name = type(event).__name__
            for subscription in self.subscriptions.get(event_type_name, []):
                if subscription.event_filter.match(event):
                    matching_subscriptions.append(subscription)
            
            # 全局订阅
            for subscription in self.global_subscriptions:
                if subscription.event_filter.match(event):
                    matching_subscriptions.append(subscription)
        
        return matching_subscriptions
    
    def _handle_subscription(self, subscription: EventSubscription, event: BaseEvent) -> None:
        """处理订阅"""
        try:
            subscription.handler(event)
            subscription.processed_count += 1
            subscription.last_processed_time = datetime.now()
            
        except Exception as e:
            subscription.error_count += 1
            self.logger.error(f"Subscription handler failed: {e}")
            raise
    
    async def _handle_subscription_async(self, subscription: EventSubscription, event: BaseEvent) -> None:
        """异步处理订阅"""
        try:
            if asyncio.iscoroutinefunction(subscription.handler):
                await subscription.handler(event)
            else:
                subscription.handler(event)
                
            subscription.processed_count += 1
            subscription.last_processed_time = datetime.now()
            
        except Exception as e:
            subscription.error_count += 1
            self.logger.error(f"Async subscription handler failed: {e}")
            raise
    
    def _process_composite_event(self, composite_event: CompositeEvent) -> None:
        """处理复合事件"""
        if composite_event.is_complete():
            # 复合事件完成，触发完成回调
            self.logger.info(f"Composite event {composite_event.event_id} completed")
            composite_event.mark_completed()
    
    @contextmanager
    def batch_publish(self):
        """批量发布上下文管理器"""
        events = []
        
        def batch_publish_func(event: BaseEvent) -> str:
            events.append(event)
            return event.event_id
        
        original_publish = self.publish
        self.publish = batch_publish_func
        
        try:
            yield
        finally:
            self.publish = original_publish
            
            # 批量发布事件
            for event in events:
                self.publish(event)
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取指标统计"""
        if not self.metrics:
            return {}
        
        base_metrics = self.metrics.get_summary()
        
        # 添加额外指标
        base_metrics.update({
            'active_events_count': len(self.active_events),
            'failed_events_count': len(self.failed_events),
            'composite_events_count': len(self.composite_events),
            'queue_size': self.event_queue.qsize(),
            'retry_queue_size': self.retry_queue.qsize(),
            'subscriptions_count': sum(len(subs) for subs in self.subscriptions.values()) + len(self.global_subscriptions)
        })
        
        return base_metrics
    
    def get_subscription_stats(self) -> Dict[str, Any]:
        """获取订阅统计"""
        stats = {}
        
        with self._subscription_lock:
            for event_type, subscriptions in self.subscriptions.items():
                stats[event_type] = [
                    {
                        'subscription_id': sub.subscription_id,
                        'processed_count': sub.processed_count,
                        'error_count': sub.error_count,
                        'last_processed_time': sub.last_processed_time.isoformat() if sub.last_processed_time else None,
                        'avg_processing_time': sub.avg_processing_time
                    }
                    for sub in subscriptions
                ]
        
        return stats
    
    def cleanup_old_events(self, max_age_hours: int = 24) -> int:
        """清理旧事件"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        with self._lock:
            # 清理活跃事件
            expired_events = [
                event_id for event_id, event in self.active_events.items()
                if event.timestamp < cutoff_time and event.status in [EventStatus.COMPLETED, EventStatus.FAILED]
            ]
            
            for event_id in expired_events:
                del self.active_events[event_id]
                cleaned_count += 1
            
            # 清理失败事件
            expired_failed = [
                event_id for event_id, event in self.failed_events.items()
                if event.timestamp < cutoff_time
            ]
            
            for event_id in expired_failed:
                del self.failed_events[event_id]
                cleaned_count += 1
        
        self.logger.info(f"Cleaned up {cleaned_count} old events")
        return cleaned_count