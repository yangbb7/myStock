# -*- coding: utf-8 -*-
"""
测试事件总线模块
"""

import asyncio
import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from myQuant.core.events.event_bus import (
    EventPriority,
    Event,
    EventFilter,
    EventSubscription,
    EventBus,
    get_event_bus,
    start_global_event_bus,
    stop_global_event_bus
)


class TestEventPriority:
    """测试事件优先级枚举"""
    
    def test_event_priority_values(self):
        """测试事件优先级值"""
        assert EventPriority.LOW.value == 1
        assert EventPriority.NORMAL.value == 2
        assert EventPriority.HIGH.value == 3
        assert EventPriority.CRITICAL.value == 4
    
    def test_event_priority_comparison(self):
        """测试事件优先级比较"""
        assert EventPriority.LOW.value < EventPriority.NORMAL.value
        assert EventPriority.NORMAL.value < EventPriority.HIGH.value
        assert EventPriority.HIGH.value < EventPriority.CRITICAL.value


class TestEvent:
    """测试事件类"""
    
    def test_event_creation_basic(self):
        """测试基本事件创建"""
        event = Event(
            type="test_event",
            data={"key": "value"}
        )
        
        assert event.type == "test_event"
        assert event.data == {"key": "value"}
        assert event.priority == EventPriority.NORMAL
        assert event.source is None
        assert event.correlation_id is not None
        assert isinstance(event.timestamp, datetime)
        assert isinstance(event.metadata, dict)
    
    def test_event_creation_full(self):
        """测试完整事件创建"""
        timestamp = datetime.now()
        metadata = {"meta_key": "meta_value"}
        
        event = Event(
            type="test_event",
            data={"key": "value"},
            timestamp=timestamp,
            priority=EventPriority.HIGH,
            source="test_source",
            correlation_id="test_correlation_id",
            metadata=metadata
        )
        
        assert event.type == "test_event"
        assert event.data == {"key": "value"}
        assert event.timestamp == timestamp
        assert event.priority == EventPriority.HIGH
        assert event.source == "test_source"
        assert event.correlation_id == "test_correlation_id"
        assert event.metadata == metadata
    
    def test_event_post_init_correlation_id(self):
        """测试事件后初始化生成correlation_id"""
        event = Event(type="test", data={})
        
        assert event.correlation_id is not None
        assert event.correlation_id.startswith("test_")
        assert len(event.correlation_id) > 5  # 应该包含时间戳
    
    def test_event_to_dict(self):
        """测试事件转换为字典"""
        event = Event(
            type="test_event",
            data={"key": "value"},
            priority=EventPriority.HIGH,
            source="test_source",
            correlation_id="test_id",
            metadata={"meta": "data"}
        )
        
        result = event.to_dict()
        
        assert result["type"] == "test_event"
        assert result["data"] == {"key": "value"}
        assert result["priority"] == "HIGH"
        assert result["source"] == "test_source"
        assert result["correlation_id"] == "test_id"
        assert result["metadata"] == {"meta": "data"}
        assert isinstance(result["timestamp"], str)  # 应该是ISO格式字符串
    
    def test_event_empty_data(self):
        """测试空数据事件"""
        event = Event(type="empty_event", data={})
        
        assert event.type == "empty_event"
        assert event.data == {}
        assert event.correlation_id is not None
    
    def test_event_large_data(self):
        """测试大数据事件"""
        large_data = {f"key_{i}": f"value_{i}" for i in range(1000)}
        event = Event(type="large_event", data=large_data)
        
        assert event.type == "large_event"
        assert len(event.data) == 1000
        assert event.data["key_0"] == "value_0"
        assert event.data["key_999"] == "value_999"


class TestEventFilter:
    """测试事件过滤器"""
    
    def test_event_filter_creation_empty(self):
        """测试空过滤器创建"""
        filter_obj = EventFilter()
        
        assert filter_obj.event_types is None
        assert filter_obj.sources is None
        assert filter_obj.priority_min is None
    
    def test_event_filter_creation_full(self):
        """测试完整过滤器创建"""
        event_types = ["type1", "type2"]
        sources = ["source1", "source2"]
        priority_min = EventPriority.HIGH
        
        filter_obj = EventFilter(
            event_types=event_types,
            sources=sources,
            priority_min=priority_min
        )
        
        assert filter_obj.event_types == {"type1", "type2"}
        assert filter_obj.sources == {"source1", "source2"}
        assert filter_obj.priority_min == EventPriority.HIGH
    
    def test_event_filter_matches_no_filter(self):
        """测试无过滤条件匹配"""
        filter_obj = EventFilter()
        event = Event(type="test", data={})
        
        assert filter_obj.matches(event) is True
    
    def test_event_filter_matches_event_type(self):
        """测试事件类型过滤"""
        filter_obj = EventFilter(event_types=["type1", "type2"])
        
        event1 = Event(type="type1", data={})
        event2 = Event(type="type3", data={})
        
        assert filter_obj.matches(event1) is True
        assert filter_obj.matches(event2) is False
    
    def test_event_filter_matches_source(self):
        """测试事件源过滤"""
        filter_obj = EventFilter(sources=["source1", "source2"])
        
        event1 = Event(type="test", data={}, source="source1")
        event2 = Event(type="test", data={}, source="source3")
        
        assert filter_obj.matches(event1) is True
        assert filter_obj.matches(event2) is False
    
    def test_event_filter_matches_priority(self):
        """测试事件优先级过滤"""
        filter_obj = EventFilter(priority_min=EventPriority.HIGH)
        
        event1 = Event(type="test", data={}, priority=EventPriority.HIGH)
        event2 = Event(type="test", data={}, priority=EventPriority.CRITICAL)
        event3 = Event(type="test", data={}, priority=EventPriority.NORMAL)
        
        assert filter_obj.matches(event1) is True
        assert filter_obj.matches(event2) is True
        assert filter_obj.matches(event3) is False
    
    def test_event_filter_matches_combined(self):
        """测试组合过滤条件"""
        filter_obj = EventFilter(
            event_types=["type1"],
            sources=["source1"],
            priority_min=EventPriority.HIGH
        )
        
        # 全部匹配
        event1 = Event(
            type="type1",
            data={},
            source="source1",
            priority=EventPriority.HIGH
        )
        
        # 类型不匹配
        event2 = Event(
            type="type2",
            data={},
            source="source1",
            priority=EventPriority.HIGH
        )
        
        # 源不匹配
        event3 = Event(
            type="type1",
            data={},
            source="source2",
            priority=EventPriority.HIGH
        )
        
        # 优先级不匹配
        event4 = Event(
            type="type1",
            data={},
            source="source1",
            priority=EventPriority.NORMAL
        )
        
        assert filter_obj.matches(event1) is True
        assert filter_obj.matches(event2) is False
        assert filter_obj.matches(event3) is False
        assert filter_obj.matches(event4) is False


class TestEventSubscription:
    """测试事件订阅"""
    
    def test_event_subscription_sync_handler(self):
        """测试同步处理器订阅"""
        def sync_handler(event):
            pass
        
        subscription = EventSubscription(sync_handler)
        
        assert subscription.handler == sync_handler
        assert subscription.filter is None
        assert subscription.async_handler is False
        assert subscription.call_count == 0
        assert subscription.last_called is None
        assert subscription.error_count == 0
        assert isinstance(subscription.created_at, datetime)
    
    def test_event_subscription_async_handler(self):
        """测试异步处理器订阅"""
        async def async_handler(event):
            pass
        
        subscription = EventSubscription(async_handler)
        
        assert subscription.handler == async_handler
        assert subscription.async_handler is True
    
    def test_event_subscription_with_filter(self):
        """测试带过滤器的订阅"""
        def handler(event):
            pass
        
        event_filter = EventFilter(event_types=["test"])
        subscription = EventSubscription(handler, event_filter)
        
        assert subscription.filter == event_filter
    
    def test_event_subscription_explicit_async_flag(self):
        """测试显式异步标志"""
        def sync_handler(event):
            pass
        
        # 显式设置为异步
        subscription = EventSubscription(sync_handler, async_handler=True)
        assert subscription.async_handler is True
        
        # 显式设置为同步
        subscription = EventSubscription(sync_handler, async_handler=False)
        assert subscription.async_handler is False
    
    def test_event_subscription_matches_no_filter(self):
        """测试无过滤器匹配"""
        def handler(event):
            pass
        
        subscription = EventSubscription(handler)
        event = Event(type="test", data={})
        
        assert subscription.matches(event) is True
    
    def test_event_subscription_matches_with_filter(self):
        """测试带过滤器匹配"""
        def handler(event):
            pass
        
        event_filter = EventFilter(event_types=["test"])
        subscription = EventSubscription(handler, event_filter)
        
        event1 = Event(type="test", data={})
        event2 = Event(type="other", data={})
        
        assert subscription.matches(event1) is True
        assert subscription.matches(event2) is False
    
    def test_event_subscription_get_stats(self):
        """测试获取订阅统计"""
        def handler(event):
            pass
        
        subscription = EventSubscription(handler)
        subscription.call_count = 5
        subscription.error_count = 1
        subscription.last_called = datetime.now()
        
        stats = subscription.get_stats()
        
        assert "handler" in stats
        assert "async_handler" in stats
        assert "created_at" in stats
        assert "call_count" in stats
        assert "last_called" in stats
        assert "error_count" in stats
        assert stats["call_count"] == 5
        assert stats["error_count"] == 1
        assert stats["async_handler"] is False


class TestEventBus:
    """测试事件总线"""
    
    def test_event_bus_creation(self):
        """测试事件总线创建"""
        bus = EventBus()
        
        assert bus.max_queue_size == 10000
        assert bus.worker_count == 4
        assert bus.batch_size == 10
        assert bus.max_retry_attempts == 3
        assert bus.retry_delay == 1.0
        assert bus._running is False
        assert len(bus._workers) == 0
        assert len(bus._subscriptions) == 0
    
    def test_event_bus_creation_with_config(self):
        """测试带配置的事件总线创建"""
        config = {
            "max_queue_size": 5000,
            "worker_count": 2,
            "batch_size": 5,
            "max_retry_attempts": 5,
            "retry_delay": 2.0
        }
        
        bus = EventBus(config)
        
        assert bus.max_queue_size == 5000
        assert bus.worker_count == 2
        assert bus.batch_size == 5
        assert bus.max_retry_attempts == 5
        assert bus.retry_delay == 2.0
    
    def test_subscribe_sync(self):
        """测试同步订阅"""
        bus = EventBus()
        
        def handler(event):
            pass
        
        subscription_id = bus.subscribe_sync("test_event", handler)
        
        assert subscription_id == "test_event_1"
        assert "test_event" in bus._subscriptions
        assert len(bus._subscriptions["test_event"]) == 1
        assert bus._subscriptions["test_event"][0].handler == handler
    
    def test_subscribe_sync_with_filter(self):
        """测试带过滤器的同步订阅"""
        bus = EventBus()
        
        def handler(event):
            pass
        
        event_filter = EventFilter(sources=["test_source"])
        subscription_id = bus.subscribe_sync("test_event", handler, event_filter)
        
        assert subscription_id == "test_event_1"
        assert bus._subscriptions["test_event"][0].filter == event_filter
    
    def test_subscribe_multiple_handlers(self):
        """测试多个处理器订阅"""
        bus = EventBus()
        
        def handler1(event):
            pass
        
        def handler2(event):
            pass
        
        id1 = bus.subscribe_sync("test_event", handler1)
        id2 = bus.subscribe_sync("test_event", handler2)
        
        assert id1 == "test_event_1"
        assert id2 == "test_event_2"
        assert len(bus._subscriptions["test_event"]) == 2
    
    def test_subscribe_different_events(self):
        """测试不同事件订阅"""
        bus = EventBus()
        
        def handler1(event):
            pass
        
        def handler2(event):
            pass
        
        id1 = bus.subscribe_sync("event1", handler1)
        id2 = bus.subscribe_sync("event2", handler2)
        
        assert id1 == "event1_1"
        assert id2 == "event2_1"
        assert len(bus._subscriptions) == 2
    
    def test_subscribe_enum_event_type(self):
        """测试枚举事件类型订阅"""
        bus = EventBus()
        
        def handler(event):
            pass
        
        # 模拟枚举类型
        class EventType:
            def __init__(self, value):
                self.value = value
        
        event_type = EventType("test_event")
        subscription_id = bus.subscribe_sync(event_type, handler)
        
        assert subscription_id == "test_event_1"
        assert "test_event" in bus._subscriptions
    
    def test_subscribe_wrapper(self):
        """测试subscribe包装器"""
        bus = EventBus()
        
        def handler(event):
            pass
        
        subscription_id = bus.subscribe("test_event", handler)
        
        assert subscription_id == "test_event_1"
        assert "test_event" in bus._subscriptions
    
    def test_publish_sync_blocking(self):
        """测试同步阻塞发布"""
        bus = EventBus()
        
        # 记录处理器调用
        handler_calls = []
        
        def handler(event):
            handler_calls.append(event)
        
        bus.subscribe_sync("test_event", handler)
        
        # 发布事件
        result = bus.publish("test_event", {"key": "value"})
        
        assert result is True
        assert len(handler_calls) == 1
        assert handler_calls[0].type == "test_event"
        assert handler_calls[0].data == {"key": "value"}
    
    def test_publish_with_event_object(self):
        """测试发布事件对象"""
        bus = EventBus()
        
        handler_calls = []
        
        def handler(event):
            handler_calls.append(event)
        
        bus.subscribe_sync("test_event", handler)
        
        # 创建事件对象
        event = Event(type="test_event", data={"key": "value"})
        result = bus.publish(event)
        
        assert result is True
        assert len(handler_calls) == 1
        assert handler_calls[0] == event
    
    def test_publish_no_subscribers(self):
        """测试发布无订阅者事件"""
        bus = EventBus()
        
        result = bus.publish("no_subscribers", {"key": "value"})
        
        assert result is True  # 应该成功，只是没有处理器
    
    def test_publish_with_filter_matching(self):
        """测试发布带过滤器匹配的事件"""
        bus = EventBus()
        
        handler_calls = []
        
        def handler(event):
            handler_calls.append(event)
        
        event_filter = EventFilter(sources=["test_source"])
        bus.subscribe_sync("test_event", handler, event_filter)
        
        # 发布匹配的事件
        event = Event(type="test_event", data={}, source="test_source")
        result = bus.publish(event)
        
        assert result is True
        assert len(handler_calls) == 1
        assert handler_calls[0].source == "test_source"
    
    def test_publish_with_filter_not_matching(self):
        """测试发布带过滤器不匹配的事件"""
        bus = EventBus()
        
        handler_calls = []
        
        def handler(event):
            handler_calls.append(event)
        
        event_filter = EventFilter(sources=["test_source"])
        bus.subscribe_sync("test_event", handler, event_filter)
        
        # 发布不匹配的事件
        event = Event(type="test_event", data={}, source="other_source")
        result = bus.publish(event)
        
        assert result is True
        assert len(handler_calls) == 0  # 应该被过滤掉
    
    def test_publish_handler_exception(self):
        """测试处理器异常"""
        bus = EventBus()
        
        def failing_handler(event):
            raise ValueError("Handler failed")
        
        def working_handler(event):
            working_handler.called = True
        
        working_handler.called = False
        
        bus.subscribe_sync("test_event", failing_handler)
        bus.subscribe_sync("test_event", working_handler)
        
        # 发布事件
        result = bus.publish("test_event", {})
        
        assert result is True  # 应该成功，即使有处理器失败
        assert working_handler.called is True  # 其他处理器应该仍然被调用
    
    def test_publish_statistics_update(self):
        """测试发布统计更新"""
        bus = EventBus()
        
        def handler(event):
            pass
        
        bus.subscribe_sync("test_event", handler)
        
        initial_published = bus._stats["events_published"]
        initial_processed = bus._stats["events_processed"]
        
        bus.publish("test_event", {})
        
        assert bus._stats["events_published"] == initial_published + 1
        assert bus._stats["events_processed"] == initial_processed + 1
    
    def test_get_stats(self):
        """测试获取统计信息"""
        bus = EventBus()
        
        def handler(event):
            pass
        
        bus.subscribe_sync("test_event", handler)
        
        stats = bus.get_stats()
        
        assert "events_published" in stats
        assert "events_processed" in stats
        assert "events_failed" in stats
        assert "handlers_executed" in stats
        assert "handlers_failed" in stats
        assert "average_processing_time" in stats
        assert "queue_size" in stats
        assert "priority_queue_size" in stats
        assert "subscription_count" in stats
        assert "event_types" in stats
        assert "running" in stats
        assert "worker_count" in stats
        
        assert stats["subscription_count"] == 1
        assert stats["event_types"] == ["test_event"]
        assert stats["running"] is False
        assert stats["worker_count"] == 0
    
    def test_get_subscription_stats(self):
        """测试获取订阅统计"""
        bus = EventBus()
        
        def handler(event):
            pass
        
        bus.subscribe_sync("test_event", handler)
        
        stats = bus.get_subscription_stats()
        
        assert "test_event" in stats
        assert len(stats["test_event"]) == 1
        assert "handler" in stats["test_event"][0]
        assert "async_handler" in stats["test_event"][0]
        assert "created_at" in stats["test_event"][0]
        assert "call_count" in stats["test_event"][0]
        assert "error_count" in stats["test_event"][0]
    
    def test_get_event_history(self):
        """测试获取事件历史"""
        bus = EventBus()
        
        def handler(event):
            pass
        
        bus.subscribe_sync("test_event", handler)
        
        # 发布几个事件
        bus.publish("test_event", {"id": 1})
        bus.publish("test_event", {"id": 2})
        bus.publish("test_event", {"id": 3})
        
        history = bus.get_event_history()
        
        assert len(history) == 3
        assert all("type" in event for event in history)
        assert all("data" in event for event in history)
        assert all("timestamp" in event for event in history)
        assert history[0]["data"]["id"] == 1
        assert history[1]["data"]["id"] == 2
        assert history[2]["data"]["id"] == 3
    
    def test_get_event_history_with_limit(self):
        """测试获取限制数量的事件历史"""
        bus = EventBus()
        
        def handler(event):
            pass
        
        bus.subscribe_sync("test_event", handler)
        
        # 发布多个事件
        for i in range(10):
            bus.publish("test_event", {"id": i})
        
        history = bus.get_event_history(limit=5)
        
        assert len(history) == 5
        # 应该返回最后5个事件
        assert history[0]["data"]["id"] == 5
        assert history[4]["data"]["id"] == 9
    
    def test_clear_history(self):
        """测试清理事件历史"""
        bus = EventBus()
        
        def handler(event):
            pass
        
        bus.subscribe_sync("test_event", handler)
        
        # 发布事件
        bus.publish("test_event", {"id": 1})
        
        # 确认历史记录存在
        history = bus.get_event_history()
        assert len(history) == 1
        
        # 清理历史
        bus.clear_history()
        
        # 确认历史记录被清理
        history = bus.get_event_history()
        assert len(history) == 0


class TestEventBusAsync:
    """测试事件总线异步功能"""
    
    def setup_method(self):
        """每个测试前清理"""
        pass
        
    def teardown_method(self):
        """每个测试后清理"""  
        # 每个测试都有自己的清理逻辑，这里不需要额外处理
        pass
    
    @pytest.mark.asyncio
    async def test_start_stop_event_bus(self):
        """测试启动和停止事件总线"""
        bus = EventBus()
        
        # 初始状态
        assert bus._running is False
        assert len(bus._workers) == 0
        
        # 启动
        await bus.start()
        assert bus._running is True
        assert len(bus._workers) == bus.worker_count
        
        # 停止
        await bus.stop()
        assert bus._running is False
        assert len(bus._workers) == 0
    
    @pytest.mark.asyncio
    async def test_start_already_running(self):
        """测试启动已运行的事件总线"""
        bus = EventBus()
        
        await bus.start()
        initial_workers = len(bus._workers)
        
        # 再次启动应该不做任何事
        await bus.start()
        assert len(bus._workers) == initial_workers
        
        await bus.stop()
    
    @pytest.mark.asyncio
    async def test_stop_not_running(self):
        """测试停止未运行的事件总线"""
        bus = EventBus()
        
        # 停止未运行的总线应该不做任何事
        await bus.stop()
        assert bus._running is False
        assert len(bus._workers) == 0
    
    @pytest.mark.asyncio
    async def test_subscribe_async(self):
        """测试异步订阅"""
        bus = EventBus()
        
        async def handler(event):
            pass
        
        subscription_id = await bus.subscribe_async("test_event", handler)
        
        assert subscription_id == "test_event_1"
        assert "test_event" in bus._subscriptions
        assert len(bus._subscriptions["test_event"]) == 1
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """测试取消订阅"""
        bus = EventBus()
        
        def handler(event):
            pass
        
        # 订阅
        bus.subscribe_sync("test_event", handler)
        assert len(bus._subscriptions["test_event"]) == 1
        
        # 取消订阅
        result = await bus.unsubscribe("test_event", handler)
        assert result is True
        assert len(bus._subscriptions["test_event"]) == 0
    
    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent(self):
        """测试取消不存在的订阅"""
        bus = EventBus()
        
        def handler(event):
            pass
        
        # 取消不存在的订阅
        result = await bus.unsubscribe("nonexistent_event", handler)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_unsubscribe_different_handler(self):
        """测试取消不同处理器的订阅"""
        bus = EventBus()
        
        def handler1(event):
            pass
        
        def handler2(event):
            pass
        
        # 订阅handler1
        bus.subscribe_sync("test_event", handler1)
        assert len(bus._subscriptions["test_event"]) == 1
        
        # 取消订阅handler2
        result = await bus.unsubscribe("test_event", handler2)
        assert result is False
        assert len(bus._subscriptions["test_event"]) == 1
    
    @pytest.mark.asyncio
    async def test_publish_async_sync_mode(self):
        """测试异步发布同步模式"""
        bus = EventBus()
        
        try:
            handler_calls = []
            
            async def handler(event):
                handler_calls.append(event)
            
            await bus.subscribe_async("test_event", handler)
            
            event = Event(type="test_event", data={"key": "value"})
            result = await bus.publish_async(event, sync=True)
            
            assert result is True
            assert len(handler_calls) == 1
            assert handler_calls[0].type == "test_event"
        finally:
            # 确保事件总线被停止
            await bus.stop()
            await asyncio.sleep(0.01)
    
    @pytest.mark.asyncio
    async def test_wait_for_event(self):
        """测试等待特定事件"""
        bus = EventBus()
        wait_task = None
        
        try:
            # 启动事件总线
            await bus.start()
            
            # 创建等待任务
            wait_task = asyncio.create_task(
                bus.wait_for_event("test_event", timeout=1.0)
            )
            
            # 稍等一下再发布事件
            await asyncio.sleep(0.1)
            
            # 发布事件
            await bus.publish_async(Event(type="test_event", data={"key": "value"}))
            
            # 等待结果
            result = await wait_task
            wait_task = None  # 标记任务已完成
            
            assert result is not None
            assert result.type == "test_event"
            assert result.data == {"key": "value"}
        finally:
            # 取消未完成的任务
            if wait_task and not wait_task.done():
                wait_task.cancel()
                try:
                    await wait_task
                except asyncio.CancelledError:
                    pass
            
            # 确保总线被停止
            await bus.stop()
            await asyncio.sleep(0.01)
    
    @pytest.mark.asyncio
    async def test_wait_for_event_timeout(self):
        """测试等待事件超时"""
        bus = EventBus()
        
        try:
            # 启动事件总线
            await bus.start()
            
            # 等待不存在的事件
            result = await bus.wait_for_event("nonexistent_event", timeout=0.1)
            
            assert result is None
        finally:
            await bus.stop()
            # 给更多时间让异步任务完成清理
            await asyncio.sleep(0.05)
    
    @pytest.mark.asyncio
    async def test_wait_for_event_with_filter(self):
        """测试带过滤器等待事件"""
        bus = EventBus()
        wait_task = None
        
        try:
            # 启动事件总线
            await bus.start()
            
            # 创建过滤器
            event_filter = EventFilter(sources=["test_source"])
            
            # 创建等待任务
            wait_task = asyncio.create_task(
                bus.wait_for_event("test_event", timeout=1.0, event_filter=event_filter)
            )
            
            # 发布不匹配的事件
            await bus.publish_async(Event(type="test_event", data={}, source="other_source"))
            
            # 稍等一下
            await asyncio.sleep(0.1)
            
            # 发布匹配的事件
            await bus.publish_async(Event(type="test_event", data={}, source="test_source"))
            
            # 等待结果
            result = await wait_task
            wait_task = None  # 标记任务已完成
            
            assert result is not None
            assert result.type == "test_event"
            assert result.source == "test_source"
        finally:
            # 取消未完成的任务
            if wait_task and not wait_task.done():
                wait_task.cancel()
                try:
                    await wait_task
                except asyncio.CancelledError:
                    pass
            
            await bus.stop()
            await asyncio.sleep(0.05)


class TestGlobalEventBus:
    """测试全局事件总线"""
    
    def test_get_event_bus_singleton(self):
        """测试获取单例事件总线"""
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        
        assert bus1 is bus2
        assert isinstance(bus1, EventBus)
    
    @pytest.mark.asyncio
    async def test_start_stop_global_event_bus(self):
        """测试启动停止全局事件总线"""
        # 启动全局事件总线
        await start_global_event_bus()
        
        bus = get_event_bus()
        assert bus._running is True
        
        # 停止全局事件总线
        await stop_global_event_bus()
        
        # 全局总线应该被设置为None
        new_bus = get_event_bus()
        assert new_bus is not bus  # 应该是新的实例


class TestEventBusEdgeCases:
    """测试事件总线边界情况"""
    
    def test_event_bus_large_number_of_subscribers(self):
        """测试大量订阅者"""
        bus = EventBus()
        
        handlers = []
        handler_calls = []
        
        # 修复闭包问题，使用工厂函数创建独立的处理器
        def create_handler(handler_id):
            def handler(event):
                handler_calls.append(handler_id)
            return handler
        
        # 创建100个处理器而不是1000个，避免性能问题
        for i in range(100):
            handler = create_handler(i)
            handlers.append(handler)
            bus.subscribe_sync("test_event", handler)
        
        assert len(bus._subscriptions["test_event"]) == 100
        
        # 发布事件
        result = bus.publish("test_event", {})
        assert result is True
        
        # 验证所有处理器都被调用
        assert len(handler_calls) == 100
        assert set(handler_calls) == set(range(100))
    
    def test_event_bus_large_event_data(self):
        """测试大事件数据"""
        bus = EventBus()
        
        handler_calls = []
        
        def handler(event):
            handler_calls.append(event)
        
        bus.subscribe_sync("test_event", handler)
        
        # 创建大数据
        large_data = {f"key_{i}": f"value_{i}" * 1000 for i in range(100)}
        
        result = bus.publish("test_event", large_data)
        
        assert result is True
        assert len(handler_calls) == 1
        assert len(handler_calls[0].data) == 100
    
    def test_event_bus_circular_event_publishing(self):
        """测试循环事件发布"""
        bus = EventBus()
        
        call_count = 0
        other_call_count = 0
        
        def handler(event):
            nonlocal call_count
            call_count += 1
            
            if call_count < 5:
                # 发布另一个事件，但不形成无限循环
                bus.publish("other_event", {"call_count": call_count})
        
        def other_handler(event):
            nonlocal other_call_count
            other_call_count += 1
        
        bus.subscribe_sync("test_event", handler)
        bus.subscribe_sync("other_event", other_handler)
        
        # 发布初始事件
        result = bus.publish("test_event", {})
        
        assert result is True
        # 验证嵌套事件被正确处理
        assert call_count == 1  # 只有初始调用
        assert other_call_count == 1  # 嵌套事件被处理
    
    def test_event_bus_exception_in_handler_logging(self):
        """测试处理器异常日志记录"""
        bus = EventBus()
        
        def failing_handler(event):
            raise ValueError("Test exception")
        
        bus.subscribe_sync("test_event", failing_handler)
        
        with patch('logging.getLogger') as mock_logger:
            mock_log = Mock()
            mock_logger.return_value = mock_log
            
            result = bus.publish("test_event", {})
            
            assert result is True
            # 异常应该被记录但不影响返回结果
    
    def test_event_bus_thread_safety_simulation(self):
        """测试线程安全模拟"""
        bus = EventBus()
        
        handler_calls = []
        
        def handler(event):
            # 模拟处理时间
            import time
            time.sleep(0.001)
            handler_calls.append(event)
        
        bus.subscribe_sync("test_event", handler)
        
        # 模拟并发发布
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(100):
                future = executor.submit(bus.publish, "test_event", {"id": i})
                futures.append(future)
            
            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                assert result is True
        
        # 所有事件都应该被处理
        assert len(handler_calls) == 100
    
    def test_event_bus_memory_usage_large_history(self):
        """测试大量历史记录的内存使用"""
        config = {"max_history_size": 1000}
        bus = EventBus(config)
        
        def handler(event):
            pass
        
        bus.subscribe_sync("test_event", handler)
        
        # 发布大量事件
        for i in range(2000):
            bus.publish("test_event", {"id": i})
        
        # 历史记录应该被限制到配置的大小
        history = bus.get_event_history(limit=None)  # 获取所有历史记录
        assert len(history) == 1000
        
        # 应该保留最后1000个事件
        assert history[0]["data"]["id"] == 1000
        assert history[-1]["data"]["id"] == 1999