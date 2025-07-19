# -*- coding: utf-8 -*-
"""
StrategyEngine - 策略引擎模块
"""

import logging
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SignalType(Enum):
    """信号类型"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class EventType(Enum):
    """事件类型"""

    BAR_DATA = "BAR_DATA"
    TICK_DATA = "TICK_DATA"


class Event:
    """事件类"""

    def __init__(self, event_type: EventType, data: Dict[str, Any]):
        self.event_type = event_type
        self.data = data
        self.timestamp = datetime.now()


class BaseStrategy(ABC):
    """策略基类"""

    def __init__(self, name: str, symbols: List[str], params: Dict[str, Any] = None):
        self.name = name
        self.symbols = symbols
        self.params = params or {}
        self.state = {}
        self.active = True

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def on_bar(self, bar_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def on_tick(self, tick_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def finalize(self):
        pass


class MAStrategy(BaseStrategy):
    """移动平均策略"""

    def initialize(self):
        self.price_history = {}
        for symbol in self.symbols:
            self.price_history[symbol] = []

    def on_bar(self, bar_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        symbol = bar_data.get("symbol")
        if symbol not in self.symbols:
            return []

        close_price = bar_data.get("close", 0)
        self.price_history[symbol].append(close_price)

        # 简单的买入信号逻辑
        if len(self.price_history[symbol]) > 5:
            recent_avg = sum(self.price_history[symbol][-5:]) / 5
            if close_price > recent_avg * 1.02:  # 价格上涨2%
                return [
                    {
                        "timestamp": bar_data.get("datetime", datetime.now()),
                        "symbol": symbol,
                        "signal_type": SignalType.BUY.value,
                        "price": close_price,
                        "quantity": 1000,
                        "strategy_name": self.name,
                    }
                ]
        return []

    def on_tick(self, tick_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []

    def finalize(self):
        pass


class StrategyEngine:
    """策略引擎"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # 配置验证
        self._validate_config()

        # 基本配置
        self.max_strategies = self.config.get("max_strategies", 10)
        self.event_queue_size = self.config.get("event_queue_size", 1000)
        self.enable_logging = self.config.get("enable_logging", True)
        self.thread_pool_size = self.config.get("thread_pool_size", 4)

        # 策略管理
        self.strategies = {}
        self.strategy_names = set()
        self.strategy_status = {}  # 策略状态跟踪

        # 事件处理
        self.event_queue = deque(maxlen=self.event_queue_size)
        self.processing_events = False

        # 线程安全
        self._lock = threading.Lock()

        self.logger = logging.getLogger(__name__)

    def _validate_config(self):
        """验证配置参数"""
        if "max_strategies" in self.config:
            if self.config["max_strategies"] <= 0:
                raise ValueError("max_strategies must be positive")

        if "event_queue_size" in self.config:
            if self.config["event_queue_size"] <= 0:
                raise ValueError("event_queue_size must be positive")

        if "max_positions" in self.config:
            if self.config["max_positions"] <= 0:
                raise ValueError("max_positions must be positive")

        if "max_orders_per_second" in self.config:
            if self.config["max_orders_per_second"] <= 0:
                raise ValueError("max_orders_per_second must be positive")

        if "thread_pool_size" in self.config:
            if self.config["thread_pool_size"] <= 0:
                raise ValueError("thread_pool_size must be positive")

    def add_strategy(self, strategy: BaseStrategy) -> str:
        """添加策略"""
        with self._lock:
            # 检查策略数量限制
            if len(self.strategies) >= self.max_strategies:
                raise RuntimeError("Maximum number of strategies reached")

            # 检查策略名称重复
            if strategy.name in self.strategy_names:
                raise ValueError("Strategy name already exists")

            # 添加策略
            strategy_id = str(uuid.uuid4())
            self.strategies[strategy_id] = strategy
            self.strategy_names.add(strategy.name)
            self.strategy_status[strategy.name] = "ACTIVE"  # 设置初始状态

            # 初始化策略
            try:
                strategy.initialize()
            except Exception as e:
                # 如果初始化失败，移除策略
                del self.strategies[strategy_id]
                self.strategy_names.discard(strategy.name)
                self.strategy_status.pop(strategy.name, None)
                raise RuntimeError(f"Strategy initialization failed: {str(e)}")

            return strategy_id

    def remove_strategy(self, strategy_id: str) -> bool:
        """移除策略"""
        with self._lock:
            if strategy_id not in self.strategies:
                return False

            strategy = self.strategies[strategy_id]

            # 调用策略清理方法
            try:
                strategy.finalize()
            except Exception as e:
                self.logger.warning(f"Strategy finalization failed: {str(e)}")

            # 移除策略
            self.strategy_names.discard(strategy.name)
            del self.strategies[strategy_id]

            return True

    def get_strategy_by_name(self, name: str) -> Optional[BaseStrategy]:
        """根据名称获取策略"""
        with self._lock:
            for strategy in self.strategies.values():
                if strategy.name == name:
                    return strategy
            return None

    def validate_bar_data(self, bar_data: Dict[str, Any]) -> bool:
        """验证Bar数据格式"""
        required_fields = ["symbol"]
        for field in required_fields:
            if field not in bar_data:
                return False
        return True

    def process_bar_data(self, bar_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理Bar数据"""
        # 验证数据格式
        if not self.validate_bar_data(bar_data):
            raise ValueError("Invalid bar data format")

        all_signals = []
        symbol = bar_data.get("symbol")

        # 处理相关策略
        for strategy in self.strategies.values():
            if not strategy.active:
                continue

            # 检查策略是否关注该股票
            if hasattr(strategy, "symbols") and symbol not in strategy.symbols:
                continue

            try:
                signals = strategy.on_bar(bar_data)
                if signals:
                    # 验证信号
                    valid_signals = []
                    for s in signals:
                        is_valid = self.validate_signal(s)
                        if is_valid:
                            valid_signals.append(s)
                        else:
                            self.logger.warning(f"信号验证失败: {s}")
                    all_signals.extend(valid_signals)
            except Exception as e:
                self.logger.error(f"策略 {strategy.name} 处理异常: {str(e)}")
                # 标记策略为错误状态
                self.strategy_status[strategy.name] = "ERROR"

        return all_signals

    def process_tick_data(self, tick_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理Tick数据"""
        all_signals = []
        symbol = tick_data.get("symbol")

        for strategy in self.strategies.values():
            if not strategy.active:
                continue

            # 检查策略是否关注该股票
            if hasattr(strategy, "symbols") and symbol not in strategy.symbols:
                continue

            try:
                signals = strategy.on_tick(tick_data)
                if signals:
                    # 验证信号
                    valid_signals = [s for s in signals if self.validate_signal(s)]
                    all_signals.extend(valid_signals)
            except Exception as e:
                self.logger.error(f"策略 {strategy.name} 处理异常: {str(e)}")

        return all_signals

    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """验证信号"""
        required_fields = ["timestamp", "symbol", "signal_type", "price", "quantity"]

        # 检查必要字段
        for field in required_fields:
            if field not in signal:
                return False

        # 检查股票代码不能为空
        if not signal["symbol"] or signal["symbol"].strip() == "":
            return False

        # 检查价格有效性
        if signal["price"] <= 0:
            return False

        # 检查数量有效性
        if signal["quantity"] <= 0:
            return False

        # 检查信号类型
        if signal["signal_type"] not in [
            SignalType.BUY.value,
            SignalType.SELL.value,
            SignalType.HOLD.value,
        ]:
            return False

        return True

    def add_event(self, event: Dict[str, Any]):
        """添加事件到队列"""
        if len(self.event_queue) >= self.event_queue_size:
            raise RuntimeError("Event queue is full")

        event_obj = Event(EventType(event["type"]), event["data"])
        self.event_queue.append(event_obj)

    def process_events(self):
        """处理事件队列"""
        if self.processing_events:
            return

        self.processing_events = True
        try:
            while self.event_queue:
                event = self.event_queue.popleft()

                if event.event_type == EventType.BAR_DATA:
                    self.process_bar_data(event.data)
                elif event.event_type == EventType.TICK_DATA:
                    self.process_tick_data(event.data)
        finally:
            self.processing_events = False

    def get_strategy_status(self, strategy_name: str) -> str:
        """获取策略状态"""
        # 首先检查strategy_status字典
        if strategy_name in self.strategy_status:
            return self.strategy_status[strategy_name]

        # 如果不在字典中，检查策略对象的active状态
        for strategy in self.strategies.values():
            if strategy.name == strategy_name:
                return "ACTIVE" if strategy.active else "INACTIVE"
        return "NOT_FOUND"

    def cleanup_old_signals(self):
        """清理旧信号"""
        pass
