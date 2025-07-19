# -*- coding: utf-8 -*-
"""
策略基类 - 定义策略的基本接口和生命周期
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from enum import Enum

from ..events.event_types import SignalEvent, OrderEvent
from ..models.signals import Signal
from ..models.orders import Order


class StrategyState(Enum):
    """策略状态枚举"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class BaseStrategy(ABC):
    """
    策略基类 - 定义策略的基本接口
    """
    
    def __init__(self, 
                 name: str,
                 symbols: List[str],
                 params: Dict[str, Any] = None,
                 **kwargs):
        """
        初始化策略
        
        Args:
            name: 策略名称
            symbols: 交易标的列表
            params: 策略参数
            **kwargs: 其他参数
        """
        self.name = name
        self.symbols = symbols
        self.params = params or {}
        
        # 策略状态
        self.state = StrategyState.INITIALIZED
        self.active = True
        
        # 时间戳
        self.created_time = datetime.now()
        self.started_time = None
        self.stopped_time = None
        
        # 数据存储
        self.data_cache = {}
        self.positions = {}
        self.orders = {}
        
        # 性能指标
        self.performance_metrics = {}
        
        # 事件处理器
        self.event_handlers = {}
        
        # 日志记录
        self.trade_log = []
        self.signal_log = []
        
    @abstractmethod
    def initialize(self, context: Any = None) -> None:
        """
        策略初始化 - 在策略开始运行前调用
        
        Args:
            context: 上下文对象，包含数据管理器、事件总线等
        """
        pass
    
    @abstractmethod
    def on_bar(self, bar_data: pd.DataFrame) -> None:
        """
        Bar数据事件处理
        
        Args:
            bar_data: Bar数据，包含OHLCV信息
        """
        pass
    
    @abstractmethod
    def on_tick(self, tick_data: pd.DataFrame) -> None:
        """
        Tick数据事件处理
        
        Args:
            tick_data: Tick数据，包含价格、成交量等信息
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        生成交易信号
        
        Args:
            data: 市场数据
            
        Returns:
            List[Signal]: 交易信号列表
        """
        pass
    
    def finalize(self) -> None:
        """
        策略清理 - 在策略停止运行后调用
        """
        self.state = StrategyState.STOPPED
        self.stopped_time = datetime.now()
        
    def start(self) -> None:
        """启动策略"""
        if self.state == StrategyState.INITIALIZED:
            self.state = StrategyState.RUNNING
            self.started_time = datetime.now()
            self.active = True
            
    def stop(self) -> None:
        """停止策略"""
        self.state = StrategyState.STOPPED
        self.stopped_time = datetime.now()
        self.active = False
        
    def pause(self) -> None:
        """暂停策略"""
        if self.state == StrategyState.RUNNING:
            self.state = StrategyState.PAUSED
            self.active = False
            
    def resume(self) -> None:
        """恢复策略"""
        if self.state == StrategyState.PAUSED:
            self.state = StrategyState.RUNNING
            self.active = True
            
    def is_active(self) -> bool:
        """检查策略是否活跃"""
        return self.active and self.state == StrategyState.RUNNING
    
    def update_params(self, params: Dict[str, Any]) -> None:
        """
        更新策略参数
        
        Args:
            params: 新的参数字典
        """
        self.params.update(params)
        
    def get_param(self, key: str, default: Any = None) -> Any:
        """
        获取策略参数
        
        Args:
            key: 参数键
            default: 默认值
            
        Returns:
            参数值
        """
        return self.params.get(key, default)
    
    def validate_params(self) -> bool:
        """
        验证策略参数
        
        Returns:
            bool: 参数是否有效
        """
        return True
    
    def cache_data(self, key: str, data: Any) -> None:
        """
        缓存数据
        
        Args:
            key: 缓存键
            data: 数据
        """
        self.data_cache[key] = data
        
    def get_cached_data(self, key: str, default: Any = None) -> Any:
        """
        获取缓存数据
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            缓存的数据
        """
        return self.data_cache.get(key, default)
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.data_cache.clear()
        
    def log_signal(self, signal: Signal) -> None:
        """
        记录信号
        
        Args:
            signal: 交易信号
        """
        self.signal_log.append({
            'timestamp': datetime.now(),
            'signal': signal,
            'strategy': self.name
        })
        
    def log_trade(self, trade_info: Dict[str, Any]) -> None:
        """
        记录交易
        
        Args:
            trade_info: 交易信息
        """
        self.trade_log.append({
            'timestamp': datetime.now(),
            'trade_info': trade_info,
            'strategy': self.name
        })
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能摘要
        
        Returns:
            Dict[str, Any]: 性能指标字典
        """
        return {
            'strategy_name': self.name,
            'symbols': self.symbols,
            'state': self.state.value,
            'created_time': self.created_time,
            'started_time': self.started_time,
            'stopped_time': self.stopped_time,
            'total_signals': len(self.signal_log),
            'total_trades': len(self.trade_log),
            'performance_metrics': self.performance_metrics
        }
    
    def __str__(self) -> str:
        return f"Strategy({self.name}, {self.symbols}, {self.state.value})"
    
    def __repr__(self) -> str:
        return self.__str__()