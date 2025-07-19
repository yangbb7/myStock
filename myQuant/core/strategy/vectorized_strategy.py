# -*- coding: utf-8 -*-
"""
矢量化策略基类 - 提供高性能的矢量化数据处理能力
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime
import warnings

from .base_strategy import BaseStrategy, StrategyState
from .technical_indicators import TechnicalIndicators
from ..models.signals import Signal, SignalType
from ..events.event_types import SignalEvent

warnings.filterwarnings('ignore')


class VectorizedStrategy(BaseStrategy):
    """
    矢量化策略基类 - 提供高性能的矢量化数据处理和信号生成
    """
    
    def __init__(self, 
                 name: str,
                 symbols: List[str],
                 params: Dict[str, Any] = None,
                 lookback_window: int = 100,
                 **kwargs):
        """
        初始化矢量化策略
        
        Args:
            name: 策略名称
            symbols: 交易标的列表
            params: 策略参数
            lookback_window: 数据回看窗口大小
            **kwargs: 其他参数
        """
        super().__init__(name, symbols, params, **kwargs)
        
        # 矢量化相关参数
        self.lookback_window = lookback_window
        self.min_periods = params.get('min_periods', 20)
        
        # 数据存储
        self.historical_data = {}
        self.indicators_cache = {}
        self.signals_cache = {}
        
        # 性能监控
        self.computation_times = {}
        self.vectorization_stats = {
            'total_computations': 0,
            'vectorized_computations': 0,
            'avg_computation_time': 0.0
        }
        
        # 技术指标配置
        self.indicators_config = self._get_default_indicators_config()
        
    def _get_default_indicators_config(self) -> Dict[str, Any]:
        """
        获取默认技术指标配置
        
        Returns:
            Dict[str, Any]: 指标配置
        """
        return {
            'sma': {'windows': [5, 10, 20, 50]},
            'ema': {'windows': [12, 26]},
            'rsi': {'window': 14},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger': {'window': 20, 'std': 2.0},
            'atr': {'window': 14}
        }
    
    def set_indicators_config(self, config: Dict[str, Any]) -> None:
        """
        设置技术指标配置
        
        Args:
            config: 指标配置字典
        """
        self.indicators_config.update(config)
        
    def update_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        更新历史数据
        
        Args:
            symbol: 交易标的
            data: 新数据
        """
        if symbol not in self.historical_data:
            self.historical_data[symbol] = pd.DataFrame()
            
        # 合并新数据并保持窗口大小
        self.historical_data[symbol] = pd.concat([
            self.historical_data[symbol], 
            data
        ]).tail(self.lookback_window)
        
        # 清除相关缓存
        if symbol in self.indicators_cache:
            del self.indicators_cache[symbol]
        if symbol in self.signals_cache:
            del self.signals_cache[symbol]
    
    def get_data(self, symbol: str, window: Optional[int] = None) -> pd.DataFrame:
        """
        获取历史数据
        
        Args:
            symbol: 交易标的
            window: 数据窗口大小
            
        Returns:
            pd.DataFrame: 历史数据
        """
        if symbol not in self.historical_data:
            return pd.DataFrame()
            
        data = self.historical_data[symbol]
        if window:
            return data.tail(window)
        return data
    
    def compute_indicators(self, symbol: str, force_update: bool = False) -> pd.DataFrame:
        """
        计算技术指标（矢量化）
        
        Args:
            symbol: 交易标的
            force_update: 是否强制更新
            
        Returns:
            pd.DataFrame: 包含技术指标的数据
        """
        start_time = datetime.now()
        
        # 检查缓存
        if not force_update and symbol in self.indicators_cache:
            return self.indicators_cache[symbol]
            
        # 获取数据
        data = self.get_data(symbol)
        if data.empty or len(data) < self.min_periods:
            return pd.DataFrame()
        
        # 矢量化计算所有指标
        try:
            indicators_data = TechnicalIndicators.calculate_multiple_indicators(
                data, self.indicators_config
            )
            
            # 缓存结果
            self.indicators_cache[symbol] = indicators_data
            
            # 更新性能统计
            computation_time = (datetime.now() - start_time).total_seconds()
            self.computation_times[f'indicators_{symbol}'] = computation_time
            self._update_vectorization_stats(computation_time)
            
            return indicators_data
            
        except Exception as e:
            print(f"Error computing indicators for {symbol}: {e}")
            return data
    
    def generate_signals_vectorized(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        矢量化生成交易信号
        
        Args:
            symbol: 交易标的
            data: 市场数据（包含技术指标）
            
        Returns:
            pd.DataFrame: 交易信号DataFrame
        """
        signals_df = pd.DataFrame(index=data.index)
        signals_df['symbol'] = symbol
        signals_df['buy_signal'] = False
        signals_df['sell_signal'] = False
        signals_df['signal_strength'] = 0.0
        signals_df['signal_reason'] = ''
        
        # 这里是默认实现，子类应该重写这个方法
        return signals_df
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        生成交易信号（重写基类方法）
        
        Args:
            data: 市场数据
            
        Returns:
            List[Signal]: 交易信号列表
        """
        signals = []
        
        for symbol in self.symbols:
            symbol_data = data[data.get('symbol', data.index) == symbol] if 'symbol' in data.columns else data
            
            if symbol_data.empty:
                continue
                
            # 更新数据
            self.update_data(symbol, symbol_data)
            
            # 计算指标
            indicators_data = self.compute_indicators(symbol)
            
            if indicators_data.empty:
                continue
                
            # 生成信号
            signals_df = self.generate_signals_vectorized(symbol, indicators_data)
            
            # 转换为Signal对象
            symbol_signals = self._convert_signals_dataframe_to_objects(signals_df)
            signals.extend(symbol_signals)
            
        return signals
    
    def _convert_signals_dataframe_to_objects(self, signals_df: pd.DataFrame) -> List[Signal]:
        """
        将信号DataFrame转换为Signal对象列表
        
        Args:
            signals_df: 信号DataFrame
            
        Returns:
            List[Signal]: Signal对象列表
        """
        signals = []
        
        for idx, row in signals_df.iterrows():
            if row['buy_signal']:
                signal = Signal(
                    symbol=row['symbol'],
                    signal_type=SignalType.BUY,
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    price=row.get('close', 0.0),
                    quantity=self._calculate_position_size(row),
                    confidence=row.get('signal_strength', 0.5),
                    metadata={
                        'reason': row.get('signal_reason', ''),
                        'indicators': row.to_dict()
                    }
                )
                signals.append(signal)
                
            elif row['sell_signal']:
                signal = Signal(
                    symbol=row['symbol'],
                    signal_type=SignalType.SELL,
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    price=row.get('close', 0.0),
                    quantity=self._calculate_position_size(row),
                    confidence=row.get('signal_strength', 0.5),
                    metadata={
                        'reason': row.get('signal_reason', ''),
                        'indicators': row.to_dict()
                    }
                )
                signals.append(signal)
                
        return signals
    
    def _calculate_position_size(self, signal_row: pd.Series) -> int:
        """
        计算持仓大小
        
        Args:
            signal_row: 信号行数据
            
        Returns:
            int: 持仓大小
        """
        # 默认实现，子类可以重写
        base_quantity = self.get_param('base_quantity', 100)
        strength = signal_row.get('signal_strength', 0.5)
        
        return int(base_quantity * strength)
    
    def batch_process_signals(self, data_batch: Dict[str, pd.DataFrame]) -> Dict[str, List[Signal]]:
        """
        批量处理多个标的的信号
        
        Args:
            data_batch: 标的数据批次 {symbol: DataFrame}
            
        Returns:
            Dict[str, List[Signal]]: 标的信号字典
        """
        start_time = datetime.now()
        results = {}
        
        for symbol, data in data_batch.items():
            if symbol in self.symbols:
                try:
                    # 更新数据
                    self.update_data(symbol, data)
                    
                    # 计算指标
                    indicators_data = self.compute_indicators(symbol)
                    
                    if not indicators_data.empty:
                        # 生成信号
                        signals_df = self.generate_signals_vectorized(symbol, indicators_data)
                        signals = self._convert_signals_dataframe_to_objects(signals_df)
                        results[symbol] = signals
                    else:
                        results[symbol] = []
                        
                except Exception as e:
                    print(f"Error processing signals for {symbol}: {e}")
                    results[symbol] = []
        
        # 更新性能统计
        batch_time = (datetime.now() - start_time).total_seconds()
        self.computation_times['batch_process'] = batch_time
        
        return results
    
    def calculate_signal_statistics(self, signals: List[Signal]) -> Dict[str, Any]:
        """
        计算信号统计信息
        
        Args:
            signals: 信号列表
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not signals:
            return {}
            
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        
        return {
            'total_signals': len(signals),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'avg_confidence': np.mean([s.confidence for s in signals]),
            'max_confidence': max([s.confidence for s in signals]) if signals else 0,
            'min_confidence': min([s.confidence for s in signals]) if signals else 0,
            'symbols_with_signals': list(set([s.symbol for s in signals]))
        }
    
    def optimize_parameters_vectorized(self, 
                                     param_ranges: Dict[str, List],
                                     optimization_data: pd.DataFrame,
                                     objective_function: Callable,
                                     max_iterations: int = 1000) -> Dict[str, Any]:
        """
        矢量化参数优化
        
        Args:
            param_ranges: 参数范围字典
            optimization_data: 优化数据
            objective_function: 目标函数
            max_iterations: 最大迭代次数
            
        Returns:
            Dict[str, Any]: 最优参数和结果
        """
        # 生成参数组合
        param_combinations = self._generate_parameter_combinations(param_ranges, max_iterations)
        
        best_params = None
        best_score = float('-inf')
        results = []
        
        for params in param_combinations:
            try:
                # 临时更新参数
                original_params = self.params.copy()
                self.params.update(params)
                
                # 计算指标
                self.indicators_config.update(params.get('indicators', {}))
                
                # 评估性能
                score = objective_function(self, optimization_data)
                results.append({
                    'params': params.copy(),
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                # 恢复原始参数
                self.params = original_params
                
            except Exception as e:
                print(f"Error in parameter optimization: {e}")
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results,
            'total_iterations': len(results)
        }
    
    def _generate_parameter_combinations(self, 
                                       param_ranges: Dict[str, List],
                                       max_combinations: int) -> List[Dict[str, Any]]:
        """
        生成参数组合
        
        Args:
            param_ranges: 参数范围
            max_combinations: 最大组合数
            
        Returns:
            List[Dict[str, Any]]: 参数组合列表
        """
        import itertools
        
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        
        combinations = list(itertools.product(*values))
        
        # 限制组合数量
        if len(combinations) > max_combinations:
            step = len(combinations) // max_combinations
            combinations = combinations[::step][:max_combinations]
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def _update_vectorization_stats(self, computation_time: float) -> None:
        """
        更新矢量化统计信息
        
        Args:
            computation_time: 计算时间
        """
        stats = self.vectorization_stats
        stats['total_computations'] += 1
        stats['vectorized_computations'] += 1
        
        # 更新平均计算时间
        total_time = stats['avg_computation_time'] * (stats['total_computations'] - 1)
        stats['avg_computation_time'] = (total_time + computation_time) / stats['total_computations']
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能摘要（重写基类方法）
        
        Returns:
            Dict[str, Any]: 性能指标字典
        """
        base_summary = super().get_performance_summary()
        
        vectorized_summary = {
            'vectorization_stats': self.vectorization_stats,
            'computation_times': self.computation_times,
            'indicators_config': self.indicators_config,
            'lookback_window': self.lookback_window,
            'cached_symbols': list(self.indicators_cache.keys())
        }
        
        base_summary.update(vectorized_summary)
        return base_summary
    
    def clear_cache(self) -> None:
        """清空所有缓存"""
        super().clear_cache()
        self.indicators_cache.clear()
        self.signals_cache.clear()
        self.historical_data.clear()
    
    # 抽象方法的默认实现
    def initialize(self, context: Any = None) -> None:
        """策略初始化"""
        self.state = StrategyState.RUNNING
        print(f"Vectorized strategy {self.name} initialized")
    
    def on_bar(self, bar_data: pd.DataFrame) -> None:
        """Bar数据事件处理"""
        if not self.is_active():
            return
            
        # 批量处理所有标的
        data_batch = {}
        for symbol in self.symbols:
            symbol_data = bar_data[bar_data.get('symbol', '') == symbol] if 'symbol' in bar_data.columns else bar_data
            if not symbol_data.empty:
                data_batch[symbol] = symbol_data
        
        if data_batch:
            signals_batch = self.batch_process_signals(data_batch)
            
            # 处理生成的信号
            for symbol, signals in signals_batch.items():
                for signal in signals:
                    self.log_signal(signal)
    
    def on_tick(self, tick_data: pd.DataFrame) -> None:
        """Tick数据事件处理"""
        # 默认实现：将tick数据聚合为bar数据处理
        self.on_bar(tick_data)