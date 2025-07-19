# -*- coding: utf-8 -*-
"""
增强策略框架演示 - 展示矢量化计算、事件驱动、生命周期管理和参数优化的完整示例
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any

from myQuant.core.strategy.vectorized_strategy import VectorizedStrategy
from myQuant.core.strategy.technical_indicators import TechnicalIndicators
from myQuant.core.strategy.strategy_lifecycle import StrategyLifecycleManager, LifecycleTransition
from myQuant.core.events.enhanced_event_types import EventFactory, SignalEvent
from myQuant.core.events.advanced_event_bus import AdvancedEventBus
from myQuant.core.optimization.parameter_optimizer import ParameterOptimizer
from myQuant.core.optimization.parameter_space import ParameterSpaceBuilder
from myQuant.core.optimization.objective_functions import SharpeRatioObjective, CompositeObjective
from myQuant.core.models.signals import Signal, SignalType


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class EnhancedMovingAverageStrategy(VectorizedStrategy):
    """增强的移动平均策略 - 展示所有新功能"""
    
    def __init__(self, name: str, symbols: List[str], params: Dict[str, Any] = None, **kwargs):
        super().__init__(name, symbols, params, **kwargs)
        
        # 策略参数
        self.fast_period = self.get_param('fast_period', 10)
        self.slow_period = self.get_param('slow_period', 30)
        self.signal_threshold = self.get_param('signal_threshold', 0.02)
        self.position_size = self.get_param('position_size', 0.5)
        
        # 设置技术指标配置
        self.set_indicators_config({
            'sma': {'windows': [self.fast_period, self.slow_period]},
            'ema': {'windows': [self.fast_period, self.slow_period]},
            'rsi': {'window': 14},
            'bollinger': {'window': 20, 'std': 2.0},
            'atr': {'window': 14}
        })
        
        # 事件总线（将在外部设置）
        self.event_bus = None
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{name}")
    
    def initialize(self, context: Any = None) -> None:
        """策略初始化"""
        super().initialize(context)
        self.logger.info(f"Enhanced MA Strategy initialized with fast={self.fast_period}, slow={self.slow_period}")
    
    def generate_signals_vectorized(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """矢量化信号生成"""
        try:
            signals_df = pd.DataFrame(index=data.index)
            signals_df['symbol'] = symbol
            signals_df['buy_signal'] = False
            signals_df['sell_signal'] = False
            signals_df['signal_strength'] = 0.0
            signals_df['signal_reason'] = ''
            
            if len(data) < max(self.fast_period, self.slow_period):
                return signals_df
            
            # 获取移动平均线
            fast_ma_col = f'sma_{self.fast_period}'
            slow_ma_col = f'sma_{self.slow_period}'
            
            if fast_ma_col not in data.columns or slow_ma_col not in data.columns:
                self.logger.warning(f"Missing MA columns for {symbol}")
                return signals_df
            
            fast_ma = data[fast_ma_col]
            slow_ma = data[slow_ma_col]
            
            # 计算信号强度
            ma_ratio = (fast_ma / slow_ma - 1).fillna(0)
            
            # 买入信号：快线上穿慢线且幅度足够
            buy_condition = (
                (fast_ma > slow_ma) & 
                (fast_ma.shift(1) <= slow_ma.shift(1)) &
                (ma_ratio > self.signal_threshold)
            )
            
            # 卖出信号：快线下穿慢线且幅度足够
            sell_condition = (
                (fast_ma < slow_ma) & 
                (fast_ma.shift(1) >= slow_ma.shift(1)) &
                (ma_ratio < -self.signal_threshold)
            )
            
            # 使用额外指标增强信号
            if 'rsi' in data.columns:
                rsi = data['rsi']
                # RSI过滤：买入时RSI不能过高，卖出时RSI不能过低
                buy_condition = buy_condition & (rsi < 70)
                sell_condition = sell_condition & (rsi > 30)
            
            if 'atr' in data.columns:
                atr = data['atr']
                # 使用ATR调整信号强度
                atr_normalized = atr / data['close'] if 'close' in data.columns else atr / 100
                signal_strength_adjustment = np.clip(atr_normalized * 2, 0.1, 2.0)
            else:
                signal_strength_adjustment = 1.0
            
            # 设置信号
            signals_df.loc[buy_condition, 'buy_signal'] = True
            signals_df.loc[sell_condition, 'sell_signal'] = True
            
            # 计算信号强度
            signals_df.loc[buy_condition, 'signal_strength'] = (
                np.abs(ma_ratio[buy_condition]) * signal_strength_adjustment[buy_condition] * self.position_size
            ).clip(0.1, 1.0)
            
            signals_df.loc[sell_condition, 'signal_strength'] = (
                np.abs(ma_ratio[sell_condition]) * signal_strength_adjustment[sell_condition] * self.position_size
            ).clip(0.1, 1.0)
            
            # 添加信号原因
            signals_df.loc[buy_condition, 'signal_reason'] = f'MA_CROSS_UP_fast_{self.fast_period}_slow_{self.slow_period}'
            signals_df.loc[sell_condition, 'signal_reason'] = f'MA_CROSS_DOWN_fast_{self.fast_period}_slow_{self.slow_period}'
            
            return signals_df
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
            return pd.DataFrame(index=data.index, columns=['symbol', 'buy_signal', 'sell_signal', 'signal_strength', 'signal_reason'])
    
    def on_bar(self, bar_data: pd.DataFrame) -> None:
        """处理Bar数据并发布事件"""
        super().on_bar(bar_data)
        
        # 发布市场数据事件
        if self.event_bus:
            for symbol in self.symbols:
                symbol_data = bar_data[bar_data.get('symbol', '') == symbol] if 'symbol' in bar_data.columns else bar_data
                if not symbol_data.empty:
                    market_event = EventFactory.create_market_data_event(
                        symbol=symbol,
                        data_type='bar',
                        data=symbol_data.to_dict('records')[-1],  # 最新的bar
                        source=self.name
                    )
                    self.event_bus.publish(market_event)
    
    def check_health(self) -> bool:
        """健康检查"""
        # 检查是否有足够的历史数据
        min_data_required = max(self.fast_period, self.slow_period) + 10
        
        for symbol in self.symbols:
            data = self.get_data(symbol)
            if len(data) < min_data_required:
                self.logger.warning(f"Insufficient data for {symbol}: {len(data)} < {min_data_required}")
                return False
        
        return True
    
    def validate_params(self) -> bool:
        """验证参数"""
        if self.fast_period >= self.slow_period:
            self.logger.error("Fast period must be less than slow period")
            return False
        
        if self.signal_threshold <= 0 or self.signal_threshold > 0.5:
            self.logger.error("Signal threshold must be between 0 and 0.5")
            return False
        
        if self.position_size <= 0 or self.position_size > 1:
            self.logger.error("Position size must be between 0 and 1")
            return False
        
        return True


def create_sample_data(symbols: List[str], start_date: str, end_date: str, freq: str = 'D') -> pd.DataFrame:
    """创建示例数据"""
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    all_data = []
    
    for symbol in symbols:
        # 生成模拟价格数据
        np.random.seed(hash(symbol) % 2**32)  # 确保每个symbol的数据一致
        
        n_periods = len(date_range)
        returns = np.random.normal(0.001, 0.02, n_periods)  # 日收益率
        
        # 添加趋势
        trend = np.linspace(0, 0.5, n_periods)
        returns += trend * 0.001
        
        # 计算价格
        prices = 100 * np.cumprod(1 + returns)
        
        # 创建OHLCV数据
        for i, (date, price) in enumerate(zip(date_range, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.randint(10000, 100000)
            
            all_data.append({
                'date': date,
                'symbol': symbol,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
    
    return pd.DataFrame(all_data).set_index('date')


def signal_handler(event):
    """信号事件处理器"""
    if hasattr(event, 'signal_type') and event.signal_type in ['BUY', 'SELL']:
        print(f"📈 Signal received: {event.signal_type} {event.symbol} @ {event.price:.2f} "
              f"(strength: {event.signal_strength:.2f})")


def main():
    """主演示函数"""
    print("🚀 Enhanced Strategy Framework Demo")
    print("=" * 50)
    
    # 1. 创建示例数据
    print("📊 Creating sample data...")
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    data = create_sample_data(
        symbols=symbols,
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    print(f"Data shape: {data.shape}")
    
    # 2. 设置事件总线
    print("\n🔄 Setting up event bus...")
    event_bus = AdvancedEventBus(max_workers=4)
    event_bus.start()
    
    # 订阅信号事件
    event_bus.subscribe(
        handler=signal_handler,
        event_types=[SignalEvent],
        async_handler=False
    )
    
    # 3. 创建策略实例
    print("\n🎯 Creating enhanced strategy...")
    strategy = EnhancedMovingAverageStrategy(
        name="enhanced_ma_strategy",
        symbols=symbols,
        params={
            'fast_period': 10,
            'slow_period': 30,
            'signal_threshold': 0.02,
            'position_size': 0.5
        },
        lookback_window=200
    )
    strategy.event_bus = event_bus
    
    # 4. 生命周期管理演示
    print("\n🔄 Demonstrating lifecycle management...")
    lifecycle_manager = StrategyLifecycleManager(
        enable_health_check=True,
        health_check_interval=10
    )
    
    # 注册策略
    lifecycle_manager.register_strategy(strategy)
    
    # 初始化策略
    print("Initializing strategy...")
    success = lifecycle_manager.initialize_strategy(strategy.name)
    print(f"Initialization: {'✅ Success' if success else '❌ Failed'}")
    
    # 启动策略
    print("Starting strategy...")
    success = lifecycle_manager.start_strategy(strategy.name)
    print(f"Start: {'✅ Success' if success else '❌ Failed'}")
    
    # 5. 矢量化计算演示
    print("\n⚡ Demonstrating vectorized computation...")
    
    # 按symbol分组数据并处理
    for symbol in symbols:
        symbol_data = data[data['symbol'] == symbol].copy()
        
        # 更新数据到策略
        strategy.update_data(symbol, symbol_data)
        
        # 计算技术指标
        indicators_data = strategy.compute_indicators(symbol)
        
        if not indicators_data.empty:
            print(f"📊 {symbol}: Computed {len([col for col in indicators_data.columns if col not in symbol_data.columns])} indicators")
            
            # 生成信号
            signals_df = strategy.generate_signals_vectorized(symbol, indicators_data)
            buy_signals = signals_df['buy_signal'].sum()
            sell_signals = signals_df['sell_signal'].sum()
            
            print(f"📈 {symbol}: Generated {buy_signals} buy signals, {sell_signals} sell signals")
    
    # 6. 参数优化演示
    print("\n🎛️  Demonstrating parameter optimization...")
    
    # 定义参数空间
    param_space = (ParameterSpaceBuilder()
                   .add_integer('fast_period', 5, 20, default=10)
                   .add_integer('slow_period', 25, 50, default=30)
                   .add_float('signal_threshold', 0.01, 0.05, default=0.02)
                   .add_float('position_size', 0.1, 1.0, default=0.5)
                   .add_constraint(lambda params: params['fast_period'] < params['slow_period'])
                   .build())
    
    # 创建目标函数
    objective = SharpeRatioObjective(
        strategy_class=EnhancedMovingAverageStrategy,
        data=data,
        risk_free_rate=0.02
    )
    
    # 创建优化器
    optimizer = ParameterOptimizer(
        strategy_class=EnhancedMovingAverageStrategy,
        parameter_space=param_space,
        data=data,
        objective_function=objective,
        algorithm='RandomSearch',  # 使用随机搜索（速度快）
        n_jobs=1,
        verbose=True
    )
    
    # 运行优化（少量试验用于演示）
    print("Running parameter optimization...")
    result = optimizer.optimize(n_trials=20)
    
    print(f"🏆 Best parameters: {result.best_params}")
    print(f"🏆 Best score (Sharpe ratio): {result.best_score:.4f}")
    print(f"⏱️  Optimization time: {result.optimization_time:.2f} seconds")
    
    # 7. 性能分析演示
    print("\n📊 Demonstrating performance analysis...")
    
    # 使用最佳参数创建新策略实例
    optimized_strategy = EnhancedMovingAverageStrategy(
        name="optimized_ma_strategy",
        symbols=symbols,
        params=result.best_params,
        lookback_window=200
    )
    
    # 模拟策略运行并计算收益
    returns = []
    for symbol in symbols:
        symbol_data = data[data['symbol'] == symbol].copy()
        optimized_strategy.update_data(symbol, symbol_data)
        indicators_data = optimized_strategy.compute_indicators(symbol)
        
        if not indicators_data.empty:
            signals_df = optimized_strategy.generate_signals_vectorized(symbol, indicators_data)
            
            # 简单的收益计算（实际应该使用回测引擎）
            symbol_returns = symbol_data['close'].pct_change().fillna(0)
            
            # 模拟持仓：买入信号后持有，卖出信号后平仓
            positions = np.zeros(len(signals_df))
            current_position = 0
            
            for i, row in signals_df.iterrows():
                if row['buy_signal']:
                    current_position = row['signal_strength']
                elif row['sell_signal']:
                    current_position = 0
                
                idx = symbol_data.index.get_loc(i)
                if idx < len(positions):
                    positions[idx] = current_position
            
            # 策略收益
            strategy_returns = pd.Series(positions[:-1], index=symbol_data.index[1:]) * symbol_returns.iloc[1:]
            returns.append(strategy_returns)
    
    # 合并所有symbol的收益
    if returns:
        combined_returns = pd.concat(returns).groupby(level=0).mean()
        
        from myQuant.core.strategy.strategy_performance import StrategyPerformance
        
        performance = StrategyPerformance(combined_returns)
        metrics = performance.calculate_metrics()
        
        print(f"📈 Total Return: {metrics.total_return:.2%}")
        print(f"📊 Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
        print(f"📉 Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"🎯 Win Rate: {metrics.win_rate:.2%}")
    
    # 8. 策略状态管理
    print("\n🔄 Strategy lifecycle status:")
    status = lifecycle_manager.get_all_strategies_status()
    for strategy_name, info in status.items():
        print(f"  {strategy_name}: {info['phase']} (active: {info['active']})")
    
    # 9. 事件总线统计
    print("\n📊 Event bus metrics:")
    metrics = event_bus.get_metrics()
    print(f"  Total events processed: {metrics.get('total_events', 0)}")
    print(f"  Events by category: {metrics.get('events_by_category', {})}")
    
    # 清理
    print("\n🧹 Cleaning up...")
    
    # 停止策略
    lifecycle_manager.stop_strategy(strategy.name)
    lifecycle_manager.shutdown()
    
    # 停止事件总线
    event_bus.stop()
    
    print("✅ Demo completed successfully!")


if __name__ == "__main__":
    main()