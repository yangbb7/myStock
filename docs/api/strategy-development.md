# 策略开发API

## 概述

myStock 策略开发API 提供了完整的策略开发框架，包括策略基类、信号生成、事件处理等功能。

## 策略基类

### BaseStrategy

所有策略必须继承自 `BaseStrategy` 类。

```python
from myQuant.core.strategy_engine import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, name, symbols, params=None):
        super().__init__(name, symbols, params)
        
    def initialize(self):
        """策略初始化"""
        pass
        
    def on_bar(self, bar_data):
        """处理K线数据"""
        return []
        
    def on_tick(self, tick_data):
        """处理tick数据"""
        return []
        
    def finalize(self):
        """策略结束"""
        pass
```

## 核心方法

### `initialize()`

策略初始化方法，在策略开始运行前调用。

**用途：**
- 初始化策略参数
- 设置状态变量
- 准备数据结构

**示例：**
```python
def initialize(self):
    self.short_window = self.params.get('short_window', 5)
    self.long_window = self.params.get('long_window', 20)
    self.position = 0
    self.prices = []
    self.signals = []
```

### `on_bar(bar_data)`

处理K线数据的核心方法。

**参数：**
- `bar_data` (dict): K线数据

**返回：**
- `list`: 交易信号列表

**bar_data 格式：**
```python
{
    'datetime': datetime.datetime,
    'symbol': str,
    'open': float,
    'high': float,
    'low': float,
    'close': float,
    'volume': int,
    'adj_close': float  # 可选
}
```

**信号格式：**
```python
{
    'timestamp': datetime.datetime,
    'symbol': str,
    'signal_type': str,  # 'BUY', 'SELL', 'HOLD'
    'price': float,
    'quantity': int,
    'strategy_name': str,
    'confidence': float,  # 可选，0-1
    'reason': str  # 可选，信号原因
}
```

**示例：**
```python
def on_bar(self, bar_data):
    symbol = bar_data['symbol']
    close_price = bar_data['close']
    
    # 更新价格历史
    self.prices.append(close_price)
    if len(self.prices) > self.long_window:
        self.prices.pop(0)
    
    # 生成信号
    if len(self.prices) >= self.long_window:
        short_ma = sum(self.prices[-self.short_window:]) / self.short_window
        long_ma = sum(self.prices) / len(self.prices)
        
        if short_ma > long_ma and self.position <= 0:
            return [{
                'timestamp': bar_data['datetime'],
                'symbol': symbol,
                'signal_type': 'BUY',
                'price': close_price,
                'quantity': 1000,
                'strategy_name': self.name,
                'confidence': 0.8,
                'reason': f'短期均线{short_ma:.2f}上穿长期均线{long_ma:.2f}'
            }]
        elif short_ma < long_ma and self.position >= 0:
            return [{
                'timestamp': bar_data['datetime'],
                'symbol': symbol,
                'signal_type': 'SELL',
                'price': close_price,
                'quantity': 1000,
                'strategy_name': self.name,
                'confidence': 0.8,
                'reason': f'短期均线{short_ma:.2f}下穿长期均线{long_ma:.2f}'
            }]
    
    return []
```

### `on_tick(tick_data)`

处理tick数据的方法，用于高频交易。

**参数：**
- `tick_data` (dict): tick数据

**返回：**
- `list`: 交易信号列表

**示例：**
```python
def on_tick(self, tick_data):
    symbol = tick_data['symbol']
    price = tick_data['price']
    
    # 高频交易逻辑
    if self.should_trade(price):
        return [{
            'timestamp': tick_data['datetime'],
            'symbol': symbol,
            'signal_type': 'BUY',
            'price': price,
            'quantity': 100,
            'strategy_name': self.name
        }]
    
    return []
```

### `finalize()`

策略结束方法，在策略停止运行时调用。

**用途：**
- 清理资源
- 保存状态
- 输出统计信息

**示例：**
```python
def finalize(self):
    print(f"策略 {self.name} 结束")
    print(f"总信号数: {len(self.signals)}")
    print(f"最终持仓: {self.position}")
```

## 内置方法

### `get_portfolio_value()`

获取当前投资组合价值。

**返回：**
- `float`: 投资组合价值

### `get_position(symbol)`

获取指定股票的持仓。

**参数：**
- `symbol` (str): 股票代码

**返回：**
- `int`: 持仓数量

### `get_cash()`

获取当前现金余额。

**返回：**
- `float`: 现金余额

### `log_info(message)`

记录信息日志。

**参数：**
- `message` (str): 日志消息

### `log_warning(message)`

记录警告日志。

**参数：**
- `message` (str): 日志消息

### `log_error(message)`

记录错误日志。

**参数：**
- `message` (str): 日志消息

## 策略示例

### 1. 移动平均策略

```python
from myQuant.core.strategy_engine import BaseStrategy

class MovingAverageStrategy(BaseStrategy):
    """移动平均策略"""
    
    def initialize(self):
        self.short_window = self.params.get('short_window', 5)
        self.long_window = self.params.get('long_window', 20)
        self.prices = {}
        self.positions = {}
        
    def on_bar(self, bar_data):
        symbol = bar_data['symbol']
        close_price = bar_data['close']
        
        # 初始化价格历史
        if symbol not in self.prices:
            self.prices[symbol] = []
            self.positions[symbol] = 0
        
        # 更新价格历史
        self.prices[symbol].append(close_price)
        if len(self.prices[symbol]) > self.long_window:
            self.prices[symbol].pop(0)
        
        # 计算移动平均
        if len(self.prices[symbol]) >= self.long_window:
            short_ma = sum(self.prices[symbol][-self.short_window:]) / self.short_window
            long_ma = sum(self.prices[symbol]) / len(self.prices[symbol])
            
            # 生成信号
            if short_ma > long_ma and self.positions[symbol] <= 0:
                self.positions[symbol] = 1
                return [{
                    'timestamp': bar_data['datetime'],
                    'symbol': symbol,
                    'signal_type': 'BUY',
                    'price': close_price,
                    'quantity': 1000,
                    'strategy_name': self.name
                }]
            elif short_ma < long_ma and self.positions[symbol] >= 0:
                self.positions[symbol] = -1
                return [{
                    'timestamp': bar_data['datetime'],
                    'symbol': symbol,
                    'signal_type': 'SELL',
                    'price': close_price,
                    'quantity': 1000,
                    'strategy_name': self.name
                }]
        
        return []
```

### 2. 均值回归策略

```python
import numpy as np
from myQuant.core.strategy_engine import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""
    
    def initialize(self):
        self.lookback = self.params.get('lookback', 20)
        self.threshold = self.params.get('threshold', 2.0)
        self.prices = {}
        self.positions = {}
        
    def on_bar(self, bar_data):
        symbol = bar_data['symbol']
        close_price = bar_data['close']
        
        # 初始化
        if symbol not in self.prices:
            self.prices[symbol] = []
            self.positions[symbol] = 0
        
        # 更新价格历史
        self.prices[symbol].append(close_price)
        if len(self.prices[symbol]) > self.lookback:
            self.prices[symbol].pop(0)
        
        # 计算Z-Score
        if len(self.prices[symbol]) >= self.lookback:
            prices_array = np.array(self.prices[symbol])
            mean_price = np.mean(prices_array)
            std_price = np.std(prices_array)
            
            if std_price > 0:
                z_score = (close_price - mean_price) / std_price
                
                # 生成信号
                if z_score < -self.threshold and self.positions[symbol] <= 0:
                    self.positions[symbol] = 1
                    return [{
                        'timestamp': bar_data['datetime'],
                        'symbol': symbol,
                        'signal_type': 'BUY',
                        'price': close_price,
                        'quantity': 1000,
                        'strategy_name': self.name,
                        'reason': f'Z-Score: {z_score:.2f} < -{self.threshold}'
                    }]
                elif z_score > self.threshold and self.positions[symbol] >= 0:
                    self.positions[symbol] = -1
                    return [{
                        'timestamp': bar_data['datetime'],
                        'symbol': symbol,
                        'signal_type': 'SELL',
                        'price': close_price,
                        'quantity': 1000,
                        'strategy_name': self.name,
                        'reason': f'Z-Score: {z_score:.2f} > {self.threshold}'
                    }]
        
        return []
```

### 3. 动量策略

```python
from myQuant.core.strategy_engine import BaseStrategy

class MomentumStrategy(BaseStrategy):
    """动量策略"""
    
    def initialize(self):
        self.momentum_window = self.params.get('momentum_window', 10)
        self.threshold = self.params.get('threshold', 0.05)
        self.prices = {}
        self.positions = {}
        
    def on_bar(self, bar_data):
        symbol = bar_data['symbol']
        close_price = bar_data['close']
        
        # 初始化
        if symbol not in self.prices:
            self.prices[symbol] = []
            self.positions[symbol] = 0
        
        # 更新价格历史
        self.prices[symbol].append(close_price)
        if len(self.prices[symbol]) > self.momentum_window:
            self.prices[symbol].pop(0)
        
        # 计算动量
        if len(self.prices[symbol]) >= self.momentum_window:
            momentum = (close_price - self.prices[symbol][0]) / self.prices[symbol][0]
            
            # 生成信号
            if momentum > self.threshold and self.positions[symbol] <= 0:
                self.positions[symbol] = 1
                return [{
                    'timestamp': bar_data['datetime'],
                    'symbol': symbol,
                    'signal_type': 'BUY',
                    'price': close_price,
                    'quantity': 1000,
                    'strategy_name': self.name,
                    'reason': f'动量: {momentum:.2%} > {self.threshold:.2%}'
                }]
            elif momentum < -self.threshold and self.positions[symbol] >= 0:
                self.positions[symbol] = -1
                return [{
                    'timestamp': bar_data['datetime'],
                    'symbol': symbol,
                    'signal_type': 'SELL',
                    'price': close_price,
                    'quantity': 1000,
                    'strategy_name': self.name,
                    'reason': f'动量: {momentum:.2%} < {-self.threshold:.2%}'
                }]
        
        return []
```

## 策略参数

### 参数传递

```python
# 创建策略时传递参数
strategy = MovingAverageStrategy(
    name="MA_Strategy",
    symbols=['000001.SZ', '000002.SZ'],
    params={
        'short_window': 5,
        'long_window': 20,
        'stop_loss': 0.05,
        'take_profit': 0.10
    }
)
```

### 参数验证

```python
def initialize(self):
    # 参数验证
    if self.params.get('short_window', 5) <= 0:
        raise ValueError("短期窗口必须大于0")
    
    if self.params.get('long_window', 20) <= self.params.get('short_window', 5):
        raise ValueError("长期窗口必须大于短期窗口")
    
    # 设置参数
    self.short_window = self.params.get('short_window', 5)
    self.long_window = self.params.get('long_window', 20)
```

## 策略状态管理

### 状态变量

```python
def initialize(self):
    # 状态变量
    self.position = 0
    self.entry_price = 0
    self.entry_time = None
    self.profit_loss = 0
    self.trade_count = 0
    self.win_count = 0
```

### 状态更新

```python
def on_bar(self, bar_data):
    # 更新状态
    if signal_type == 'BUY':
        self.position = quantity
        self.entry_price = price
        self.entry_time = bar_data['datetime']
        self.trade_count += 1
    elif signal_type == 'SELL':
        if self.position > 0:
            pnl = (price - self.entry_price) * self.position
            self.profit_loss += pnl
            if pnl > 0:
                self.win_count += 1
        self.position = 0
        self.entry_price = 0
```

## 错误处理

```python
def on_bar(self, bar_data):
    try:
        # 策略逻辑
        return self.generate_signals(bar_data)
    except Exception as e:
        self.log_error(f"策略执行错误: {e}")
        return []
```

## 调试技巧

### 日志记录

```python
def on_bar(self, bar_data):
    symbol = bar_data['symbol']
    close_price = bar_data['close']
    
    self.log_info(f"处理 {symbol} 数据: 价格={close_price}")
    
    # 策略逻辑
    signals = self.generate_signals(bar_data)
    
    if signals:
        self.log_info(f"生成信号: {signals}")
    
    return signals
```

### 状态监控

```python
def on_bar(self, bar_data):
    # 监控关键状态
    if self.trade_count % 10 == 0:
        self.log_info(f"交易统计: 总数={self.trade_count}, 胜率={self.win_count/self.trade_count:.2%}")
```

## 最佳实践

1. **参数化设计**: 所有关键参数都应该可配置
2. **错误处理**: 总是包含适当的错误处理
3. **日志记录**: 记录关键操作和决策
4. **状态管理**: 正确维护策略状态
5. **性能考虑**: 避免不必要的计算
6. **代码复用**: 提取通用逻辑到工具方法

## 相关文档

- [核心API](core-api.md)
- [回测引擎API](backtest-engine.md)
- [第一个策略教程](../tutorials/first-strategy.md)