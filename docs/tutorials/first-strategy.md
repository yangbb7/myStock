# 创建你的第一个交易策略

## 概述

本教程将指导你创建第一个交易策略，并在 myStock 系统中运行回测。我们将实现一个简单的移动平均交叉策略。

## 什么是移动平均交叉策略？

移动平均交叉策略是一种经典的技术分析策略：
- 当短期移动平均线上穿长期移动平均线时，产生买入信号
- 当短期移动平均线下穿长期移动平均线时，产生卖出信号

## 步骤1：创建策略文件

创建文件 `my_first_strategy.py`：

```python
from myQuant.core.strategy_engine import BaseStrategy
from datetime import datetime

class MyFirstStrategy(BaseStrategy):
    """我的第一个交易策略 - 移动平均交叉策略"""
    
    def initialize(self):
        """初始化策略"""
        print(f"初始化策略: {self.name}")
        
        # 获取参数
        self.short_window = self.params.get('short_window', 5)
        self.long_window = self.params.get('long_window', 20)
        
        # 验证参数
        if self.short_window >= self.long_window:
            raise ValueError("短期窗口必须小于长期窗口")
        
        # 初始化状态变量
        self.prices = {}  # 存储各股票的价格历史
        self.positions = {}  # 存储各股票的持仓状态
        self.trade_count = 0  # 交易次数
        self.signals_generated = []  # 生成的信号列表
        
        print(f"策略参数: 短期窗口={self.short_window}, 长期窗口={self.long_window}")
        
    def on_bar(self, bar_data):
        """处理每个K线数据"""
        symbol = bar_data.get('symbol')
        close_price = bar_data.get('close', 0)
        current_time = bar_data.get('datetime', datetime.now())
        
        # 初始化股票数据
        if symbol not in self.prices:
            self.prices[symbol] = []
            self.positions[symbol] = 0
        
        # 更新价格历史
        self.prices[symbol].append(close_price)
        
        # 保持价格历史长度
        if len(self.prices[symbol]) > self.long_window:
            self.prices[symbol].pop(0)
        
        # 只有足够的历史数据才能计算移动平均
        if len(self.prices[symbol]) < self.long_window:
            return []
        
        # 计算移动平均
        short_ma = sum(self.prices[symbol][-self.short_window:]) / self.short_window
        long_ma = sum(self.prices[symbol]) / len(self.prices[symbol])
        
        # 生成交易信号
        signals = []
        
        # 金叉：短期MA上穿长期MA，且当前无持仓
        if short_ma > long_ma and self.positions[symbol] <= 0:
            signal = {
                'timestamp': current_time,
                'symbol': symbol,
                'signal_type': 'BUY',
                'price': close_price,
                'quantity': 1000,
                'strategy_name': self.name,
                'confidence': 0.8,
                'reason': f'金叉：短期MA({short_ma:.2f}) > 长期MA({long_ma:.2f})'
            }
            signals.append(signal)
            self.positions[symbol] = 1
            self.trade_count += 1
            self.signals_generated.append(signal)
            
        # 死叉：短期MA下穿长期MA，且当前有持仓
        elif short_ma < long_ma and self.positions[symbol] >= 0:
            signal = {
                'timestamp': current_time,
                'symbol': symbol,
                'signal_type': 'SELL',
                'price': close_price,
                'quantity': 1000,
                'strategy_name': self.name,
                'confidence': 0.8,
                'reason': f'死叉：短期MA({short_ma:.2f}) < 长期MA({long_ma:.2f})'
            }
            signals.append(signal)
            self.positions[symbol] = -1
            self.trade_count += 1
            self.signals_generated.append(signal)
        
        return signals
    
    def on_tick(self, tick_data):
        """处理tick数据（本策略不使用）"""
        return []
    
    def finalize(self):
        """策略结束时的清理工作"""
        print(f"策略 {self.name} 执行完毕")
        print(f"总交易次数: {self.trade_count}")
        print(f"总信号数: {len(self.signals_generated)}")
        
        # 打印一些统计信息
        buy_signals = [s for s in self.signals_generated if s['signal_type'] == 'BUY']
        sell_signals = [s for s in self.signals_generated if s['signal_type'] == 'SELL']
        
        print(f"买入信号: {len(buy_signals)}")
        print(f"卖出信号: {len(sell_signals)}")
        
        # 打印最近的几个信号
        if self.signals_generated:
            print("最近的信号:")
            for signal in self.signals_generated[-3:]:
                print(f"  {signal['timestamp']}: {signal['symbol']} {signal['signal_type']} @ {signal['price']}")
```

## 步骤2：创建测试脚本

创建文件 `test_strategy.py`：

```python
from myQuant.core.trading_system import TradingSystem
from myQuant import create_default_config
from my_first_strategy import MyFirstStrategy
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_data():
    """创建测试数据"""
    # 生成100天的模拟数据
    symbols = ['000001.SZ', '000002.SZ']
    days = 100
    base_date = datetime.now() - timedelta(days=days)
    
    data_list = []
    
    for symbol in symbols:
        # 不同股票使用不同的基础价格
        base_price = 15.0 if symbol == '000001.SZ' else 25.0
        
        for i in range(days):
            current_date = base_date + timedelta(days=i)
            
            # 模拟价格变化（带趋势）
            trend = 0.001 if i < 50 else -0.001  # 前50天上涨，后50天下跌
            noise = np.random.normal(0, 0.02)  # 随机噪声
            
            price_change = trend + noise
            base_price = max(1.0, base_price * (1 + price_change))
            
            # 生成OHLC数据
            open_price = base_price * np.random.uniform(0.99, 1.01)
            high_price = base_price * np.random.uniform(1.00, 1.03)
            low_price = base_price * np.random.uniform(0.97, 1.00)
            close_price = base_price
            volume = np.random.randint(500000, 2000000)
            
            data_list.append({
                'datetime': current_date,
                'symbol': symbol,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume,
                'adj_close': round(close_price, 2)
            })
    
    return pd.DataFrame(data_list)

def run_strategy_test():
    """运行策略测试"""
    print("🚀 开始策略测试...")
    
    # 1. 创建测试数据
    print("📊 生成测试数据...")
    sample_data = create_sample_data()
    print(f"生成了 {len(sample_data)} 条数据")
    
    # 2. 创建配置
    config = create_default_config()
    
    # 3. 创建交易系统
    print("⚙️  初始化交易系统...")
    trading_system = TradingSystem(config)
    
    # 4. 创建策略
    print("🎯 创建策略...")
    strategy = MyFirstStrategy(
        name="MyFirstStrategy",
        symbols=['000001.SZ', '000002.SZ'],
        params={
            'short_window': 5,
            'long_window': 20
        }
    )
    
    # 5. 添加策略到系统
    trading_system.add_strategy(strategy)
    print(f"✅ 策略添加成功: {strategy.name}")
    
    # 6. 运行策略
    print("🔄 运行策略...")
    signals_generated = 0
    
    # 按日期排序数据
    sample_data = sample_data.sort_values(['datetime', 'symbol']).reset_index(drop=True)
    
    # 逐条处理数据
    for _, row in sample_data.iterrows():
        bar_data = row.to_dict()
        
        # 让策略处理数据
        signals = strategy.on_bar(bar_data)
        signals_generated += len(signals)
        
        # 打印前5个信号
        if signals and signals_generated <= 5:
            signal = signals[0]
            print(f"🔔 信号 {signals_generated}: {signal['symbol']} {signal['signal_type']} @ {signal['price']:.2f}")
            print(f"   原因: {signal['reason']}")
    
    # 7. 结束策略
    strategy.finalize()
    
    # 8. 显示结果
    print("\n" + "="*50)
    print("📊 策略测试结果")
    print("="*50)
    print(f"💰 总信号数: {signals_generated}")
    print(f"📈 买入信号数: {len([s for s in strategy.signals_generated if s['signal_type'] == 'BUY'])}")
    print(f"📉 卖出信号数: {len([s for s in strategy.signals_generated if s['signal_type'] == 'SELL'])}")
    print(f"🔄 总交易次数: {strategy.trade_count}")
    
    # 9. 分析信号分布
    if strategy.signals_generated:
        print("\n📊 信号分布:")
        for symbol in ['000001.SZ', '000002.SZ']:
            symbol_signals = [s for s in strategy.signals_generated if s['symbol'] == symbol]
            print(f"  {symbol}: {len(symbol_signals)} 个信号")
    
    print("="*50)
    print("✅ 策略测试完成!")

if __name__ == "__main__":
    run_strategy_test()
```

## 步骤3：运行测试

```bash
python test_strategy.py
```

预期输出：
```
🚀 开始策略测试...
📊 生成测试数据...
生成了 200 条数据
⚙️  初始化交易系统...
🎯 创建策略...
初始化策略: MyFirstStrategy
策略参数: 短期窗口=5, 长期窗口=20
✅ 策略添加成功: MyFirstStrategy
🔄 运行策略...
🔔 信号 1: 000001.SZ BUY @ 15.23
   原因: 金叉：短期MA(15.12) > 长期MA(14.98)
🔔 信号 2: 000002.SZ BUY @ 25.45
   原因: 金叉：短期MA(25.31) > 长期MA(25.02)
...
策略 MyFirstStrategy 执行完毕
总交易次数: 8
总信号数: 8
买入信号: 4
卖出信号: 4

==================================================
📊 策略测试结果
==================================================
💰 总信号数: 8
📈 买入信号数: 4
📉 卖出信号数: 4
🔄 总交易次数: 8

📊 信号分布:
  000001.SZ: 4 个信号
  000002.SZ: 4 个信号
==================================================
✅ 策略测试完成!
```

## 步骤4：优化策略

### 添加止损止盈

```python
def initialize(self):
    # ... 其他初始化代码
    self.stop_loss = self.params.get('stop_loss', 0.05)  # 5%止损
    self.take_profit = self.params.get('take_profit', 0.10)  # 10%止盈
    self.entry_prices = {}  # 记录入场价格

def on_bar(self, bar_data):
    # ... 移动平均逻辑
    
    # 止损止盈逻辑
    if self.positions[symbol] > 0 and symbol in self.entry_prices:
        entry_price = self.entry_prices[symbol]
        current_return = (close_price - entry_price) / entry_price
        
        # 止损
        if current_return < -self.stop_loss:
            signal = {
                'timestamp': current_time,
                'symbol': symbol,
                'signal_type': 'SELL',
                'price': close_price,
                'quantity': 1000,
                'strategy_name': self.name,
                'reason': f'止损: 收益率{current_return:.2%} < -{self.stop_loss:.2%}'
            }
            signals.append(signal)
            self.positions[symbol] = 0
            del self.entry_prices[symbol]
            
        # 止盈
        elif current_return > self.take_profit:
            signal = {
                'timestamp': current_time,
                'symbol': symbol,
                'signal_type': 'SELL',
                'price': close_price,
                'quantity': 1000,
                'strategy_name': self.name,
                'reason': f'止盈: 收益率{current_return:.2%} > {self.take_profit:.2%}'
            }
            signals.append(signal)
            self.positions[symbol] = 0
            del self.entry_prices[symbol]
```

### 添加成交量过滤

```python
def on_bar(self, bar_data):
    # ... 其他逻辑
    
    volume = bar_data.get('volume', 0)
    avg_volume = sum(self.volumes[symbol]) / len(self.volumes[symbol])
    
    # 只有成交量足够大才交易
    if volume < avg_volume * 0.5:
        return []
    
    # ... 生成信号
```

## 步骤5：与回测引擎集成

```python
from myQuant.core.engines.backtest_engine import BacktestEngine

def run_backtest():
    """运行完整回测"""
    # 创建回测引擎
    backtest_config = {
        'initial_capital': 1000000,
        'commission_rate': 0.0003,
        'start_date': '2023-01-01',
        'end_date': '2023-12-31'
    }
    
    backtest_engine = BacktestEngine(backtest_config)
    
    # 添加策略
    strategy = MyFirstStrategy(
        name="MyFirstStrategy",
        symbols=['000001.SZ', '000002.SZ'],
        params={'short_window': 5, 'long_window': 20}
    )
    
    backtest_engine.add_strategy(strategy)
    
    # 运行回测
    result = backtest_engine.run_backtest()
    
    # 显示结果
    print(f"最终价值: ¥{result['final_value']:,.2f}")
    print(f"总收益率: {result['total_return']:.2%}")
    print(f"夏普比率: {result['sharpe_ratio']:.3f}")
    print(f"最大回撤: {result['max_drawdown']:.2%}")
```

## 常见问题

### Q1: 策略没有生成信号？
- 检查数据是否足够（需要至少 `long_window` 条数据）
- 检查移动平均线计算是否正确
- 添加调试信息查看中间结果

### Q2: 信号过于频繁？
- 增加信号过滤条件
- 添加冷却期（避免连续信号）
- 提高信号阈值

### Q3: 如何添加更多技术指标？
- 在 `initialize()` 中初始化指标状态
- 在 `on_bar()` 中更新指标值
- 结合多个指标生成信号

## 下一步

1. 尝试不同的参数组合
2. 添加更多技术指标
3. 实现多股票组合策略
4. 集成风险管理模块
5. 进行绩效分析和优化

恭喜！你已经成功创建了第一个交易策略。继续探索更多高级功能，构建更复杂的策略！