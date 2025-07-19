# 核心API参考

## 概述

myStock 核心 API 提供了完整的量化交易功能，包括数据管理、策略引擎、回测引擎、绩效分析等。

## 快速导入

```python
from myQuant.infrastructure.container import get_container

# 获取容器
container = get_container()

# 获取核心组件
data_manager = container.data_manager()
strategy_engine = container.strategy_engine()
backtest_engine = container.backtest_engine()
performance_analyzer = container.performance_analyzer()
trading_system = container.trading_system()
```

## 核心组件

### 1. TradingSystem - 交易系统

主要的交易系统类，协调所有组件。

```python
from myQuant.core.trading_system import TradingSystem

# 创建交易系统
config = {
    'initial_capital': 1000000,
    'commission_rate': 0.0003,
    'min_commission': 5.0
}
trading_system = TradingSystem(config)
```

#### 方法

##### `add_strategy(strategy)`
添加交易策略到系统。

**参数:**
- `strategy` (BaseStrategy): 策略对象

**返回:** 
- `str`: 策略ID

**示例:**
```python
from myQuant.core.strategy_engine import BaseStrategy

class MyStrategy(BaseStrategy):
    # ... 策略实现
    pass

strategy = MyStrategy(name="MyStrategy", symbols=['000001.SZ'])
strategy_id = trading_system.add_strategy(strategy)
```

##### `process_market_tick(tick_data)`
处理市场tick数据。

**参数:**
- `tick_data` (dict): tick数据字典

**返回:** 
- `dict`: 处理结果

**示例:**
```python
tick_data = {
    'datetime': datetime.now(),
    'symbol': '000001.SZ',
    'close': 12.50,
    'volume': 1000000,
    'open': 12.45,
    'high': 12.55,
    'low': 12.40
}

result = trading_system.process_market_tick(tick_data)
print(f"信号数: {result['signals_count']}")
```

##### `get_system_status()`
获取系统状态。

**返回:** 
- `dict`: 系统状态信息

**示例:**
```python
status = trading_system.get_system_status()
print(f"运行状态: {status['is_running']}")
print(f"策略数量: {status['strategies_count']}")
```

### 2. DataManager - 数据管理器

负责数据的获取、存储和管理。

```python
from myQuant.core.managers.data_manager import DataManager

# 创建数据管理器
data_manager = DataManager()
```

#### 方法

##### `get_price_data(symbol, start_date, end_date)`
获取价格数据。

**参数:**
- `symbol` (str): 股票代码
- `start_date` (str): 开始日期 (YYYY-MM-DD)
- `end_date` (str): 结束日期 (YYYY-MM-DD)

**返回:** 
- `pd.DataFrame`: 价格数据

**示例:**
```python
data = data_manager.get_price_data('000001.SZ', '2023-01-01', '2023-12-31')
print(f"获取到 {len(data)} 条数据")
print(data.head())
```

##### `validate_price_data(data)`
验证价格数据的有效性。

**参数:**
- `data` (pd.DataFrame): 价格数据

**返回:** 
- `bool`: 是否有效

**示例:**
```python
is_valid = data_manager.validate_price_data(data)
if not is_valid:
    print("数据验证失败")
```

##### `calculate_ma(prices, period)`
计算移动平均线。

**参数:**
- `prices` (pd.Series): 价格序列
- `period` (int): 周期

**返回:** 
- `pd.Series`: 移动平均线

**示例:**
```python
ma_20 = data_manager.calculate_ma(data['close'], 20)
print(f"20日均线: {ma_20.tail()}")
```

### 3. StrategyEngine - 策略引擎

管理和执行交易策略。

```python
from myQuant.core.strategy_engine import StrategyEngine

# 创建策略引擎
strategy_engine = StrategyEngine()
```

#### 方法

##### `add_strategy(strategy)`
添加策略到引擎。

**参数:**
- `strategy` (BaseStrategy): 策略对象

**返回:** 
- `str`: 策略ID

##### `remove_strategy(strategy_id)`
移除策略。

**参数:**
- `strategy_id` (str): 策略ID

**返回:** 
- `bool`: 是否成功

##### `get_strategies()`
获取所有策略。

**返回:** 
- `dict`: 策略字典

### 4. BacktestEngine - 回测引擎

执行历史回测。

```python
from myQuant.core.engines.backtest_engine import BacktestEngine

# 创建回测引擎
config = {
    'initial_capital': 1000000,
    'commission_rate': 0.0003,
    'start_date': '2023-01-01',
    'end_date': '2023-12-31'
}
backtest_engine = BacktestEngine(config)
```

#### 方法

##### `run_backtest(start_date, end_date)`
运行回测。

**参数:**
- `start_date` (str): 开始日期
- `end_date` (str): 结束日期

**返回:** 
- `dict`: 回测结果

**示例:**
```python
result = backtest_engine.run_backtest('2023-01-01', '2023-12-31')
print(f"最终价值: {result['final_value']}")
print(f"总收益率: {result['total_return']:.2%}")
```

##### `add_strategy(strategy)`
添加策略到回测引擎。

**参数:**
- `strategy` (BaseStrategy): 策略对象

##### `load_historical_data(data_manager, symbols, start_date, end_date)`
加载历史数据。

**参数:**
- `data_manager` (DataManager): 数据管理器
- `symbols` (list): 股票代码列表
- `start_date` (str): 开始日期
- `end_date` (str): 结束日期

### 5. PerformanceAnalyzer - 绩效分析器

分析投资组合绩效。

```python
from myQuant.core.analysis.performance_analyzer import PerformanceAnalyzer

# 创建绩效分析器
config = {
    'risk_free_rate': 0.03,
    'trading_days_per_year': 252
}
analyzer = PerformanceAnalyzer(config)
```

#### 方法

##### `analyze_portfolio(portfolio_values, benchmark_returns)`
分析投资组合绩效。

**参数:**
- `portfolio_values` (pd.Series): 投资组合价值序列
- `benchmark_returns` (pd.Series, 可选): 基准收益率

**返回:** 
- `dict`: 绩效分析结果

**示例:**
```python
import pandas as pd

# 模拟投资组合价值
portfolio_values = pd.Series([1000000, 1050000, 1020000, 1080000, 1100000])

# 分析绩效
results = analyzer.analyze_portfolio(portfolio_values)

print(f"总收益率: {results['returns']['total_return']:.2%}")
print(f"年化收益率: {results['returns']['annualized_return']:.2%}")
print(f"夏普比率: {results['returns']['sharpe_ratio']:.3f}")
print(f"最大回撤: {results['risk']['max_drawdown']['max_drawdown']:.2%}")
```

##### `calculate_sharpe_ratio(returns)`
计算夏普比率。

**参数:**
- `returns` (pd.Series): 收益率序列

**返回:** 
- `float`: 夏普比率

##### `calculate_max_drawdown(portfolio_values)`
计算最大回撤。

**参数:**
- `portfolio_values` (pd.Series): 投资组合价值序列

**返回:** 
- `dict`: 最大回撤信息

### 6. OrderManager - 订单管理器

管理交易订单。

```python
from myQuant.core.managers.order_manager import OrderManager

# 创建订单管理器
order_manager = OrderManager()
```

#### 方法

##### `create_order(symbol, side, quantity, order_type, price)`
创建订单。

**参数:**
- `symbol` (str): 股票代码
- `side` (OrderSide): 订单方向
- `quantity` (int): 数量
- `order_type` (OrderType): 订单类型
- `price` (float): 价格

**返回:** 
- `str`: 订单ID

**示例:**
```python
from myQuant.core.managers.order_manager import OrderSide, OrderType

order_id = order_manager.create_order(
    symbol='000001.SZ',
    side=OrderSide.BUY,
    quantity=1000,
    order_type=OrderType.MARKET,
    price=12.50
)
```

##### `cancel_order(order_id)`
取消订单。

**参数:**
- `order_id` (str): 订单ID

**返回:** 
- `dict`: 取消结果

### 7. RiskManager - 风险管理器

管理交易风险。

```python
from myQuant.core.managers.risk_manager import RiskManager

# 创建风险管理器
risk_manager = RiskManager()
```

#### 方法

##### `check_signal_risk(signal, positions)`
检查信号风险。

**参数:**
- `signal` (dict): 交易信号
- `positions` (dict): 当前持仓

**返回:** 
- `dict`: 风险检查结果

##### `calculate_position_size(signal, portfolio_value)`
计算仓位大小。

**参数:**
- `signal` (dict): 交易信号
- `portfolio_value` (float): 投资组合价值

**返回:** 
- `int`: 建议仓位大小

## 异常处理

系统提供统一的异常处理机制：

```python
from myQuant.core.exceptions import (
    DataException, 
    ConfigurationException,
    OrderException,
    StrategyException
)

try:
    # 业务逻辑
    data = data_manager.get_price_data('INVALID', '2023-01-01', '2023-12-31')
except DataException as e:
    print(f"数据异常: {e}")
except ConfigurationException as e:
    print(f"配置异常: {e}")
```

## 配置管理

使用统一的配置管理系统：

```python
from myQuant.infrastructure.config.settings import get_config

# 获取配置
config = get_config()

# 访问配置
print(f"初始资金: {config.trading.initial_capital}")
print(f"佣金率: {config.trading.commission_rate}")
print(f"风险无风险利率: {config.trading.risk_free_rate}")
```

## 日志系统

系统内置完整的日志功能：

```python
import logging

# 获取日志器
logger = logging.getLogger(__name__)

# 记录日志
logger.info("策略初始化完成")
logger.warning("数据质量问题")
logger.error("策略执行失败")
```

## 最佳实践

1. **使用依赖注入容器**：通过容器获取组件，确保正确的依赖关系
2. **异常处理**：始终使用try-catch处理可能的异常
3. **配置管理**：使用统一的配置系统，避免硬编码
4. **日志记录**：适当记录关键操作和异常
5. **数据验证**：处理数据前先验证有效性

## 相关文档

- [策略开发API](strategy-development.md)
- [数据管理API](data-management.md)
- [回测引擎API](backtest-engine.md)
- [绩效分析API](performance-analysis.md)