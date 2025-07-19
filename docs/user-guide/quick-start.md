# 快速开始指南

## 5分钟快速上手 myStock

### 1. 环境准备

确保你的系统满足以下要求：
- Python 3.8+
- 8GB+ RAM
- 稳定的网络连接

### 2. 安装 myStock

```bash
# 克隆项目
git clone https://github.com/your-org/myStock.git
cd myStock

# 安装依赖
pip install -r requirements.txt

# 验证安装
python main.py --version
```

### 3. 第一次运行

#### 交互式模式
```bash
python main.py
```

选择菜单选项：
1. 运行回测演示
2. 运行异步数据演示
3. 运行实时交易演示

#### 命令行模式
```bash
# 回测演示
python main.py --backtest

# 实时交易演示
python main.py --live

# 异步数据演示
python main.py --async-data
```

### 4. 创建你的第一个策略

创建文件 `my_first_strategy.py`：

```python
from myQuant.core.strategy_engine import BaseStrategy

class SimpleMAStrategy(BaseStrategy):
    """简单移动平均策略"""
    
    def initialize(self):
        """初始化策略"""
        self.short_window = 5
        self.long_window = 20
        self.prices = []
        
    def on_bar(self, bar_data):
        """处理每个bar数据"""
        symbol = bar_data.get('symbol')
        close_price = bar_data.get('close', 0)
        
        # 记录价格
        self.prices.append(close_price)
        
        # 保持价格列表长度
        if len(self.prices) > self.long_window:
            self.prices.pop(0)
            
        # 计算移动平均
        if len(self.prices) >= self.long_window:
            short_ma = sum(self.prices[-self.short_window:]) / self.short_window
            long_ma = sum(self.prices) / len(self.prices)
            
            # 生成信号
            if short_ma > long_ma:
                return [{
                    'timestamp': bar_data.get('datetime'),
                    'symbol': symbol,
                    'signal_type': 'BUY',
                    'price': close_price,
                    'quantity': 1000,
                    'strategy_name': self.name
                }]
            elif short_ma < long_ma:
                return [{
                    'timestamp': bar_data.get('datetime'),
                    'symbol': symbol,
                    'signal_type': 'SELL',
                    'price': close_price,
                    'quantity': 1000,
                    'strategy_name': self.name
                }]
        
        return []
```

### 5. 运行策略

```python
from myQuant.core.trading_system import TradingSystem
from myQuant import create_default_config
from my_first_strategy import SimpleMAStrategy

# 创建配置
config = create_default_config()

# 创建交易系统
trading_system = TradingSystem(config)

# 创建策略
strategy = SimpleMAStrategy(
    name="MyFirstStrategy",
    symbols=['000001.SZ', '000002.SZ'],
    params={'short_window': 5, 'long_window': 20}
)

# 添加策略
trading_system.add_strategy(strategy)

# 运行回测（需要历史数据）
# ...
```

### 6. 使用依赖注入

myStock 使用依赖注入容器管理组件：

```python
from myQuant.infrastructure.container import get_container

# 获取容器
container = get_container()

# 获取组件
data_manager = container.data_manager()
strategy_engine = container.strategy_engine()
performance_analyzer = container.performance_analyzer()

# 使用组件
data = data_manager.get_price_data('000001.SZ', '2023-01-01', '2023-12-31')
print(f"获取到 {len(data)} 条数据")
```

### 7. 配置管理

```python
from myQuant.infrastructure.config.settings import get_config

# 获取配置
config = get_config()

# 查看配置
print(f"初始资金: {config.trading.initial_capital}")
print(f"佣金率: {config.trading.commission_rate}")
print(f"最大回撤限制: {config.trading.max_drawdown_limit}")

# 修改配置
config.trading.initial_capital = 2000000
```

### 8. 绩效分析

```python
from myQuant.core.analysis.performance_analyzer import PerformanceAnalyzer
import pandas as pd

# 创建分析器
analyzer = PerformanceAnalyzer()

# 模拟投资组合价值序列
portfolio_values = pd.Series([1000000, 1050000, 1020000, 1080000, 1100000])

# 分析绩效
results = analyzer.analyze_portfolio(portfolio_values)

print(f"总收益率: {results['returns']['total_return']:.2%}")
print(f"年化收益率: {results['returns']['annualized_return']:.2%}")
print(f"夏普比率: {results['returns']['sharpe_ratio']:.3f}")
print(f"最大回撤: {results['risk']['max_drawdown']['max_drawdown']:.2%}")
```

### 9. 数据管理

```python
from myQuant.core.managers.data_manager import DataManager

# 创建数据管理器
data_manager = DataManager()

# 获取价格数据
data = data_manager.get_price_data('000001.SZ', '2023-01-01', '2023-12-31')

# 数据验证
is_valid = data_manager.validate_price_data(data)
print(f"数据有效性: {is_valid}")

# 计算技术指标
ma_20 = data_manager.calculate_ma(data['close'], 20)
print(f"20日均线: {ma_20.tail()}")
```

### 10. 常见问题

#### Q: 如何获取实时数据？
A: myStock 支持多种数据源，包括本地文件、API 接口等。参考 [数据获取教程](../tutorials/data-fetching.md)。

#### Q: 如何自定义指标？
A: 继承 `BaseStrategy` 类并实现自定义逻辑。参考 [策略开发文档](../api/strategy-development.md)。

#### Q: 如何调试策略？
A: 使用日志系统和调试工具。参考 [调试指南](../tutorials/debugging.md)。

#### Q: 系统支持哪些市场？
A: 目前主要支持中国A股市场，未来将扩展到其他市场。

### 下一步

- 阅读 [基础教程](basic-tutorial.md) 了解更多细节
- 查看 [API文档](../api/core-api.md) 了解所有可用功能
- 尝试 [示例策略](../tutorials/first-strategy.md) 

恭喜！你已经成功完成了 myStock 的快速入门。现在可以开始构建自己的量化交易策略了！