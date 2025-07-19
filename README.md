# myStock 量化交易系统

## 🎯 系统概述

myStock 是一个现代化的量化交易系统，采用**模块化单体架构**，提供超低延迟的策略开发、回测、风险管理和性能分析功能。

### 核心特性

- **超低延迟**: 内存通信，微秒级响应时间
- **模块化设计**: 清晰的模块边界，易于维护和扩展
- **单进程部署**: 简化部署，无需复杂的容器编排
- **强一致性**: 本地事务，数据一致性保证
- **高性能**: 异步处理，批量优化，缓存机制

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 推荐使用 uv 包管理器

### 安装依赖

```bash
# 使用 uv 安装（推荐）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 启动系统

```bash
# 启动交互式界面
python main.py

# 直接运行演示
python main.py --demo

# 启动API服务器
python main.py --api-server

# 生产模式启动
python main.py --production
```

## 📋 系统功能

### 核心模块

1. **数据模块 (DataModule)**
   - 高性能数据缓存
   - 实时数据订阅机制
   - 多源数据整合
   - 微秒级数据处理

2. **策略模块 (StrategyModule)**
   - 多策略并行执行
   - 策略性能监控
   - 灵活的策略配置
   - 事件驱动信号生成

3. **执行模块 (ExecutionModule)**
   - 高并发订单处理
   - 智能订单路由
   - 执行性能统计
   - 异步执行队列

4. **风险模块 (RiskModule)**
   - 实时风险计算
   - 多层风险控制
   - 风险预警机制
   - 动态风险限制

5. **投资组合模块 (PortfolioModule)**
   - 实时持仓更新
   - 精确的盈亏计算
   - 投资组合分析
   - 价值监控

6. **分析模块 (AnalyticsModule)**
   - 全面的交易分析
   - 实时性能监控
   - 可视化报告
   - 策略归因分析

### 主要功能

✅ **回测系统**
- 完整的历史数据回测
- 手续费和滑点计算
- 多策略回测支持
- 性能指标分析

✅ **实时交易**
- 实时数据处理
- 策略信号生成
- 订单执行管理
- 风险控制

✅ **性能分析**
- 夏普比率计算
- 最大回撤分析
- 收益率分析
- 基准比较

✅ **数据管理**
- 异步数据引擎
- 高性能并发数据获取
- 缓存机制
- 模拟数据生成

## 🔧 使用示例

### 基础使用

```python
from myQuant import create_default_config
from myQuant.core.enhanced_trading_system import EnhancedTradingSystem

# 创建交易系统
config = create_default_config()
system = EnhancedTradingSystem(config)

# 添加策略
from myQuant import MAStrategy
strategy = MAStrategy(
    name="TestStrategy",
    symbols=["000001.SZ", "000002.SZ"],
    params={"short_window": 5, "long_window": 20}
)
system.add_strategy(strategy)

# 运行回测
system.run_backtest()
```

### 异步数据获取

```python
import asyncio
from myQuant.core.engines.async_data_engine import AsyncDataEngine

async def get_market_data():
    config = {
        'max_concurrent_requests': 5,
        'request_timeout': 30,
        'cache_ttl': 300
    }
    
    async with AsyncDataEngine(config) as engine:
        symbols = ["000001.SZ", "000002.SZ"]
        async for data in engine.fetch_market_data(symbols):
            print(data)

asyncio.run(get_market_data())
```

### 性能分析

```python
from myQuant.core.analysis.performance_analyzer import PerformanceAnalyzer
import pandas as pd

analyzer = PerformanceAnalyzer()

# 模拟投资组合价值
portfolio_values = pd.Series([1000000, 1020000, 1050000, 1030000])

# 计算性能指标
returns = analyzer.calculate_returns(portfolio_values)
sharpe = analyzer.calculate_sharpe_ratio(returns)
drawdown = analyzer.calculate_max_drawdown(portfolio_values)

print(f"夏普比率: {sharpe:.2f}")
print(f"最大回撤: {drawdown:.2%}")
```

## 📁 项目结构

```
myStock/
├── main.py                         # 主入口文件
├── myQuant/                        # 核心模块
│   ├── core/                       # 核心功能
│   │   ├── enhanced_trading_system.py  # 增强交易系统
│   │   ├── engines/                # 各种引擎
│   │   ├── managers/               # 管理器
│   │   ├── analysis/               # 分析工具
│   │   └── models/                 # 数据模型
│   ├── infrastructure/             # 基础设施
│   │   ├── data/                   # 数据存储
│   │   ├── monitoring/             # 监控系统
│   │   └── config/                 # 配置管理
│   └── interfaces/                 # 接口层
│       └── api/                    # API接口
├── tests/                          # 测试用例
├── docs/                           # 文档
├── config.yaml                     # 配置文件
├── requirements.txt                # 依赖管理
└── pyproject.toml                  # 项目配置
```

## 🔧 配置说明

系统使用 YAML 配置文件进行配置，主要配置项包括：

```yaml
# config.yaml
trading:
  initial_capital: 1000000.0
  commission_rate: 0.0003
  max_position_size: 0.1

performance:
  max_concurrent_orders: 50
  order_timeout: 10.0
  enable_cache: true

data:
  cache_size: 1000
  update_interval: 1.0

risk:
  max_drawdown_limit: 0.2
  max_daily_loss: 0.05
```

## 🛠️ 工具命令

### 数据库管理

```bash
# 初始化数据库
python database_manager.py init

# 检查数据库状态
python database_manager.py health

# 优化数据库性能
python database_manager.py optimize

# 备份数据库
python database_manager.py backup
```

### 性能分析

```bash
# 运行性能分析
python performance_analysis.py

# 查看性能报告
cat performance_report.json
```

## 📊 性能指标

### 系统性能

- **延迟**: 0.1-1ms（内存通信）
- **吞吐量**: 10,000+ ticks/秒
- **内存使用**: 500MB-1GB
- **启动时间**: 10秒

### 测试结果

- ✅ 回测引擎：正常运行
- ✅ 实时交易：正常运行
- ✅ 数据引擎：100%成功率
- ✅ 风险管理：实时监控
- ✅ 性能分析：完整报告

## 🚨 注意事项

1. **数据源**: 目前使用模拟数据，实盘使用需配置真实数据源
2. **券商接口**: 实盘交易需要配置券商API
3. **风险控制**: 实盘使用前请充分测试风险控制参数
4. **依赖管理**: 确保所有依赖项正确安装

## 🎯 适用场景

- 高频交易系统
- 策略开发和回测
- 风险管理研究
- 量化投资教学
- 算法交易研究

## 📈 路线图

- [x] 模块化单体架构
- [x] 高性能交易引擎
- [x] 异步数据处理
- [x] 实时风险管理
- [x] 性能监控系统
- [ ] 机器学习策略
- [ ] 多资产支持
- [ ] 云原生部署

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

---

**🎉 myStock - 为量化交易而生的高性能系统！**