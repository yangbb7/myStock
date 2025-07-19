# myStock 量化交易系统文档

## 概述

myStock 是一个功能完整的量化交易系统，专为中国股票市场设计。该系统采用现代化的架构，包含数据管理、策略引擎、回测引擎、风险管理、绩效分析等核心模块。

## 文档结构

### 📚 用户指南
- [快速开始](user-guide/quick-start.md) - 5分钟快速上手
- [安装指南](user-guide/installation.md) - 详细的安装步骤
- [基础教程](user-guide/basic-tutorial.md) - 从零开始的完整教程
- [高级功能](user-guide/advanced-features.md) - 高级功能和配置

### 🔧 API文档
- [核心API](api/core-api.md) - 核心组件API参考
- [策略开发](api/strategy-development.md) - 策略开发API
- [数据管理](api/data-management.md) - 数据管理API
- [回测引擎](api/backtest-engine.md) - 回测引擎API
- [绩效分析](api/performance-analysis.md) - 绩效分析API

### 🎯 教程
- [第一个策略](tutorials/first-strategy.md) - 创建你的第一个交易策略
- [数据获取](tutorials/data-fetching.md) - 数据获取和管理
- [回测实战](tutorials/backtesting.md) - 完整的回测流程
- [风险管理](tutorials/risk-management.md) - 风险管理最佳实践

## 系统架构

```
myStock/
├── core/                    # 核心模块
│   ├── analysis/           # 绩效分析
│   ├── engines/            # 各种引擎
│   ├── managers/           # 管理器组件
│   ├── models/             # 数据模型
│   └── processors/         # 数据处理器
├── infrastructure/          # 基础设施
│   ├── config/             # 配置管理
│   ├── monitoring/         # 监控系统
│   └── container.py        # 依赖注入容器
├── application/            # 应用层
│   └── factory.py          # 应用工厂
└── interfaces/             # 接口层
    ├── api/                # API接口
    └── cli/                # 命令行接口
```

## 核心特性

### 🚀 高性能
- 异步数据处理引擎
- 多线程并发处理
- 内存和CPU优化

### 🔒 企业级
- 统一异常处理机制
- 完整的日志系统
- 依赖注入容器
- 配置管理系统

### 📊 丰富功能
- 多种技术指标
- 完整的绩效分析
- 风险管理模块
- 回测引擎

### 🛠️ 易于扩展
- 插件化架构
- 策略基类
- 自定义数据源
- 模块化设计

## 快速开始

### 1. 安装
```bash
pip install -r requirements.txt
```

### 2. 运行演示
```bash
# 交互式模式
python main.py

# 回测演示
python main.py --backtest

# 实时交易演示
python main.py --live
```

### 3. 创建第一个策略
```python
from myQuant.core.strategy_engine import BaseStrategy

class MyStrategy(BaseStrategy):
    def initialize(self):
        self.short_window = 5
        self.long_window = 20
        
    def on_bar(self, bar_data):
        # 策略逻辑
        pass
```

## 依赖注入

系统使用依赖注入容器管理组件生命周期：

```python
from myQuant.infrastructure.container import get_container

# 获取容器
container = get_container()

# 获取组件
data_manager = container.data_manager()
strategy_engine = container.strategy_engine()
trading_system = container.trading_system()
```

## 配置管理

使用统一的配置管理系统：

```python
from myQuant.infrastructure.config.settings import get_config

config = get_config()
print(f"初始资金: {config.trading.initial_capital}")
print(f"佣金率: {config.trading.commission_rate}")
```

## 异常处理

系统提供统一的异常处理机制：

```python
from myQuant.core.exceptions import DataException, ConfigurationException

try:
    # 业务逻辑
    pass
except DataException as e:
    print(f"数据异常: {e}")
except ConfigurationException as e:
    print(f"配置异常: {e}")
```

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 许可证

MIT License - 详见 LICENSE 文件

## 支持

- 📧 邮箱: support@mystock.com
- 📱 微信群: 扫码加入
- 🐛 问题反馈: GitHub Issues
- 📖 更多文档: https://docs.mystock.com