# myStock 量化交易系统项目概述

## 项目目的
myStock 是一个现代化的量化交易系统，采用模块化单体架构，为个人散户投资者提供策略开发、回测、风险管理和性能分析功能。

## 技术栈
### 后端
- Python 3.13+
- FastAPI (Web框架)
- SQLAlchemy (ORM)
- Pandas/NumPy (数据处理)
- TA-Lib (技术指标)
- asyncio (异步处理)

### 前端
- React 19 + TypeScript
- Ant Design (UI组件库)
- ECharts/Recharts (数据可视化)
- Zustand (状态管理)
- Socket.io (WebSocket通信)

### 数据存储
- PostgreSQL (主数据库)
- Redis (缓存)
- InfluxDB (时序数据)

## 系统架构
- 模块化单体架构，采用六大核心模块：
  1. DataModule - 数据管理
  2. StrategyModule - 策略引擎
  3. ExecutionModule - 执行管理
  4. RiskModule - 风险控制
  5. PortfolioModule - 投资组合
  6. AnalyticsModule - 性能分析

## 主要特性
- 超低延迟内存通信
- 多数据源支持（Tushare、Yahoo、东财、券商API）
- 实时WebSocket数据推送
- 完整的回测系统
- 风险管理和监控
- 用户认证和权限管理