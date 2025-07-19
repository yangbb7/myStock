# myQuant 模块化单体架构文档

## 📋 目录
- [架构概述](#架构概述)
- [设计理念](#设计理念)
- [模块设计](#模块设计)
- [API接口](#api接口)
- [配置管理](#配置管理)
- [部署指南](#部署指南)
- [性能优化](#性能优化)
- [故障处理](#故障处理)
- [最佳实践](#最佳实践)

## 🎯 架构概述

myQuant 模块化单体架构是对原有微服务架构的优化重构，旨在提供：

- **超低延迟**: 内存调用替代网络调用，延迟从毫秒级降至微秒级
- **强一致性**: 单进程内的强一致性事务，避免分布式事务复杂性
- **简化部署**: 单进程部署，无需复杂的容器编排
- **易于调试**: 集中式日志和调试，简化问题排查
- **高性能**: 优化的内存管理和批处理，提升整体性能

### 架构对比

| 特性 | 微服务架构 | 模块化单体 |
|------|------------|------------|
| 延迟 | 10-100ms | 0.1-1ms |
| 部署复杂度 | 高 | 低 |
| 调试难度 | 高 | 低 |
| 数据一致性 | 最终一致性 | 强一致性 |
| 适用场景 | 大团队、高并发 | 小团队、低延迟 |

## 🏗️ 设计理念

### 1. 模块化设计
- **清晰的边界**: 每个模块有明确的职责和接口
- **松耦合**: 模块间通过事件和接口通信
- **高内聚**: 相关功能集中在同一模块内

### 2. 单进程架构
- **统一资源管理**: 内存、线程、连接池统一管理
- **高效通信**: 内存调用替代网络调用
- **简化事务**: 本地事务保证数据一致性

### 3. 事件驱动
- **异步处理**: 非阻塞的事件处理机制
- **解耦合**: 模块间通过事件通信
- **可扩展**: 易于添加新的事件处理器

## 📦 模块设计

### 核心模块架构

```
EnhancedTradingSystem
├── DataModule          # 数据模块
├── StrategyModule      # 策略模块  
├── ExecutionModule     # 执行模块
├── RiskModule          # 风险模块
├── PortfolioModule     # 投资组合模块
└── AnalyticsModule     # 分析模块
```

### 1. 数据模块 (DataModule)

**职责**: 
- 市场数据获取和处理
- 实时数据流管理
- 数据缓存和存储

**主要功能**:
```python
async def get_market_data(symbol: str, period: str) -> Dict[str, Any]
async def process_realtime_data(tick_data: Dict[str, Any])
def subscribe_to_data(symbol: str, callback: Callable)
```

**特点**:
- 高性能数据缓存
- 实时数据订阅机制
- 多源数据整合

### 2. 策略模块 (StrategyModule)

**职责**:
- 策略管理和执行
- 信号生成和处理
- 策略性能统计

**主要功能**:
```python
async def add_strategy(strategy_name: str, strategy_config: Dict[str, Any])
async def process_market_data(tick_data: Dict[str, Any]) -> List[Dict[str, Any]]
def get_strategy_performance() -> Dict[str, Any]
```

**特点**:
- 多策略并行执行
- 策略性能监控
- 灵活的策略配置

### 3. 执行模块 (ExecutionModule)

**职责**:
- 订单创建和管理
- 交易执行和确认
- 执行性能优化

**主要功能**:
```python
async def create_order(signal: Dict[str, Any]) -> str
async def get_order_status(order_id: str) -> Dict[str, Any]
```

**特点**:
- 高并发订单处理
- 智能订单路由
- 执行性能统计

### 4. 风险模块 (RiskModule)

**职责**:
- 实时风险监控
- 风险限额管理
- 风险指标计算

**主要功能**:
```python
async def check_signal_risk(signal: Dict[str, Any], current_positions: Dict[str, Any]) -> Dict[str, Any]
async def update_pnl(pnl_change: float)
def get_risk_metrics() -> Dict[str, Any]
```

**特点**:
- 实时风险计算
- 多层风险控制
- 风险预警机制

### 5. 投资组合模块 (PortfolioModule)

**职责**:
- 持仓管理
- 投资组合价值计算
- 盈亏统计

**主要功能**:
```python
async def update_position(execution_result: Dict[str, Any])
def get_portfolio_summary() -> Dict[str, Any]
```

**特点**:
- 实时持仓更新
- 精确的盈亏计算
- 投资组合分析

### 6. 分析模块 (AnalyticsModule)

**职责**:
- 交易记录和分析
- 性能指标计算
- 报告生成

**主要功能**:
```python
async def record_trade(trade_data: Dict[str, Any])
def get_performance_report() -> Dict[str, Any]
```

**特点**:
- 全面的交易分析
- 实时性能监控
- 可视化报告

## 🌐 API接口

### RESTful API设计

基于FastAPI构建的高性能API接口，提供完整的系统访问能力。

#### 系统管理API
```
POST /system/start     # 启动系统
POST /system/stop      # 停止系统
GET  /health          # 健康检查
GET  /metrics         # 系统指标
```

#### 数据API
```
GET  /data/market/{symbol}     # 获取市场数据
POST /data/tick               # 处理tick数据
```

#### 策略API
```
POST /strategy/add            # 添加策略
GET  /strategy/performance    # 获取策略性能
```

#### 交易API
```
POST /order/create           # 创建订单
GET  /order/status/{id}      # 获取订单状态
```

#### 投资组合API
```
GET  /portfolio/summary      # 获取投资组合摘要
```

#### 风险API
```
GET  /risk/metrics          # 获取风险指标
```

#### 分析API
```
GET  /analytics/performance  # 获取性能报告
```

### API特点

- **高性能**: 基于FastAPI的异步处理
- **类型安全**: Pydantic模型验证
- **自动文档**: OpenAPI/Swagger文档
- **CORS支持**: 跨域请求支持

## ⚙️ 配置管理

### 配置层次结构

```
MonolithConfig
├── DatabaseConfig      # 数据库配置
├── TradingConfig      # 交易配置
├── PerformanceConfig  # 性能配置
├── LoggingConfig      # 日志配置
├── APIConfig          # API配置
└── MonitoringConfig   # 监控配置
```

### 配置加载顺序

1. **默认配置**: 内置默认值
2. **文件配置**: YAML/JSON配置文件
3. **环境变量**: 环境变量覆盖
4. **运行时覆盖**: 代码中的覆盖配置

### 配置文件示例

```yaml
# config/monolith_config.yaml
environment: "production"
debug: false

enabled_modules:
  - "data"
  - "strategy"
  - "execution"
  - "risk"
  - "portfolio"
  - "analytics"

trading:
  initial_capital: 1000000.0
  commission_rate: 0.0003
  max_position_size: 0.1

performance:
  max_concurrent_orders: 50
  order_timeout: 10.0
  enable_cache: true
```

### 环境变量配置

```bash
# 基础配置
export ENVIRONMENT=production
export DEBUG=false

# 交易配置
export TRADING_INITIAL_CAPITAL=1000000.0
export TRADING_COMMISSION_RATE=0.0003

# 性能配置
export PERF_MAX_CONCURRENT_ORDERS=50
export PERF_ORDER_TIMEOUT=10.0

# API配置
export API_HOST=0.0.0.0
export API_PORT=8000
```

## 🚀 部署指南

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 创建必要目录
mkdir -p logs data config
```

### 2. 配置文件

```bash
# 复制配置文件
cp config/monolith_config.yaml config/production.yaml

# 编辑配置
nano config/production.yaml
```

### 3. 启动系统

```bash
# 演示模式
python main_monolith.py demo

# API服务模式
python main_monolith.py api

# 使用配置文件
python main_monolith.py api --config config/production.yaml
```

### 4. Docker部署

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "main_monolith.py", "api"]
```

```bash
# 构建和运行
docker build -t myquant-monolith .
docker run -d -p 8000:8000 myquant-monolith
```

### 5. 系统服务

```bash
# 创建systemd服务
sudo nano /etc/systemd/system/myquant.service
```

```ini
[Unit]
Description=myQuant Monolith Trading System
After=network.target

[Service]
Type=simple
User=myquant
WorkingDirectory=/opt/myquant
ExecStart=/opt/myquant/venv/bin/python main_monolith.py api
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 启用服务
sudo systemctl enable myquant
sudo systemctl start myquant
sudo systemctl status myquant
```

## ⚡ 性能优化

### 1. 内存优化

- **对象池**: 复用频繁创建的对象
- **缓存机制**: 智能缓存热点数据
- **内存监控**: 实时监控内存使用

```python
# 对象池示例
from myQuant.core.optimizations import tick_pool, signal_pool

# 获取对象
tick = tick_pool.get_object()
signal = signal_pool.get_object()

# 使用后归还
tick_pool.return_object(tick)
signal_pool.return_object(signal)
```

### 2. 并发优化

- **异步处理**: 使用asyncio提高并发性
- **批处理**: 批量处理数据和订单
- **线程池**: 合理使用线程池

```python
# 异步处理示例
async def process_multiple_signals(signals):
    tasks = []
    for signal in signals:
        task = asyncio.create_task(process_signal(signal))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

### 3. 数据库优化

- **连接池**: 复用数据库连接
- **批量操作**: 批量插入和更新
- **索引优化**: 合理创建索引

```python
# 批量操作示例
async def batch_insert_trades(trades):
    async with database.transaction():
        await database.executemany(
            "INSERT INTO trades (...) VALUES (...)",
            trades
        )
```

### 4. 缓存策略

- **多级缓存**: 内存缓存 + Redis缓存
- **缓存预热**: 启动时预加载热点数据
- **缓存更新**: 智能缓存失效策略

```python
# 缓存使用示例
from myQuant.core.optimizations import price_cache

# 设置缓存
price_cache.set_price("AAPL", 150.0, volume=1000000)

# 获取缓存
price = price_cache.get_price("AAPL")
```

## 🔧 故障处理

### 1. 常见问题

#### 系统启动失败
```bash
# 检查日志
tail -f logs/monolith.log

# 检查配置
python -c "from myQuant.config.monolith_config import load_config; print(load_config().validate())"

# 检查端口占用
netstat -tlnp | grep 8000
```

#### 数据库连接失败
```bash
# 检查数据库文件
ls -la data/myquant.db

# 检查权限
chmod 644 data/myquant.db

# 重建数据库
python -c "from myQuant.infrastructure.database import create_tables; create_tables()"
```

#### API响应超时
```bash
# 检查系统资源
htop
df -h

# 检查API状态
curl http://localhost:8000/health

# 重启系统
sudo systemctl restart myquant
```

### 2. 监控指标

#### 系统指标
- CPU使用率
- 内存使用率
- 磁盘I/O
- 网络I/O

#### 业务指标
- 交易延迟
- 订单执行率
- 策略性能
- 风险指标

#### 告警设置
```python
# 性能告警
if cpu_usage > 80:
    send_alert("CPU使用率过高")

if memory_usage > 90:
    send_alert("内存使用率过高")

# 业务告警
if trade_latency > 100:  # 100ms
    send_alert("交易延迟过高")

if daily_loss > max_daily_loss:
    send_alert("日损失超限")
```

### 3. 备份恢复

#### 数据备份
```bash
# 数据库备份
cp data/myquant.db backup/myquant_$(date +%Y%m%d_%H%M%S).db

# 配置备份
cp -r config backup/config_$(date +%Y%m%d_%H%M%S)

# 日志备份
tar -czf backup/logs_$(date +%Y%m%d_%H%M%S).tar.gz logs/
```

#### 系统恢复
```bash
# 停止服务
sudo systemctl stop myquant

# 恢复数据
cp backup/myquant_20240101_120000.db data/myquant.db

# 恢复配置
cp -r backup/config_20240101_120000/* config/

# 启动服务
sudo systemctl start myquant
```

## 💡 最佳实践

### 1. 开发实践

#### 模块开发
- 遵循单一职责原则
- 使用依赖注入
- 编写完整的单元测试
- 使用类型提示

```python
# 良好的模块设计示例
class DataModule(ModuleInterface):
    def __init__(self, config: SystemConfig):
        super().__init__("data", config)
        self.data_manager = DataManager(config)
        
    async def initialize(self):
        await super().initialize()
        await self.data_manager.initialize()
        
    async def process_data(self, data: MarketData) -> ProcessResult:
        # 具体实现
        pass
```

#### 错误处理
```python
# 统一的错误处理
try:
    result = await process_data(data)
except ValidationError as e:
    logger.error(f"数据验证失败: {e}")
    raise
except ProcessingError as e:
    logger.error(f"数据处理失败: {e}")
    # 降级处理
    result = fallback_process(data)
```

### 2. 运维实践

#### 日志管理
```python
# 结构化日志
logger.info("订单创建", extra={
    "order_id": order_id,
    "symbol": symbol,
    "quantity": quantity,
    "price": price
})
```

#### 性能监控
```python
# 性能指标收集
@metrics.timer('order_processing_time')
async def process_order(order):
    # 处理逻辑
    pass
```

#### 配置管理
```python
# 配置热重载
async def reload_config():
    new_config = config_manager.reload_config()
    await system.update_config(new_config)
```

### 3. 安全实践

#### 访问控制
```python
# API访问控制
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not is_authenticated(request):
        raise HTTPException(401, "未认证")
    return await call_next(request)
```

#### 数据加密
```python
# 敏感数据加密
encrypted_data = encrypt_sensitive_data(data)
```

#### 输入验证
```python
# 严格的输入验证
class OrderRequest(BaseModel):
    symbol: str = Field(..., regex=r'^[A-Z]{1,6}$')
    quantity: int = Field(..., gt=0)
    price: float = Field(..., gt=0)
```

## 📈 未来规划

### 1. 短期目标
- 完善监控体系
- 优化性能指标
- 增强错误处理
- 丰富策略类型

### 2. 中期目标
- 支持多交易所
- 增加机器学习策略
- 实现高可用部署
- 完善回测系统

### 3. 长期目标
- 构建生态系统
- 开源核心组件
- 支持插件架构
- 国际化支持

---

## 📞 技术支持

- **文档**: [完整文档](docs/)
- **示例**: [代码示例](examples/)
- **FAQ**: [常见问题](docs/FAQ.md)
- **社区**: [讨论区](https://github.com/myquant/community)

---

**myQuant 模块化单体架构** - 为量化交易而生的高性能架构方案