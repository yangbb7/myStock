# myStock 量化交易系统详细设计文档

## 1. 项目概述

### 1.1 项目背景
myStock是一个面向普通A股交易者的量化交易系统，基于现有的专业技术架构，旨在提供完整、用户友好的A股交易辅助功能。

### 1.2 设计目标
- 提供专业级量化交易能力
- 简化用户操作界面
- 确保系统安全性和稳定性
- 支持高频实时数据处理
- 提供完整的风险管理机制

### 1.3 技术特色
- **模块化单体架构**：避免微服务复杂性，保持高性能
- **前后端分离**：React + FastAPI架构
- **实时数据流**：WebSocket + 异步处理
- **强类型系统**：TypeScript前端 + Pydantic后端
- **现代化部署**：Docker容器化 + CI/CD

## 2. 系统架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Web前端 (React + TypeScript)              │
├─────────────────────────────────────────────────────────────┤
│                    API网关层 (FastAPI)                      │
├─────────────────────────────────────────────────────────────┤
│                      业务逻辑层                              │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │  行情模块    │   策略模块   │   订单模块   │   风险模块   │  │
│  ├─────────────┼─────────────┼─────────────┼─────────────┤  │
│  │  分析模块    │   组合模块   │   通知模块   │   报表模块   │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      数据访问层                              │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │  实时数据    │   历史数据   │   用户数据   │   系统数据   │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     基础设施层                               │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │  数据库      │   缓存       │   消息队列   │   文件存储   │  │
│  │  SQLite      │   Redis      │   RabbitMQ   │   本地存储   │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心模块设计

#### 2.2.1 行情数据模块
```python
# 核心数据结构
class MarketData(BaseModel):
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    turnover: Decimal
    
# 技术指标计算
class TechnicalIndicators:
    @staticmethod
    def calculate_ma(prices: List[Decimal], period: int) -> List[Decimal]
    
    @staticmethod
    def calculate_macd(prices: List[Decimal]) -> Dict[str, List[Decimal]]
    
    @staticmethod
    def calculate_rsi(prices: List[Decimal], period: int = 14) -> List[Decimal]
```

#### 2.2.2 交易订单模块
```python
class OrderManager:
    async def create_order(self, order_request: OrderRequest) -> OrderResponse
    async def cancel_order(self, order_id: str) -> CancelResponse
    async def get_order_status(self, order_id: str) -> OrderStatus
    async def get_order_history(self, filters: OrderFilters) -> List[Order]
    
class OrderValidation:
    def validate_order_params(self, order: OrderRequest) -> ValidationResult
    def check_risk_limits(self, order: OrderRequest) -> RiskCheckResult
    def verify_position_limits(self, order: OrderRequest) -> bool
```

#### 2.2.3 投资组合模块
```python
class PortfolioManager:
    def get_current_positions(self) -> Dict[str, Position]
    def calculate_portfolio_value(self) -> PortfolioValue
    def get_performance_metrics(self) -> PerformanceMetrics
    def generate_risk_report(self) -> RiskReport
    
class Position:
    symbol: str
    quantity: int
    average_price: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    percentage: float
```

## 3. 数据库设计

### 3.1 数据库架构

**主数据库**: SQLite (生产环境可扩展到PostgreSQL)
**缓存层**: Redis (可选)
**时序数据**: 使用分片存储策略

### 3.2 核心表结构

#### 3.2.1 用户相关表
```sql
-- 用户表
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- 用户配置表
CREATE TABLE user_configs (
    user_id INTEGER PRIMARY KEY,
    risk_tolerance DECIMAL(3,2) DEFAULT 0.02,
    max_position_size DECIMAL(3,2) DEFAULT 0.10,
    notification_settings JSON,
    trading_preferences JSON,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

#### 3.2.2 市场数据表
```sql
-- 股票基础信息表
CREATE TABLE stocks (
    symbol VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    sector VARCHAR(50),
    industry VARCHAR(100),
    market VARCHAR(10) NOT NULL, -- SH/SZ
    listing_date DATE,
    total_shares BIGINT,
    float_shares BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- K线数据表 (按时间周期分表)
CREATE TABLE kline_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    open_price DECIMAL(10,3) NOT NULL,
    high_price DECIMAL(10,3) NOT NULL,
    low_price DECIMAL(10,3) NOT NULL,
    close_price DECIMAL(10,3) NOT NULL,
    volume BIGINT DEFAULT 0,
    turnover DECIMAL(15,2) DEFAULT 0,
    UNIQUE(symbol, trade_date),
    INDEX idx_symbol_date (symbol, trade_date)
);

-- 实时行情表
CREATE TABLE real_time_quotes (
    symbol VARCHAR(20) PRIMARY KEY,
    current_price DECIMAL(10,3),
    change_amount DECIMAL(10,3),
    change_percent DECIMAL(5,2),
    volume BIGINT,
    turnover DECIMAL(15,2),
    bid_price_1 DECIMAL(10,3),
    bid_volume_1 BIGINT,
    ask_price_1 DECIMAL(10,3),
    ask_volume_1 BIGINT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 3.2.3 交易相关表
```sql
-- 订单表
CREATE TABLE orders (
    id VARCHAR(36) PRIMARY KEY, -- UUID
    user_id INTEGER NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    order_type VARCHAR(20) NOT NULL, -- MARKET/LIMIT/STOP
    side VARCHAR(10) NOT NULL, -- BUY/SELL
    quantity INTEGER NOT NULL,
    price DECIMAL(10,3),
    stop_price DECIMAL(10,3),
    filled_quantity INTEGER DEFAULT 0,
    average_fill_price DECIMAL(10,3),
    status VARCHAR(20) DEFAULT 'PENDING', -- PENDING/FILLED/CANCELLED/REJECTED
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    INDEX idx_user_status (user_id, status),
    INDEX idx_symbol_created (symbol, created_at)
);

-- 持仓表
CREATE TABLE positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    quantity INTEGER NOT NULL,
    average_price DECIMAL(10,3) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, symbol),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 交易记录表
CREATE TABLE transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id VARCHAR(36) NOT NULL,
    user_id INTEGER NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10,3) NOT NULL,
    commission DECIMAL(8,2) DEFAULT 0,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    INDEX idx_user_executed (user_id, executed_at),
    INDEX idx_symbol_executed (symbol, executed_at)
);
```

#### 3.2.4 策略和分析表
```sql
-- 策略配置表
CREATE TABLE strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    name VARCHAR(100) NOT NULL,
    type VARCHAR(50) NOT NULL,
    parameters JSON NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 提醒设置表
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    alert_type VARCHAR(50) NOT NULL, -- PRICE/VOLUME/INDICATOR
    condition_type VARCHAR(20) NOT NULL, -- ABOVE/BELOW/CROSS
    threshold_value DECIMAL(15,6),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    triggered_at TIMESTAMP NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    INDEX idx_user_active (user_id, is_active)
);

-- 风险管理表
CREATE TABLE risk_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    date DATE NOT NULL,
    portfolio_value DECIMAL(15,2),
    daily_pnl DECIMAL(15,2),
    max_drawdown DECIMAL(5,2),
    var_95 DECIMAL(15,2), -- Value at Risk
    beta DECIMAL(5,3),
    sharpe_ratio DECIMAL(5,3),
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, date),
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### 3.3 数据分片策略

#### 3.3.1 时间序列数据分片
```python
# 按月分片存储K线数据
class KlineShardManager:
    def get_table_name(self, symbol: str, date: datetime) -> str:
        return f"kline_{date.strftime('%Y%m')}"
    
    def create_monthly_table(self, year_month: str):
        table_name = f"kline_{year_month}"
        # 创建月度分表
        pass
```

#### 3.3.2 索引优化策略
```sql
-- 复合索引
CREATE INDEX idx_kline_symbol_date ON kline_daily(symbol, trade_date DESC);
CREATE INDEX idx_orders_user_status_created ON orders(user_id, status, created_at DESC);
CREATE INDEX idx_transactions_user_date ON transactions(user_id, executed_at DESC);

-- 部分索引 (SQLite 3.8.0+)
CREATE INDEX idx_active_alerts ON alerts(user_id, symbol) WHERE is_active = TRUE;
CREATE INDEX idx_pending_orders ON orders(user_id, symbol) WHERE status = 'PENDING';
```

## 4. API接口设计

### 4.1 RESTful API规范

**基础URL**: `http://localhost:8000/api/v1`
**认证方式**: JWT Token
**数据格式**: JSON
**状态码**: 标准HTTP状态码

### 4.2 核心API接口

#### 4.2.1 行情数据API
```python
# 获取K线数据
GET /api/v1/market/kline/{symbol}
Query Parameters:
- period: 1min|5min|15min|30min|1h|1d|1w|1M
- start_date: YYYY-MM-DD
- end_date: YYYY-MM-DD
- limit: 数量限制 (默认1000)

Response:
{
    "success": true,
    "data": {
        "symbol": "000001.SZ",
        "period": "1d",
        "records": [
            {
                "timestamp": "2024-01-01T09:30:00",
                "open": 12.50,
                "high": 12.80,
                "low": 12.30,
                "close": 12.75,
                "volume": 1000000,
                "turnover": 12650000.00
            }
        ]
    }
}

# 获取实时行情
GET /api/v1/market/realtime/{symbol}
Response:
{
    "success": true,
    "data": {
        "symbol": "000001.SZ",
        "current_price": 12.75,
        "change_amount": 0.25,
        "change_percent": 2.00,
        "volume": 15000000,
        "bid_ask": {
            "bid1": {"price": 12.74, "volume": 50000},
            "ask1": {"price": 12.75, "volume": 30000}
        },
        "timestamp": "2024-01-01T14:30:00"
    }
}

# 技术指标计算
POST /api/v1/market/indicators
Request Body:
{
    "symbol": "000001.SZ",
    "indicators": ["MA", "MACD", "RSI"],
    "parameters": {
        "MA": {"periods": [5, 10, 20]},
        "RSI": {"period": 14}
    },
    "period": "1d",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
}
```

#### 4.2.2 交易订单API
```python
# 创建订单
POST /api/v1/orders
Request Body:
{
    "symbol": "000001.SZ",
    "side": "BUY",
    "order_type": "LIMIT",
    "quantity": 1000,
    "price": 12.50,
    "time_in_force": "DAY"
}

Response:
{
    "success": true,
    "data": {
        "order_id": "ord_1234567890",
        "status": "PENDING",
        "created_at": "2024-01-01T09:30:00"
    }
}

# 查询订单状态
GET /api/v1/orders/{order_id}
Response:
{
    "success": true,
    "data": {
        "order_id": "ord_1234567890",
        "symbol": "000001.SZ",
        "side": "BUY",
        "quantity": 1000,
        "filled_quantity": 500,
        "average_fill_price": 12.48,
        "status": "PARTIALLY_FILLED",
        "created_at": "2024-01-01T09:30:00",
        "updated_at": "2024-01-01T09:35:00"
    }
}

# 撤销订单
DELETE /api/v1/orders/{order_id}
Response:
{
    "success": true,
    "message": "订单撤销成功"
}

# 查询订单历史
GET /api/v1/orders/history
Query Parameters:
- symbol: 股票代码
- status: 订单状态
- start_date: 开始日期
- end_date: 结束日期
- page: 页码
- size: 每页数量
```

#### 4.2.3 投资组合API
```python
# 获取投资组合概览
GET /api/v1/portfolio/summary
Response:
{
    "success": true,
    "data": {
        "total_value": 1250000.00,
        "cash_balance": 250000.00,
        "position_value": 1000000.00,
        "total_pnl": 250000.00,
        "total_return": 25.00,
        "positions_count": 8,
        "last_updated": "2024-01-01T15:00:00"
    }
}

# 获取持仓详情
GET /api/v1/portfolio/positions
Response:
{
    "success": true,
    "data": {
        "positions": [
            {
                "symbol": "000001.SZ",
                "name": "平安银行",
                "quantity": 10000,
                "average_price": 12.00,
                "current_price": 12.75,
                "market_value": 127500.00,
                "unrealized_pnl": 7500.00,
                "percentage": 10.20,
                "sector": "金融"
            }
        ],
        "total_value": 1000000.00
    }
}

# 获取交易记录
GET /api/v1/portfolio/transactions
Query Parameters:
- symbol: 股票代码
- start_date: 开始日期
- end_date: 结束日期
- page: 页码
- size: 每页数量

# 生成绩效报告
GET /api/v1/portfolio/performance
Query Parameters:
- period: 1d|1w|1m|3m|6m|1y|ytd|all
Response:
{
    "success": true,
    "data": {
        "returns": {
            "total_return": 25.00,
            "annualized_return": 18.50,
            "daily_returns": [...],
            "cumulative_returns": [...]
        },
        "risk_metrics": {
            "volatility": 15.20,
            "sharpe_ratio": 1.22,
            "max_drawdown": -8.50,
            "var_95": -25000.00
        },
        "benchmark_comparison": {
            "benchmark": "沪深300",
            "alpha": 3.20,
            "beta": 1.15,
            "tracking_error": 5.80
        }
    }
}
```

#### 4.2.4 风险管理API
```python
# 获取风险指标
GET /api/v1/risk/metrics
Response:
{
    "success": true,
    "data": {
        "portfolio_risk": {
            "var_95": -25000.00,
            "expected_shortfall": -35000.00,
            "beta": 1.15,
            "tracking_error": 5.80
        },
        "position_limits": {
            "max_position_size": 0.10,
            "current_max_position": 0.08,
            "concentration_risk": "MEDIUM"
        },
        "daily_limits": {
            "max_daily_loss": 50000.00,
            "current_daily_pnl": -12000.00,
            "remaining_risk_budget": 38000.00
        },
        "alerts": [
            {
                "type": "CONCENTRATION_RISK",
                "message": "金融板块仓位过重",
                "severity": "MEDIUM"
            }
        ]
    }
}

# 订单风险检查
POST /api/v1/risk/check-order
Request Body:
{
    "symbol": "000001.SZ",
    "side": "BUY",
    "quantity": 10000,
    "price": 12.50
}

Response:
{
    "success": true,
    "data": {
        "risk_level": "MEDIUM",
        "checks": [
            {
                "type": "POSITION_SIZE",
                "status": "PASS",
                "message": "仓位大小符合限制"
            },
            {
                "type": "CASH_AVAILABLE",
                "status": "WARNING",
                "message": "现金余额较低"
            }
        ],
        "recommendations": [
            "建议减少购买数量至8000股"
        ]
    }
}
```

#### 4.2.5 选股筛选API
```python
# 股票筛选
POST /api/v1/screening/filter
Request Body:
{
    "filters": {
        "fundamental": {
            "pe_ratio": {"min": 5, "max": 30},
            "pb_ratio": {"min": 0.5, "max": 5},
            "roe": {"min": 0.10},
            "market_cap": {"min": 1000000000}
        },
        "technical": {
            "price_change_1d": {"min": -0.05, "max": 0.10},
            "volume_ratio": {"min": 1.5},
            "ma_position": "ABOVE_MA20"
        },
        "sector": ["科技", "医药", "新能源"],
        "market": ["SZ", "SH"]
    },
    "sort_by": "pe_ratio",
    "order": "asc",
    "limit": 50
}

Response:
{
    "success": true,
    "data": {
        "results": [
            {
                "symbol": "000001.SZ",
                "name": "平安银行",
                "sector": "金融",
                "current_price": 12.75,
                "pe_ratio": 8.5,
                "pb_ratio": 0.85,
                "roe": 0.125,
                "market_cap": 247500000000,
                "score": 85.5
            }
        ],
        "total_count": 125,
        "filters_applied": {...}
    }
}
```

### 4.3 WebSocket实时数据API

#### 4.3.1 连接管理
```javascript
// 建立WebSocket连接
const ws = new WebSocket('ws://localhost:8000/ws');

// 认证
ws.send(JSON.stringify({
    type: 'auth',
    token: 'jwt_token_here'
}));

// 订阅行情数据
ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'market_data',
    symbols: ['000001.SZ', '000002.SZ']
}));
```

#### 4.3.2 数据推送格式
```javascript
// 实时行情推送
{
    "type": "market_data",
    "data": {
        "symbol": "000001.SZ",
        "price": 12.75,
        "change": 0.25,
        "change_percent": 2.00,
        "volume": 15000000,
        "timestamp": "2024-01-01T14:30:00"
    }
}

// 订单状态更新
{
    "type": "order_update",
    "data": {
        "order_id": "ord_1234567890",
        "status": "FILLED",
        "filled_quantity": 1000,
        "average_fill_price": 12.48
    }
}

// 风险提醒
{
    "type": "risk_alert",
    "data": {
        "alert_type": "POSITION_LIMIT",
        "severity": "HIGH",
        "message": "单一股票仓位超过限制",
        "symbol": "000001.SZ",
        "current_percentage": 12.5,
        "limit_percentage": 10.0
    }
}
```

## 5. 前端组件设计

### 5.1 组件架构

```
src/
├── components/           # 可复用组件
│   ├── Charts/          # 图表组件
│   │   ├── CandlestickChart.tsx
│   │   ├── TechnicalIndicators.tsx
│   │   └── RealTimeChart.tsx
│   ├── Orders/          # 订单相关组件
│   │   ├── OrderCreateForm.tsx
│   │   ├── OrderList.tsx
│   │   └── OrderStatusMonitor.tsx
│   ├── Portfolio/       # 投资组合组件
│   │   ├── PortfolioSummary.tsx
│   │   ├── PositionList.tsx
│   │   └── PerformanceChart.tsx
│   └── Common/          # 通用组件
│       ├── DataTable.tsx
│       ├── LoadingSpinner.tsx
│       └── ErrorBoundary.tsx
├── pages/               # 页面组件
│   ├── Dashboard/       # 仪表板
│   ├── Trading/         # 交易页面
│   ├── Portfolio/       # 投资组合
│   ├── Analysis/        # 分析工具
│   └── Settings/        # 设置页面
├── services/            # API服务
├── hooks/               # 自定义Hooks
├── stores/              # 状态管理
└── utils/               # 工具函数
```

### 5.2 核心组件实现

#### 5.2.1 实时K线图组件
```typescript
interface CandlestickChartProps {
  symbol: string;
  period: '1min' | '5min' | '15min' | '1d';
  height?: number;
  showVolume?: boolean;
  showIndicators?: string[];
}

const CandlestickChart: React.FC<CandlestickChartProps> = ({
  symbol,
  period,
  height = 400,
  showVolume = true,
  showIndicators = []
}) => {
  const [data, setData] = useState<KlineData[]>([]);
  const [loading, setLoading] = useState(true);
  
  // 实时数据订阅
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/market/${symbol}`);
    
    ws.onmessage = (event) => {
      const newData = JSON.parse(event.data);
      setData(prev => updateKlineData(prev, newData));
    };
    
    return () => ws.close();
  }, [symbol]);
  
  // ECharts配置
  const chartOptions = useMemo(() => ({
    grid: [
      { left: '10%', right: '8%', height: '50%' },
      { left: '10%', right: '8%', top: '63%', height: '16%' }
    ],
    xAxis: [
      { type: 'category', data: data.map(d => d.timestamp) },
      { type: 'category', data: data.map(d => d.timestamp) }
    ],
    yAxis: [
      { scale: true },
      { scale: true, gridIndex: 1 }
    ],
    series: [
      {
        name: 'K线',
        type: 'candlestick',
        data: data.map(d => [d.open, d.close, d.low, d.high]),
        itemStyle: {
          color: '#ef232a',
          color0: '#14b143',
          borderColor: '#ef232a',
          borderColor0: '#14b143'
        }
      },
      {
        name: '成交量',
        type: 'bar',
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: data.map(d => d.volume)
      }
    ]
  }), [data]);
  
  return (
    <div style={{ height }}>
      <ReactECharts option={chartOptions} style={{ height: '100%' }} />
    </div>
  );
};
```

#### 5.2.2 订单创建表单组件
```typescript
interface OrderFormProps {
  symbol?: string;
  onSubmit: (order: OrderRequest) => void;
  onCancel?: () => void;
}

const OrderCreateForm: React.FC<OrderFormProps> = ({
  symbol,
  onSubmit,
  onCancel
}) => {
  const [form] = Form.useForm();
  const [orderType, setOrderType] = useState<OrderType>('MARKET');
  const [estimatedValue, setEstimatedValue] = useState(0);
  
  // 表单验证规则
  const validationRules = {
    symbol: [{ required: true, message: '请选择股票代码' }],
    quantity: [
      { required: true, message: '请输入数量' },
      { type: 'number', min: 100, message: '最小购买100股' }
    ],
    price: orderType !== 'MARKET' ? [
      { required: true, message: '请输入价格' },
      { type: 'number', min: 0.01, message: '价格必须大于0' }
    ] : []
  };
  
  // 实时估算订单价值
  const handleFormChange = (changedValues: any, allValues: any) => {
    const { quantity, price } = allValues;
    if (quantity && price) {
      setEstimatedValue(quantity * price);
    }
  };
  
  // 风险检查
  const performRiskCheck = async (values: OrderRequest) => {
    try {
      const riskResult = await api.risk.checkOrder(values);
      if (riskResult.risk_level === 'HIGH') {
        Modal.confirm({
          title: '风险提示',
          content: '此订单风险较高，是否继续？',
          onOk: () => onSubmit(values)
        });
      } else {
        onSubmit(values);
      }
    } catch (error) {
      message.error('风险检查失败');
    }
  };
  
  return (
    <Card title="创建订单">
      <Form
        form={form}
        layout="vertical"
        onFinish={performRiskCheck}
        onValuesChange={handleFormChange}
        initialValues={{ symbol, order_type: 'MARKET' }}
      >
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item name="symbol" label="股票代码" rules={validationRules.symbol}>
              <StockSelect />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item name="side" label="买卖方向">
              <Radio.Group>
                <Radio value="BUY">买入</Radio>
                <Radio value="SELL">卖出</Radio>
              </Radio.Group>
            </Form.Item>
          </Col>
        </Row>
        
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item name="order_type" label="订单类型">
              <Select onChange={setOrderType}>
                <Option value="MARKET">市价单</Option>
                <Option value="LIMIT">限价单</Option>
                <Option value="STOP">止损单</Option>
              </Select>
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item name="quantity" label="数量" rules={validationRules.quantity}>
              <InputNumber style={{ width: '100%' }} step={100} />
            </Form.Item>
          </Col>
        </Row>
        
        {orderType !== 'MARKET' && (
          <Form.Item name="price" label="价格" rules={validationRules.price}>
            <InputNumber style={{ width: '100%' }} step={0.01} precision={2} />
          </Form.Item>
        )}
        
        {estimatedValue > 0 && (
          <Alert
            message={`预估金额: ¥${estimatedValue.toLocaleString()}`}
            type="info"
            style={{ marginBottom: 16 }}
          />
        )}
        
        <Form.Item>
          <Space>
            <Button type="primary" htmlType="submit">
              提交订单
            </Button>
            {onCancel && (
              <Button onClick={onCancel}>取消</Button>
            )}
          </Space>
        </Form.Item>
      </Form>
    </Card>
  );
};
```

#### 5.2.3 投资组合概览组件
```typescript
const PortfolioSummary: React.FC = () => {
  const { data: portfolio, loading } = useQuery({
    queryKey: ['portfolio-summary'],
    queryFn: () => api.portfolio.getSummary(),
    refetchInterval: 5000 // 5秒刷新一次
  });
  
  const { data: riskMetrics } = useQuery({
    queryKey: ['risk-metrics'],
    queryFn: () => api.risk.getMetrics()
  });
  
  if (loading) return <LoadingSkeleton />;
  
  return (
    <Row gutter={16}>
      <Col span={6}>
        <StatisticCard
          title="总资产"
          value={portfolio.total_value}
          precision={2}
          prefix="¥"
          valueStyle={{ color: '#3f8600' }}
        />
      </Col>
      <Col span={6}>
        <StatisticCard
          title="今日盈亏"
          value={portfolio.daily_pnl}
          precision={2}
          prefix="¥"
          valueStyle={{ 
            color: portfolio.daily_pnl >= 0 ? '#3f8600' : '#cf1322' 
          }}
        />
      </Col>
      <Col span={6}>
        <StatisticCard
          title="总收益率"
          value={portfolio.total_return}
          precision={2}
          suffix="%"
          valueStyle={{ 
            color: portfolio.total_return >= 0 ? '#3f8600' : '#cf1322' 
          }}
        />
      </Col>
      <Col span={6}>
        <StatisticCard
          title="持仓数量"
          value={portfolio.positions_count}
          suffix="只"
        />
      </Col>
      
      <Col span={24} style={{ marginTop: 16 }}>
        <Card title="资产分布">
          <PieChart
            data={[
              { name: '现金', value: portfolio.cash_balance },
              { name: '股票', value: portfolio.position_value }
            ]}
            height={200}
          />
        </Card>
      </Col>
      
      {riskMetrics && (
        <Col span={24} style={{ marginTop: 16 }}>
          <RiskMetricsCard metrics={riskMetrics} />
        </Col>
      )}
    </Row>
  );
};
```

### 5.3 状态管理

#### 5.3.1 使用Zustand进行状态管理
```typescript
// stores/portfolioStore.ts
interface PortfolioState {
  positions: Position[];
  totalValue: number;
  dailyPnl: number;
  isLoading: boolean;
  
  // Actions
  fetchPositions: () => Promise<void>;
  updatePosition: (symbol: string, updates: Partial<Position>) => void;
  addPosition: (position: Position) => void;
  removePosition: (symbol: string) => void;
}

export const usePortfolioStore = create<PortfolioState>((set, get) => ({
  positions: [],
  totalValue: 0,
  dailyPnl: 0,
  isLoading: false,
  
  fetchPositions: async () => {
    set({ isLoading: true });
    try {
      const data = await api.portfolio.getPositions();
      set({ 
        positions: data.positions,
        totalValue: data.total_value,
        isLoading: false 
      });
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },
  
  updatePosition: (symbol, updates) => {
    set(state => ({
      positions: state.positions.map(pos =>
        pos.symbol === symbol ? { ...pos, ...updates } : pos
      )
    }));
  },
  
  addPosition: (position) => {
    set(state => ({
      positions: [...state.positions, position]
    }));
  },
  
  removePosition: (symbol) => {
    set(state => ({
      positions: state.positions.filter(pos => pos.symbol !== symbol)
    }));
  }
}));
```

#### 5.3.2 实时数据状态管理
```typescript
// stores/marketDataStore.ts
interface MarketDataState {
  quotes: Record<string, RealTimeQuote>;
  subscriptions: Set<string>;
  connectionStatus: 'connected' | 'disconnected' | 'connecting';
  
  // Actions
  subscribe: (symbols: string[]) => void;
  unsubscribe: (symbols: string[]) => void;
  updateQuote: (symbol: string, quote: RealTimeQuote) => void;
}

export const useMarketDataStore = create<MarketDataState>((set, get) => ({
  quotes: {},
  subscriptions: new Set(),
  connectionStatus: 'disconnected',
  
  subscribe: (symbols) => {
    const { subscriptions } = get();
    const newSymbols = symbols.filter(s => !subscriptions.has(s));
    
    if (newSymbols.length > 0) {
      // 发送WebSocket订阅消息
      websocketManager.subscribe('market_data', newSymbols);
      
      set(state => ({
        subscriptions: new Set([...state.subscriptions, ...newSymbols])
      }));
    }
  },
  
  unsubscribe: (symbols) => {
    websocketManager.unsubscribe('market_data', symbols);
    
    set(state => {
      const newSubscriptions = new Set(state.subscriptions);
      symbols.forEach(s => newSubscriptions.delete(s));
      return { subscriptions: newSubscriptions };
    });
  },
  
  updateQuote: (symbol, quote) => {
    set(state => ({
      quotes: {
        ...state.quotes,
        [symbol]: quote
      }
    }));
  }
}));
```

## 6. 技术实施方案

### 6.1 开发环境搭建

#### 6.1.1 后端环境
```bash
# Python虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 数据库初始化
python -m myQuant.infrastructure.database.migration_manager init
python -m myQuant.infrastructure.database.migration_manager migrate

# 启动开发服务器
python main.py --api-server --debug
```

#### 6.1.2 前端环境
```bash
# 安装依赖
cd frontend
npm install

# 启动开发服务器
npm run dev

# 类型检查
npm run type-check

# 代码质量检查
npm run lint
npm run format
```

### 6.2 部署架构

#### 6.2.1 Docker容器化
```dockerfile
# 后端Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py", "--production"]
```

```dockerfile
# 前端Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
```

#### 6.2.2 Docker Compose配置
```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite:///data/myquant.db
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - redis
  
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - frontend
      - backend

volumes:
  redis_data:
```

### 6.3 CI/CD流程

#### 6.3.1 GitHub Actions配置
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-backend:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=myQuant --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  test-frontend:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
        cache-dependency-path: frontend/package-lock.json
    
    - name: Install dependencies
      run: |
        cd frontend
        npm ci
    
    - name: Run tests
      run: |
        cd frontend
        npm run test
        npm run type-check
        npm run lint

  build-and-deploy:
    needs: [test-backend, test-frontend]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build and push Docker images
      run: |
        docker build -t mystock-backend .
        docker build -t mystock-frontend ./frontend
        
        # Push to registry
        docker tag mystock-backend ${{ secrets.REGISTRY_URL }}/mystock-backend:latest
        docker tag mystock-frontend ${{ secrets.REGISTRY_URL }}/mystock-frontend:latest
        docker push ${{ secrets.REGISTRY_URL }}/mystock-backend:latest
        docker push ${{ secrets.REGISTRY_URL }}/mystock-frontend:latest
    
    - name: Deploy to production
      run: |
        # 部署脚本
        ./scripts/deploy.sh
```

### 6.4 监控和日志

#### 6.4.1 应用监控
```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import functools

# 定义指标
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')
ACTIVE_USERS = Gauge('active_users_total', 'Number of active users')
ORDER_COUNT = Counter('orders_total', 'Total orders created', ['status'])

def monitor_endpoint(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            REQUEST_COUNT.labels(method='GET', endpoint=func.__name__).inc()
            return result
        finally:
            REQUEST_LATENCY.observe(time.time() - start_time)
    return wrapper

# 启动监控服务器
def start_monitoring():
    start_http_server(8080)
```

#### 6.4.2 结构化日志
```python
# infrastructure/logging.py
import logging
import json
from datetime import datetime
from typing import Any, Dict

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(JsonFormatter())
        
        # 文件处理器
        file_handler = logging.FileHandler('logs/app.log')
        file_handler.setFormatter(JsonFormatter())
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        self._log('INFO', message, **kwargs)
    
    def error(self, message: str, error: Exception = None, **kwargs):
        if error:
            kwargs['error_type'] = type(error).__name__
            kwargs['error_message'] = str(error)
        self._log('ERROR', message, **kwargs)
    
    def _log(self, level: str, message: str, **kwargs):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            **kwargs
        }
        getattr(self.logger, level.lower())(json.dumps(log_data))

class JsonFormatter(logging.Formatter):
    def format(self, record):
        return record.getMessage()
```

### 6.5 性能优化策略

#### 6.5.1 后端性能优化
```python
# 数据库查询优化
class OptimizedPortfolioManager:
    @lru_cache(maxsize=128)
    def get_position_value(self, symbol: str, price: Decimal) -> Decimal:
        """缓存仓位价值计算"""
        position = self.positions.get(symbol)
        if not position:
            return Decimal('0')
        return position.quantity * price
    
    async def batch_update_prices(self, price_data: Dict[str, Decimal]):
        """批量更新价格，减少数据库访问"""
        async with self.db.transaction():
            for symbol, price in price_data.items():
                await self.update_position_price(symbol, price)
    
    def get_portfolio_summary_cached(self) -> Dict[str, Any]:
        """使用Redis缓存投资组合摘要"""
        cache_key = f"portfolio_summary_{self.user_id}"
        cached = self.redis.get(cache_key)
        
        if cached:
            return json.loads(cached)
        
        summary = self.calculate_portfolio_summary()
        self.redis.setex(cache_key, 60, json.dumps(summary))  # 缓存1分钟
        return summary

# 异步任务处理
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncTaskManager:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def calculate_indicators_async(self, data: pd.DataFrame) -> Dict[str, Any]:
        """异步计算技术指标"""
        loop = asyncio.get_event_loop()
        
        # CPU密集型任务使用线程池
        indicators = await loop.run_in_executor(
            self.executor,
            self._calculate_indicators_sync,
            data
        )
        
        return indicators
    
    def _calculate_indicators_sync(self, data: pd.DataFrame) -> Dict[str, Any]:
        """同步计算技术指标"""
        return {
            'ma_5': data['close'].rolling(5).mean(),
            'ma_20': data['close'].rolling(20).mean(),
            'rsi': calculate_rsi(data['close']),
            'macd': calculate_macd(data['close'])
        }
```

#### 6.5.2 前端性能优化
```typescript
// 虚拟化长列表
import { FixedSizeList as List } from 'react-window';

const VirtualizedPositionList: React.FC<{ positions: Position[] }> = ({ positions }) => {
  const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => (
    <div style={style}>
      <PositionItem position={positions[index]} />
    </div>
  );
  
  return (
    <List
      height={400}
      itemCount={positions.length}
      itemSize={60}
      width="100%"
    >
      {Row}
    </List>
  );
};

// 懒加载图表
const LazyChart = React.lazy(() => import('./CandlestickChart'));

const ChartContainer: React.FC = () => (
  <Suspense fallback={<ChartSkeleton />}>
    <LazyChart />
  </Suspense>
);

// 防抖搜索
const useDebounce = (value: string, delay: number) => {
  const [debouncedValue, setDebouncedValue] = useState(value);
  
  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);
    
    return () => clearTimeout(handler);
  }, [value, delay]);
  
  return debouncedValue;
};

// 内存优化的状态管理
const useOptimizedQuery = (queryKey: string[], queryFn: () => Promise<any>) => {
  return useQuery({
    queryKey,
    queryFn,
    staleTime: 30000, // 30秒内不重新请求
    cacheTime: 300000, // 5分钟缓存
    refetchOnWindowFocus: false,
    refetchOnReconnect: 'always'
  });
};
```

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "\u5206\u6790\u5f53\u524d\u9879\u76ee\u7ed3\u6784\u548c\u4ee3\u7801\u72b6\u6001", "status": "completed", "priority": "high"}, {"id": "2", "content": "\u521b\u5efa\u7cfb\u7edf\u67b6\u6784\u8bbe\u8ba1", "status": "completed", "priority": "high"}, {"id": "3", "content": "\u8bbe\u8ba1\u6570\u636e\u5e93\u6a21\u578b\u548cAPI\u63a5\u53e3", "status": "completed", "priority": "high"}, {"id": "4", "content": "\u5236\u5b9a\u524d\u7aef\u7ec4\u4ef6\u8bbe\u8ba1\u65b9\u6848", "status": "completed", "priority": "medium"}, {"id": "5", "content": "\u7f16\u5199\u8be6\u7ec6\u6280\u672f\u5b9e\u65bd\u65b9\u6848", "status": "completed", "priority": "medium"}, {"id": "6", "content": "\u751f\u6210\u5b8c\u6574\u8bbe\u8ba1\u6587\u6863", "status": "in_progress", "priority": "high"}]