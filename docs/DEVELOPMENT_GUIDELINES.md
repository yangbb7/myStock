# myQuant 项目开发规范指导文档

## 📖 目录

1. [项目概述](#项目概述)
2. [技术栈规范](#技术栈规范)
3. [开发环境配置](#开发环境配置)
4. [代码规范](#代码规范)
5. [架构设计原则](#架构设计原则)
6. [测试规范](#测试规范)
7. [版本管理](#版本管理)
8. [部署指南](#部署指南)
9. [性能优化](#性能优化)
10. [安全规范](#安全规范)

---

## 🎯 项目概述

myQuant 是一个高性能、低延迟的量化交易系统，支持：

- **回测引擎**：历史数据回测和策略验证
- **实时交易**：生产环境实时交易执行
- **风险管理**：实时风险监控和控制
- **数据管理**：多源数据接入和处理
- **Web界面**：实时监控和交易管理界面

---

## 🛠 技术栈规范

### 后端技术栈

| 技术 | 版本要求 | 用途 |
|------|----------|------|
| Python | >= 3.13 | 核心开发语言 |
| FastAPI | 0.116.1 | Web API框架 |
| SQLite/PostgreSQL | - | 数据存储 |
| Redis | >= 6.2.0 | 缓存和消息队列 |
| asyncio | Built-in | 异步编程 |
| pandas | 2.3.1 | 数据分析 |
| numpy | >= 1.24.0 | 数值计算 |

### 前端技术栈

| 技术 | 版本要求 | 用途 |
|------|----------|------|
| React | 19.1.0 | UI框架 |
| TypeScript | ~5.8.3 | 类型安全 |
| Vite | ^7.0.4 | 构建工具 |
| Ant Design | ^5.26.5 | UI组件库 |
| React Query | ^5.83.0 | 状态管理 |
| ECharts | ^5.6.0 | 图表库 |

### 开发工具

| 工具 | 用途 |
|------|------|
| pytest | Python测试框架 |
| ESLint + Prettier | 代码格式化 |
| Husky | Git钩子管理 |
| Docker | 容器化部署 |

---

## ⚙️ 开发环境配置

### 后端环境设置

```bash
# 1. 安装Python依赖
pip install -r requirements.txt

# 2. 设置环境变量
cp .env.example .env

# 3. 初始化数据库
python -m myQuant.infrastructure.database.migration_manager

# 4. 运行测试
pytest tests/

# 5. 启动开发服务器
python main.py --api-server
```

### 前端环境设置

```bash
# 1. 安装依赖
cd frontend
npm install

# 2. 启动开发服务器
npm run dev

# 3. 运行测试
npm test

# 4. 代码检查
npm run lint
npm run type-check
```

---

## 📝 代码规范

### Python 代码规范

#### 1. 代码风格

```python
# ✅ 正确示例
class PortfolioManager:
    """投资组合管理器
    
    管理投资组合的持仓、风险和绩效分析。
    """
    
    def __init__(self, initial_capital: float) -> None:
        self.initial_capital = initial_capital
        self._positions: Dict[str, Position] = {}
        self.logger = logging.getLogger(__name__)
    
    async def update_position(
        self, 
        symbol: str, 
        quantity: int, 
        price: float
    ) -> bool:
        """更新持仓信息
        
        Args:
            symbol: 股票代码
            quantity: 数量
            price: 价格
            
        Returns:
            bool: 更新是否成功
            
        Raises:
            ValueError: 当参数无效时
        """
        if quantity <= 0:
            raise ValueError("数量必须大于0")
        
        # 业务逻辑
        return True
```

#### 2. 命名规范

```python
# 类名：PascalCase
class RiskManager:
    pass

# 函数/变量名：snake_case
def calculate_portfolio_value() -> float:
    pass

# 常量：UPPER_SNAKE_CASE
MAX_POSITION_SIZE = 0.1

# 私有成员：下划线前缀
class Position:
    def __init__(self):
        self._private_data = {}
```

#### 3. 类型注解

```python
from typing import Dict, List, Optional, Union
from decimal import Decimal

# 必须使用类型注解
def process_orders(
    orders: List[Order],
    portfolio: Portfolio,
    risk_limits: Optional[Dict[str, float]] = None
) -> Tuple[List[Order], List[str]]:
    """处理订单列表"""
    executed_orders: List[Order] = []
    errors: List[str] = []
    
    for order in orders:
        try:
            result = execute_order(order)
            executed_orders.append(result)
        except Exception as e:
            errors.append(str(e))
    
    return executed_orders, errors
```

#### 4. 异常处理

```python
# ✅ 正确的异常处理
from myQuant.core.exceptions import TradingError, DataError

async def fetch_market_data(symbol: str) -> MarketData:
    """获取市场数据"""
    try:
        data = await data_provider.get_data(symbol)
        if not data:
            raise DataError(f"无法获取 {symbol} 的数据")
        return data
    except aiohttp.ClientError as e:
        self.logger.error(f"网络错误: {e}")
        raise DataError(f"网络请求失败: {e}") from e
    except Exception as e:
        self.logger.error(f"未知错误: {e}")
        raise TradingError(f"获取数据时发生错误: {e}") from e
```

### TypeScript 代码规范

#### 1. 组件结构

```typescript
// ✅ 正确的组件结构
interface PortfolioCardProps {
  portfolio: Portfolio;
  onUpdate?: (portfolio: Portfolio) => void;
  className?: string;
}

export const PortfolioCard: React.FC<PortfolioCardProps> = ({
  portfolio,
  onUpdate,
  className = '',
}) => {
  const [loading, setLoading] = useState(false);
  
  const handleUpdate = useCallback(async () => {
    setLoading(true);
    try {
      const updatedPortfolio = await updatePortfolio(portfolio.id);
      onUpdate?.(updatedPortfolio);
    } catch (error) {
      console.error('Failed to update portfolio:', error);
    } finally {
      setLoading(false);
    }
  }, [portfolio.id, onUpdate]);

  return (
    <Card className={`portfolio-card ${className}`}>
      {/* 组件内容 */}
    </Card>
  );
};
```

#### 2. 类型定义

```typescript
// types/portfolio.ts
export interface Portfolio {
  id: string;
  name: string;
  totalValue: number;
  positions: Position[];
  createdAt: Date;
  updatedAt: Date;
}

export interface Position {
  symbol: string;
  quantity: number;
  avgCost: number;
  currentPrice: number;
  unrealizedPnL: number;
}

// API响应类型
export interface ApiResponse<T> {
  data: T;
  status: 'success' | 'error';
  message?: string;
  timestamp: string;
}
```

#### 3. Hooks 使用

```typescript
// hooks/usePortfolio.ts
export const usePortfolio = (portfolioId: string) => {
  return useQuery({
    queryKey: ['portfolio', portfolioId],
    queryFn: () => portfolioService.getPortfolio(portfolioId),
    staleTime: 5 * 60 * 1000, // 5分钟
    cacheTime: 10 * 60 * 1000, // 10分钟
  });
};

// 在组件中使用
const PortfolioView: React.FC<{ id: string }> = ({ id }) => {
  const { data: portfolio, isLoading, error } = usePortfolio(id);
  
  if (isLoading) return <LoadingSpinner />;
  if (error) return <ErrorMessage error={error} />;
  if (!portfolio) return <NotFound />;
  
  return <PortfolioDetails portfolio={portfolio} />;
};
```

---

## 🏗 架构设计原则

### 1. 分层架构

```
myQuant/
├── interfaces/          # 接口层（API、CLI）
├── application/         # 应用服务层
├── core/               # 核心业务逻辑
│   ├── engines/        # 核心引擎
│   ├── managers/       # 业务管理器
│   ├── models/         # 数据模型
│   └── strategy/       # 策略框架
└── infrastructure/     # 基础设施层
    ├── database/       # 数据访问
    ├── monitoring/     # 监控
    └── config/         # 配置
```

### 2. 依赖注入

```python
# ✅ 使用依赖注入
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    # 配置
    config = providers.Configuration()
    
    # 数据库
    database = providers.Singleton(
        Database,
        url=config.database.url
    )
    
    # 服务
    portfolio_service = providers.Factory(
        PortfolioService,
        repository=providers.Factory(
            PortfolioRepository,
            database=database
        )
    )
```

### 3. 事件驱动架构

```python
# 事件系统
from myQuant.core.events import EventBus, Event

class OrderExecutedEvent(Event):
    def __init__(self, order: Order):
        self.order = order
        self.timestamp = datetime.now()

# 事件处理器
class PortfolioEventHandler:
    def __init__(self, portfolio_manager: PortfolioManager):
        self.portfolio_manager = portfolio_manager
    
    async def handle_order_executed(self, event: OrderExecutedEvent):
        await self.portfolio_manager.update_position(
            event.order.symbol,
            event.order.quantity,
            event.order.price
        )

# 注册事件处理器
event_bus = EventBus()
event_bus.subscribe(OrderExecutedEvent, handler.handle_order_executed)
```

---

## 🧪 测试规范

### Python 测试

#### 1. 测试结构

```python
# tests/unit/test_portfolio_manager.py
import pytest
from unittest.mock import Mock, patch
from myQuant.core.managers.portfolio_manager import PortfolioManager

class TestPortfolioManager:
    """投资组合管理器测试"""
    
    @pytest.fixture
    def portfolio_manager(self):
        """测试用的投资组合管理器"""
        return PortfolioManager(initial_capital=100000.0)
    
    @pytest.fixture
    def sample_position(self):
        """示例持仓"""
        return Position(
            symbol="000001.SZ",
            quantity=1000,
            avg_cost=10.0
        )
    
    def test_add_position_success(self, portfolio_manager, sample_position):
        """测试添加持仓成功"""
        # Arrange
        initial_positions = len(portfolio_manager.positions)
        
        # Act
        result = portfolio_manager.add_position(sample_position)
        
        # Assert
        assert result is True
        assert len(portfolio_manager.positions) == initial_positions + 1
        assert sample_position.symbol in portfolio_manager.positions
    
    @pytest.mark.asyncio
    async def test_update_position_async(self, portfolio_manager):
        """测试异步更新持仓"""
        # Arrange
        symbol = "000001.SZ"
        quantity = 1000
        price = 10.5
        
        # Act
        result = await portfolio_manager.update_position(symbol, quantity, price)
        
        # Assert
        assert result is True
        position = portfolio_manager.get_position(symbol)
        assert position.current_price == price
    
    @pytest.mark.parametrize("quantity,price,expected", [
        (1000, 10.0, 10000.0),
        (500, 20.0, 10000.0),
        (2000, 5.0, 10000.0),
    ])
    def test_calculate_position_value(self, portfolio_manager, quantity, price, expected):
        """参数化测试持仓价值计算"""
        position = Position("TEST", quantity, price)
        value = portfolio_manager.calculate_position_value(position)
        assert value == expected
```

#### 2. 集成测试

```python
# tests/integration/test_trading_system.py
import pytest
from myQuant.core.enhanced_trading_system import EnhancedTradingSystem, SystemConfig

@pytest.mark.integration
class TestTradingSystemIntegration:
    """交易系统集成测试"""
    
    @pytest.fixture
    def trading_system(self):
        """集成测试用的交易系统"""
        config = SystemConfig(
            initial_capital=100000.0,
            database_url="sqlite:///:memory:",
            enabled_modules=[
                SystemModule.DATA,
                SystemModule.STRATEGY,
                SystemModule.EXECUTION
            ]
        )
        return EnhancedTradingSystem(config)
    
    @pytest.mark.slow
    async def test_end_to_end_trading_flow(self, trading_system):
        """端到端交易流程测试"""
        # 1. 添加策略
        strategy = MAStrategy("test_strategy", ["000001.SZ"])
        trading_system.add_strategy(strategy)
        
        # 2. 启动系统
        await trading_system.start()
        
        # 3. 模拟市场数据
        market_data = MarketData(
            symbol="000001.SZ",
            price=10.0,
            timestamp=datetime.now()
        )
        await trading_system.process_market_data(market_data)
        
        # 4. 验证结果
        orders = trading_system.get_pending_orders()
        assert len(orders) > 0
        
        # 5. 清理
        await trading_system.shutdown()
```

### 前端测试

#### 1. 组件测试

```typescript
// __tests__/PortfolioCard.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PortfolioCard } from '../PortfolioCard';

const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: { retry: false },
    mutations: { retry: false },
  },
});

const mockPortfolio: Portfolio = {
  id: '1',
  name: 'Test Portfolio',
  totalValue: 100000,
  positions: [],
  createdAt: new Date(),
  updatedAt: new Date(),
};

describe('PortfolioCard', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = createTestQueryClient();
  });

  const renderWithProviders = (ui: React.ReactElement) => {
    return render(
      <QueryClientProvider client={queryClient}>
        {ui}
      </QueryClientProvider>
    );
  };

  it('renders portfolio information correctly', () => {
    renderWithProviders(<PortfolioCard portfolio={mockPortfolio} />);
    
    expect(screen.getByText('Test Portfolio')).toBeInTheDocument();
    expect(screen.getByText('¥100,000')).toBeInTheDocument();
  });

  it('calls onUpdate when refresh button is clicked', async () => {
    const mockOnUpdate = jest.fn();
    
    renderWithProviders(
      <PortfolioCard portfolio={mockPortfolio} onUpdate={mockOnUpdate} />
    );
    
    fireEvent.click(screen.getByRole('button', { name: /refresh/i }));
    
    await waitFor(() => {
      expect(mockOnUpdate).toHaveBeenCalledWith(expect.objectContaining({
        id: '1'
      }));
    });
  });
});
```

#### 2. 测试工具配置

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    globals: true,
  },
});

// src/test/setup.ts
import '@testing-library/jest-dom';
import { server } from './mocks/server';

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());
```

---

## 📋 版本管理

### Git 工作流

```bash
# 功能开发分支
git checkout -b feature/portfolio-rebalancing
git add .
git commit -m "feat: add portfolio rebalancing algorithm"
git push origin feature/portfolio-rebalancing

# 创建Pull Request
gh pr create --title "Add portfolio rebalancing" --body "Implements automatic portfolio rebalancing"
```

### 提交信息规范

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**类型说明：**
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档修改
- `style`: 代码格式修改
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建或辅助工具变动

**示例：**
```
feat(portfolio): add real-time position tracking

- Implement WebSocket connection for real-time data
- Add position update notifications
- Include PnL calculations

Closes #123
```

---

## 🚀 部署指南

### 开发环境

```bash
# 后端
python main.py --api-server

# 前端
cd frontend && npm run dev
```

### 生产环境

```bash
# 使用Docker部署
docker-compose -f docker-compose.prod.yml up -d

# 或者使用脚本
./deploy.sh production
```

### 环境变量

```bash
# .env.production
DATABASE_URL=postgresql://user:pass@host:5432/myquant
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
API_SECRET_KEY=your-secret-key-here
CORS_ORIGINS=https://myquant.example.com
```

---

## ⚡ 性能优化

### 后端优化

```python
# 1. 使用异步编程
async def batch_update_positions(positions: List[Position]):
    """批量更新持仓"""
    tasks = [update_position(pos) for pos in positions]
    await asyncio.gather(*tasks)

# 2. 数据库优化
from sqlalchemy.orm import selectinload

def get_portfolio_with_positions(portfolio_id: int):
    """预加载关联数据"""
    return session.query(Portfolio)\
        .options(selectinload(Portfolio.positions))\
        .filter(Portfolio.id == portfolio_id)\
        .first()

# 3. 缓存策略
from functools import lru_cache

@lru_cache(maxsize=1000)
def calculate_technical_indicator(symbol: str, period: int) -> float:
    """缓存技术指标计算结果"""
    return expensive_calculation(symbol, period)
```

### 前端优化

```typescript
// 1. 组件懒加载
const PortfolioPage = lazy(() => import('./pages/Portfolio/PortfolioPage'));

// 2. 数据虚拟化
import { FixedSizeList as List } from 'react-window';

const VirtualizedTable: React.FC<{ data: any[] }> = ({ data }) => (
  <List
    height={600}
    itemCount={data.length}
    itemSize={50}
    itemData={data}
  >
    {({ index, style, data }) => (
      <div style={style}>
        {/* 行内容 */}
      </div>
    )}
  </List>
);

// 3. 状态优化
const useOptimizedPortfolio = (portfolioId: string) => {
  return useQuery({
    queryKey: ['portfolio', portfolioId],
    queryFn: () => portfolioService.getPortfolio(portfolioId),
    staleTime: 5 * 60 * 1000,
    select: (data) => ({
      ...data,
      totalValue: data.positions.reduce((sum, pos) => sum + pos.value, 0)
    })
  });
};
```

---

## 🔒 安全规范

### 1. API安全

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    """验证JWT令牌"""
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/api/orders")
async def create_order(
    order: OrderRequest,
    user: dict = Depends(verify_token)
):
    """创建订单（需要认证）"""
    # 验证用户权限
    if not user.get("can_trade"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # 处理订单
    return await order_service.create_order(order, user_id=user["sub"])
```

### 2. 数据验证

```python
from pydantic import BaseModel, validator
from decimal import Decimal

class OrderRequest(BaseModel):
    symbol: str
    quantity: int
    price: Decimal
    order_type: str
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not re.match(r'^[0-9]{6}\.(SZ|SH)$', v):
            raise ValueError('Invalid symbol format')
        return v
    
    @validator('quantity')
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be positive')
        if v > 1000000:
            raise ValueError('Quantity too large')
        return v
    
    @validator('price')
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v
```

### 3. 前端安全

```typescript
// 输入验证和转义
import DOMPurify from 'dompurify';

const sanitizeInput = (input: string): string => {
  return DOMPurify.sanitize(input);
};

// API调用安全
const apiClient = axios.create({
  baseURL: process.env.REACT_APP_API_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器添加认证
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});
```

---

## 📊 监控和日志

### 日志配置

```python
# myQuant/infrastructure/monitoring/logging.py
import logging
import structlog

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """配置结构化日志"""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # 配置根日志器
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        filename=log_file,
        format="%(message)s"
    )

# 使用示例
logger = structlog.get_logger()
logger.info("Order executed", 
           symbol="000001.SZ", 
           quantity=1000, 
           price=10.5,
           user_id="user123")
```

### 性能监控

```python
# 监控装饰器
import time
from functools import wraps

def monitor_performance(func):
    """监控函数执行性能"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info("Function executed",
                       function=func.__name__,
                       execution_time=execution_time,
                       status="success")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Function failed",
                        function=func.__name__,
                        execution_time=execution_time,
                        error=str(e),
                        status="error")
            raise
    return wrapper

# 使用示例
@monitor_performance
async def execute_strategy(strategy: Strategy):
    """执行策略"""
    return await strategy.run()
```

---

## 🔧 开发工具配置

### VS Code 配置

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "typescript.preferences.quoteStyle": "single",
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  }
}
```

### Pre-commit 配置

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-eslint
    rev: v8.44.0
    hooks:
      - id: eslint
        files: \.(js|ts|tsx)$
        additional_dependencies:
          - eslint@8.44.0
          - typescript-eslint@5.61.0
```

---

## 📚 资源链接

### 官方文档
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [React 文档](https://react.dev/)
- [TypeScript 文档](https://www.typescriptlang.org/docs/)

### 最佳实践
- [Python 风格指南](https://pep8.org/)
- [React 最佳实践](https://react.dev/learn/thinking-in-react)
- [TypeScript 最佳实践](https://github.com/microsoft/TypeScript/wiki/Coding-guidelines)

### 项目特定资源
- [项目架构文档](./MONOLITH_ARCHITECTURE.md)
- [API 文档](./api/core-api.md)
- [快速开始指南](./user-guide/quick-start.md)

---

**版本**: v1.0.0  
**最后更新**: 2025-07-20  
**维护者**: myQuant 开发团队

---

*本文档会随着项目发展持续更新，请确保使用最新版本。*