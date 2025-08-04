# 前端Web交互界面设计文档

## 概述

本设计文档为myQuant模块化单体量化交易系统的前端Web界面提供详细的技术设计方案。该前端界面将与基于FastAPI的后端系统进行集成，提供现代化、响应式的用户体验，支持实时数据监控、策略管理、风险控制等核心功能。

### 设计目标

- **高性能**: 支持实时数据更新，响应时间 < 100ms
- **用户友好**: 直观的界面设计，符合量化交易用户习惯
- **可扩展**: 模块化架构，易于添加新功能
- **可靠性**: 错误处理和降级机制，确保系统稳定性
- **响应式**: 支持桌面和平板设备，适配不同屏幕尺寸

## 架构设计

### 技术栈选择

基于项目需求和现代前端开发最佳实践，选择以下技术栈：

#### 核心框架
- **React 18.2+**: 现代化的前端框架，支持并发特性和Suspense
- **TypeScript 5.0+**: 类型安全，提高代码质量和开发效率
- **Vite 4.0+**: 快速的构建工具，支持热重载和优化打包

#### UI组件库
- **Ant Design 5.0+**: 企业级UI组件库，提供丰富的组件和主题定制
- **@ant-design/pro-components**: 高级业务组件，适合后台管理系统

#### 状态管理
- **Zustand**: 轻量级状态管理库，简单易用
- **React Query (TanStack Query)**: 服务端状态管理，支持缓存和同步

#### 图表和可视化
- **Apache ECharts**: 强大的图表库，支持实时数据更新
- **@ant-design/charts**: 基于G2Plot的React图表组件

#### 实时通信
- **Socket.IO Client**: WebSocket客户端，支持实时数据推送
- **EventSource**: Server-Sent Events，用于服务器推送

#### 工具库
- **Axios**: HTTP客户端，用于API调用
- **Day.js**: 轻量级日期处理库
- **Lodash**: 实用工具函数库

### 系统架构

```
Frontend Architecture
├── src/
│   ├── components/          # 通用组件
│   │   ├── Layout/         # 布局组件
│   │   ├── Charts/         # 图表组件
│   │   ├── Forms/          # 表单组件
│   │   └── Common/         # 通用组件
│   ├── pages/              # 页面组件
│   │   ├── Dashboard/      # 仪表板
│   │   ├── Strategy/       # 策略管理
│   │   ├── Data/           # 数据监控
│   │   ├── Orders/         # 订单管理
│   │   ├── Portfolio/      # 投资组合
│   │   ├── Risk/           # 风险管理
│   │   ├── Backtest/       # 回测分析
│   │   └── System/         # 系统管理
│   ├── services/           # API服务
│   │   ├── api.ts          # API客户端
│   │   ├── websocket.ts    # WebSocket服务
│   │   └── types.ts        # 类型定义
│   ├── stores/             # 状态管理
│   │   ├── systemStore.ts  # 系统状态
│   │   ├── dataStore.ts    # 数据状态
│   │   └── userStore.ts    # 用户状态
│   ├── hooks/              # 自定义Hooks
│   │   ├── useApi.ts       # API调用Hook
│   │   ├── useWebSocket.ts # WebSocket Hook
│   │   └── useRealTime.ts  # 实时数据Hook
│   ├── utils/              # 工具函数
│   │   ├── format.ts       # 格式化函数
│   │   ├── constants.ts    # 常量定义
│   │   └── helpers.ts      # 辅助函数
│   └── styles/             # 样式文件
│       ├── globals.css     # 全局样式
│       ├── variables.css   # CSS变量
│       └── themes/         # 主题配置
```

## 组件设计

### 1. 布局组件 (Layout)

#### 主布局 (MainLayout)
```typescript
interface MainLayoutProps {
  children: React.ReactNode;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  return (
    <Layout className="main-layout">
      <Header />
      <Layout>
        <Sider />
        <Content>{children}</Content>
      </Layout>
      <Footer />
    </Layout>
  );
};
```

#### 导航菜单 (Navigation)
- 系统仪表板
- 策略管理
- 实时数据
- 订单管理
- 投资组合
- 风险监控
- 回测分析
- 系统管理

### 2. 仪表板组件 (Dashboard)

#### 系统状态卡片 (SystemStatusCard)
```typescript
interface SystemStatus {
  isRunning: boolean;
  uptime: number;
  modulesStatus: ModuleStatus[];
}

const SystemStatusCard: React.FC = () => {
  const { data: systemStatus } = useQuery({
    queryKey: ['systemHealth'],
    queryFn: () => api.getSystemHealth(),
    refetchInterval: 5000, // 5秒刷新
  });

  return (
    <Card title="系统状态">
      <Row gutter={16}>
        <Col span={8}>
          <Statistic
            title="运行状态"
            value={systemStatus?.isRunning ? '运行中' : '已停止'}
            valueStyle={{ color: systemStatus?.isRunning ? '#3f8600' : '#cf1322' }}
          />
        </Col>
        <Col span={8}>
          <Statistic
            title="运行时间"
            value={formatUptime(systemStatus?.uptime)}
          />
        </Col>
        <Col span={8}>
          <Statistic
            title="活跃模块"
            value={systemStatus?.modulesStatus?.filter(m => m.initialized).length}
            suffix={`/ ${systemStatus?.modulesStatus?.length}`}
          />
        </Col>
      </Row>
    </Card>
  );
};
```

#### 投资组合概览 (PortfolioOverview)
```typescript
const PortfolioOverview: React.FC = () => {
  const { data: portfolio } = useQuery({
    queryKey: ['portfolioSummary'],
    queryFn: () => api.getPortfolioSummary(),
    refetchInterval: 1000, // 1秒刷新
  });

  return (
    <Card title="投资组合概览">
      <Row gutter={16}>
        <Col span={6}>
          <Statistic
            title="总价值"
            value={portfolio?.totalValue}
            precision={2}
            prefix="¥"
            valueStyle={{ color: '#3f8600' }}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="现金余额"
            value={portfolio?.cashBalance}
            precision={2}
            prefix="¥"
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="持仓数量"
            value={portfolio?.positionsCount}
            suffix="只"
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="未实现盈亏"
            value={portfolio?.unrealizedPnl}
            precision={2}
            prefix="¥"
            valueStyle={{ 
              color: portfolio?.unrealizedPnl >= 0 ? '#3f8600' : '#cf1322' 
            }}
          />
        </Col>
      </Row>
    </Card>
  );
};
```

### 3. 策略管理组件 (Strategy)

#### 策略列表 (StrategyList)
```typescript
interface Strategy {
  name: string;
  signalsGenerated: number;
  successfulTrades: number;
  totalPnl: number;
}

const StrategyList: React.FC = () => {
  const { data: strategies } = useQuery({
    queryKey: ['strategyPerformance'],
    queryFn: () => api.getStrategyPerformance(),
    refetchInterval: 5000,
  });

  const columns = [
    {
      title: '策略名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '信号数量',
      dataIndex: 'signalsGenerated',
      key: 'signalsGenerated',
    },
    {
      title: '成功交易',
      dataIndex: 'successfulTrades',
      key: 'successfulTrades',
    },
    {
      title: '总盈亏',
      dataIndex: 'totalPnl',
      key: 'totalPnl',
      render: (value: number) => (
        <span style={{ color: value >= 0 ? '#3f8600' : '#cf1322' }}>
          ¥{value.toFixed(2)}
        </span>
      ),
    },
  ];

  return (
    <Card title="策略列表">
      <Table
        columns={columns}
        dataSource={Object.entries(strategies || {}).map(([name, data]) => ({
          key: name,
          name,
          ...data,
        }))}
        pagination={false}
      />
    </Card>
  );
};
```

#### 策略配置表单 (StrategyConfigForm)
```typescript
interface StrategyConfig {
  name: string;
  symbols: string[];
  initialCapital: number;
  riskTolerance: number;
  maxPositionSize: number;
  stopLoss?: number;
  takeProfit?: number;
  indicators: Record<string, any>;
}

const StrategyConfigForm: React.FC = () => {
  const [form] = Form.useForm();
  const addStrategyMutation = useMutation({
    mutationFn: (config: StrategyConfig) => api.addStrategy(config),
    onSuccess: () => {
      message.success('策略添加成功');
      form.resetFields();
    },
    onError: (error) => {
      message.error(`策略添加失败: ${error.message}`);
    },
  });

  const onFinish = (values: StrategyConfig) => {
    addStrategyMutation.mutate(values);
  };

  return (
    <Card title="添加策略">
      <Form
        form={form}
        layout="vertical"
        onFinish={onFinish}
      >
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item
              name="name"
              label="策略名称"
              rules={[{ required: true, message: '请输入策略名称' }]}
            >
              <Input placeholder="请输入策略名称" />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              name="symbols"
              label="股票代码"
              rules={[{ required: true, message: '请选择股票代码' }]}
            >
              <Select
                mode="multiple"
                placeholder="请选择股票代码"
                options={[
                  { label: '000001.SZ', value: '000001.SZ' },
                  { label: '000002.SZ', value: '000002.SZ' },
                  { label: '600000.SH', value: '600000.SH' },
                  { label: '600036.SH', value: '600036.SH' },
                ]}
              />
            </Form.Item>
          </Col>
        </Row>
        
        <Row gutter={16}>
          <Col span={8}>
            <Form.Item
              name="initialCapital"
              label="初始资金"
              rules={[{ required: true, message: '请输入初始资金' }]}
            >
              <InputNumber
                style={{ width: '100%' }}
                min={10000}
                max={10000000}
                step={10000}
                formatter={(value) => `¥ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                parser={(value) => value!.replace(/¥\s?|(,*)/g, '')}
              />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item
              name="riskTolerance"
              label="风险容忍度"
              rules={[{ required: true, message: '请输入风险容忍度' }]}
            >
              <InputNumber
                style={{ width: '100%' }}
                min={0.01}
                max={0.1}
                step={0.01}
                formatter={(value) => `${(value! * 100).toFixed(0)}%`}
                parser={(value) => value!.replace('%', '') / 100}
              />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item
              name="maxPositionSize"
              label="最大仓位比例"
              rules={[{ required: true, message: '请输入最大仓位比例' }]}
            >
              <InputNumber
                style={{ width: '100%' }}
                min={0.01}
                max={1}
                step={0.01}
                formatter={(value) => `${(value! * 100).toFixed(0)}%`}
                parser={(value) => value!.replace('%', '') / 100}
              />
            </Form.Item>
          </Col>
        </Row>

        <Form.Item>
          <Button
            type="primary"
            htmlType="submit"
            loading={addStrategyMutation.isPending}
          >
            添加策略
          </Button>
        </Form.Item>
      </Form>
    </Card>
  );
};
```

### 4. 实时数据组件 (Data)

#### 市场数据表格 (MarketDataTable)
```typescript
interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: string;
}

const MarketDataTable: React.FC = () => {
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  
  // WebSocket连接用于实时数据
  useWebSocket('/ws/market-data', {
    onMessage: (event) => {
      const data = JSON.parse(event.data);
      setMarketData(prev => {
        const index = prev.findIndex(item => item.symbol === data.symbol);
        if (index >= 0) {
          const newData = [...prev];
          newData[index] = data;
          return newData;
        } else {
          return [...prev, data];
        }
      });
    },
  });

  const columns = [
    {
      title: '股票代码',
      dataIndex: 'symbol',
      key: 'symbol',
    },
    {
      title: '最新价',
      dataIndex: 'price',
      key: 'price',
      render: (value: number) => `¥${value.toFixed(2)}`,
    },
    {
      title: '涨跌额',
      dataIndex: 'change',
      key: 'change',
      render: (value: number) => (
        <span style={{ color: value >= 0 ? '#3f8600' : '#cf1322' }}>
          {value >= 0 ? '+' : ''}¥{value.toFixed(2)}
        </span>
      ),
    },
    {
      title: '涨跌幅',
      dataIndex: 'changePercent',
      key: 'changePercent',
      render: (value: number) => (
        <span style={{ color: value >= 0 ? '#3f8600' : '#cf1322' }}>
          {value >= 0 ? '+' : ''}{(value * 100).toFixed(2)}%
        </span>
      ),
    },
    {
      title: '成交量',
      dataIndex: 'volume',
      key: 'volume',
      render: (value: number) => value.toLocaleString(),
    },
    {
      title: '更新时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (value: string) => dayjs(value).format('HH:mm:ss'),
    },
  ];

  return (
    <Card title="实时行情">
      <Table
        columns={columns}
        dataSource={marketData}
        rowKey="symbol"
        pagination={false}
        size="small"
      />
    </Card>
  );
};
```

#### K线图组件 (CandlestickChart)
```typescript
interface CandlestickData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

const CandlestickChart: React.FC<{ symbol: string }> = ({ symbol }) => {
  const { data: chartData } = useQuery({
    queryKey: ['marketData', symbol],
    queryFn: () => api.getMarketData(symbol, '1d'),
    enabled: !!symbol,
  });

  const option = useMemo(() => {
    if (!chartData?.records) return {};

    const data = chartData.records.map((item: any) => [
      item.timestamp,
      item.open,
      item.close,
      item.low,
      item.high,
      item.volume,
    ]);

    return {
      title: {
        text: `${symbol} K线图`,
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
        },
      },
      xAxis: {
        type: 'category',
        data: data.map((item: any) => dayjs(item[0]).format('MM-DD')),
      },
      yAxis: [
        {
          type: 'value',
          scale: true,
          splitArea: {
            show: true,
          },
        },
        {
          type: 'value',
          scale: true,
          splitArea: {
            show: true,
          },
        },
      ],
      series: [
        {
          name: 'K线',
          type: 'candlestick',
          data: data.map((item: any) => [item[1], item[2], item[3], item[4]]),
          itemStyle: {
            color: '#ef232a',
            color0: '#14b143',
            borderColor: '#ef232a',
            borderColor0: '#14b143',
          },
        },
        {
          name: '成交量',
          type: 'bar',
          yAxisIndex: 1,
          data: data.map((item: any) => item[5]),
          itemStyle: {
            color: '#7fbe9e',
          },
        },
      ],
    };
  }, [chartData, symbol]);

  return (
    <Card title="K线图">
      <ReactECharts
        option={option}
        style={{ height: '400px' }}
        notMerge={true}
        lazyUpdate={true}
      />
    </Card>
  );
};
```

## 数据模型

### API响应类型定义

```typescript
// 系统健康状态
interface SystemHealth {
  systemRunning: boolean;
  uptimeSeconds: number;
  modules: Record<string, ModuleHealth>;
  timestamp: string;
}

interface ModuleHealth {
  module: string;
  initialized: boolean;
  metrics: Record<string, any>;
  timestamp: string;
}

// 系统指标
interface SystemMetrics {
  system: {
    running: boolean;
    uptime: number;
    modulesCount: number;
  };
  data?: Record<string, any>;
  strategy?: Record<string, any>;
  execution?: Record<string, any>;
  risk?: Record<string, any>;
  portfolio?: Record<string, any>;
  analytics?: Record<string, any>;
  portfolioSummary?: PortfolioSummary;
  riskMetrics?: RiskMetrics;
  performance?: PerformanceMetrics;
}

// 投资组合摘要
interface PortfolioSummary {
  totalValue: number;
  cashBalance: number;
  positions: Record<string, Position>;
  unrealizedPnl: number;
  positionsCount: number;
}

interface Position {
  symbol: string;
  quantity: number;
  averagePrice: number;
  currentPrice: number;
  unrealizedPnl: number;
}

// 风险指标
interface RiskMetrics {
  dailyPnl: number;
  currentDrawdown: number;
  riskLimits: {
    maxPositionSize: number;
    maxDrawdownLimit: number;
    maxDailyLoss: number;
  };
  riskUtilization: {
    dailyLossRatio: number;
    drawdownRatio: number;
  };
}

// 策略性能
interface StrategyPerformance {
  [strategyName: string]: {
    signalsGenerated: number;
    successfulTrades: number;
    totalPnl: number;
  };
}

// 订单状态
interface OrderStatus {
  orderId: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  price?: number;
  status: 'PENDING' | 'FILLED' | 'REJECTED' | 'ERROR';
  timestamp: string;
  executedPrice?: number;
  executedQuantity?: number;
}

// 市场数据
interface MarketDataResponse {
  records: MarketDataRecord[];
  shape: [number, number];
  columns: string[];
}

interface MarketDataRecord {
  datetime: string;
  symbol: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  adjClose: number;
}
```

## 错误处理

### 全局错误处理

```typescript
// 错误边界组件
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
    // 可以发送错误报告到监控服务
  }

  render() {
    if (this.state.hasError) {
      return (
        <Result
          status="error"
          title="系统错误"
          subTitle="抱歉，系统遇到了一个错误。请刷新页面重试。"
          extra={
            <Button type="primary" onClick={() => window.location.reload()}>
              刷新页面
            </Button>
          }
        />
      );
    }

    return this.props.children;
  }
}

// API错误处理
const handleApiError = (error: any) => {
  if (error.response) {
    // 服务器响应错误
    const { status, data } = error.response;
    switch (status) {
      case 400:
        message.error('请求参数错误');
        break;
      case 401:
        message.error('未授权访问');
        break;
      case 403:
        message.error('访问被拒绝');
        break;
      case 404:
        message.error('请求的资源不存在');
        break;
      case 500:
        message.error('服务器内部错误');
        break;
      default:
        message.error(data?.message || '请求失败');
    }
  } else if (error.request) {
    // 网络错误
    message.error('网络连接失败，请检查网络设置');
  } else {
    // 其他错误
    message.error('发生未知错误');
  }
};
```

### 降级处理

```typescript
// 数据降级组件
const DataFallback: React.FC<{ error?: Error; retry?: () => void }> = ({ 
  error, 
  retry 
}) => {
  return (
    <Card>
      <Empty
        image={Empty.PRESENTED_IMAGE_SIMPLE}
        description={
          <span>
            数据加载失败
            <br />
            {error?.message}
          </span>
        }
      >
        {retry && (
          <Button type="primary" onClick={retry}>
            重试
          </Button>
        )}
      </Empty>
    </Card>
  );
};

// 使用Suspense和ErrorBoundary
const DataWithFallback: React.FC = () => {
  return (
    <ErrorBoundary>
      <Suspense fallback={<Spin size="large" />}>
        <MarketDataTable />
      </Suspense>
    </ErrorBoundary>
  );
};
```

## 测试策略

### 单元测试

```typescript
// 组件测试示例
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SystemStatusCard } from '../SystemStatusCard';
import * as api from '../../services/api';

// Mock API
jest.mock('../../services/api');
const mockApi = api as jest.Mocked<typeof api>;

describe('SystemStatusCard', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
      },
    });
  });

  it('should display system status correctly', async () => {
    const mockSystemHealth = {
      systemRunning: true,
      uptimeSeconds: 3600,
      modules: {
        data: { initialized: true },
        strategy: { initialized: true },
      },
    };

    mockApi.getSystemHealth.mockResolvedValue(mockSystemHealth);

    render(
      <QueryClientProvider client={queryClient}>
        <SystemStatusCard />
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByText('运行中')).toBeInTheDocument();
      expect(screen.getByText('2 / 2')).toBeInTheDocument();
    });
  });

  it('should handle API error gracefully', async () => {
    mockApi.getSystemHealth.mockRejectedValue(new Error('API Error'));

    render(
      <QueryClientProvider client={queryClient}>
        <SystemStatusCard />
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByText('数据加载失败')).toBeInTheDocument();
    });
  });
});
```

### 集成测试

```typescript
// E2E测试示例 (使用Playwright)
import { test, expect } from '@playwright/test';

test.describe('Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
  });

  test('should display system status', async ({ page }) => {
    await expect(page.locator('[data-testid="system-status"]')).toBeVisible();
    await expect(page.locator('text=运行中')).toBeVisible();
  });

  test('should update portfolio data in real-time', async ({ page }) => {
    const portfolioValue = page.locator('[data-testid="portfolio-value"]');
    const initialValue = await portfolioValue.textContent();
    
    // 等待数据更新
    await page.waitForTimeout(2000);
    
    const updatedValue = await portfolioValue.textContent();
    // 验证数据是否更新（在实际测试中可能需要模拟数据变化）
  });

  test('should handle system start/stop', async ({ page }) => {
    await page.click('[data-testid="stop-system-btn"]');
    await expect(page.locator('text=已停止')).toBeVisible();
    
    await page.click('[data-testid="start-system-btn"]');
    await expect(page.locator('text=运行中')).toBeVisible();
  });
});
```

## 部署配置

### 构建配置 (vite.config.ts)

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          antd: ['antd', '@ant-design/pro-components'],
          charts: ['echarts', '@ant-design/charts'],
        },
      },
    },
  },
});
```

### Docker配置

```dockerfile
# Dockerfile
FROM node:18-alpine as builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Nginx配置

```nginx
# nginx.conf
server {
    listen 80;
    server_name localhost;
    
    root /usr/share/nginx/html;
    index index.html;
    
    # 前端路由支持
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    # API代理
    location /api/ {
        proxy_pass http://backend:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    # WebSocket代理
    location /ws/ {
        proxy_pass http://backend:8000/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
    
    # 静态资源缓存
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

这个设计文档提供了完整的前端架构设计，包括技术栈选择、组件设计、数据模型、错误处理、测试策略和部署配置。设计充分考虑了与myQuant后端系统的集成，确保能够充分利用系统的API接口和实时数据能力。