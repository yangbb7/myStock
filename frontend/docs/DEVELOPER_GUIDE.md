# myQuant Frontend 开发者指南

## 目录

1. [开发环境设置](#开发环境设置)
2. [项目架构](#项目架构)
3. [开发规范](#开发规范)
4. [API集成](#api集成)
5. [组件开发](#组件开发)
6. [状态管理](#状态管理)
7. [测试指南](#测试指南)
8. [构建和部署](#构建和部署)
9. [故障排除](#故障排除)

## 开发环境设置

### 系统要求

- **Node.js**: 18.0.0 或更高版本
- **npm**: 9.0.0 或更高版本
- **Git**: 2.30.0 或更高版本

### 环境安装

```bash
# 克隆项目
git clone <repository-url>
cd frontend

# 安装依赖
npm install

# 复制环境变量文件
cp .env.example .env

# 启动开发服务器
npm run dev
```

### 开发工具配置

#### VS Code 推荐插件

```json
{
  "recommendations": [
    "esbenp.prettier-vscode",
    "dbaeumer.vscode-eslint",
    "bradlc.vscode-tailwindcss",
    "ms-vscode.vscode-typescript-next",
    "formulahendry.auto-rename-tag",
    "christian-kohler.path-intellisense"
  ]
}
```

#### VS Code 设置

```json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "typescript.preferences.importModuleSpecifier": "relative"
}
```

## 项目架构

### 目录结构

```
src/
├── components/              # 可复用组件
│   ├── Layout/             # 布局组件
│   │   ├── MainLayout.tsx
│   │   ├── Header.tsx
│   │   ├── Sidebar.tsx
│   │   └── Footer.tsx
│   ├── Charts/             # 图表组件
│   │   ├── CandlestickChart.tsx
│   │   ├── LineChart.tsx
│   │   └── PerformanceChart.tsx
│   ├── Forms/              # 表单组件
│   │   ├── StrategyForm.tsx
│   │   ├── OrderForm.tsx
│   │   └── ConfigForm.tsx
│   └── Common/             # 通用组件
│       ├── LoadingSpinner.tsx
│       ├── ErrorBoundary.tsx
│       └── DataTable.tsx
├── pages/                  # 页面组件
│   ├── Dashboard/          # 仪表板页面
│   ├── Strategy/           # 策略管理页面
│   ├── Data/               # 数据监控页面
│   ├── Orders/             # 订单管理页面
│   ├── Portfolio/          # 投资组合页面
│   ├── Risk/               # 风险管理页面
│   ├── Backtest/           # 回测分析页面
│   └── System/             # 系统管理页面
├── services/               # 服务层
│   ├── api/                # API服务
│   │   ├── client.ts       # HTTP客户端配置
│   │   ├── endpoints.ts    # API端点定义
│   │   ├── types.ts        # API类型定义
│   │   └── hooks.ts        # API Hooks
│   ├── websocket/          # WebSocket服务
│   │   ├── client.ts       # WebSocket客户端
│   │   ├── handlers.ts     # 消息处理器
│   │   └── types.ts        # WebSocket类型
│   └── storage/            # 本地存储服务
│       ├── localStorage.ts
│       └── sessionStorage.ts
├── stores/                 # 状态管理
│   ├── systemStore.ts      # 系统状态
│   ├── dataStore.ts        # 数据状态
│   ├── strategyStore.ts    # 策略状态
│   ├── orderStore.ts       # 订单状态
│   ├── portfolioStore.ts   # 投资组合状态
│   └── userStore.ts        # 用户状态
├── hooks/                  # 自定义Hooks
│   ├── useApi.ts           # API调用Hook
│   ├── useWebSocket.ts     # WebSocket Hook
│   ├── useRealTime.ts      # 实时数据Hook
│   ├── useLocalStorage.ts  # 本地存储Hook
│   └── useDebounce.ts      # 防抖Hook
├── utils/                  # 工具函数
│   ├── constants.ts        # 常量定义
│   ├── format.ts           # 格式化函数
│   ├── validation.ts       # 验证函数
│   ├── helpers.ts          # 辅助函数
│   └── types.ts            # 通用类型定义
├── styles/                 # 样式文件
│   ├── globals.css         # 全局样式
│   ├── variables.css       # CSS变量
│   ├── components.css      # 组件样式
│   └── themes/             # 主题配置
│       ├── light.css
│       └── dark.css
└── assets/                 # 静态资源
    ├── images/
    ├── icons/
    └── fonts/
```

### 技术栈

#### 核心框架
- **React 18**: 前端框架，支持并发特性
- **TypeScript 5**: 类型安全的JavaScript
- **Vite 4**: 快速构建工具

#### UI组件库
- **Ant Design 5**: 企业级UI组件库
- **@ant-design/pro-components**: 高级业务组件
- **@ant-design/charts**: 图表组件

#### 状态管理
- **Zustand**: 轻量级状态管理
- **React Query**: 服务端状态管理

#### 图表和可视化
- **Apache ECharts**: 数据可视化图表库
- **echarts-for-react**: ECharts的React封装

#### 工具库
- **Axios**: HTTP客户端
- **Day.js**: 日期处理库
- **Lodash**: 实用工具函数库
- **Socket.IO**: 实时通信

## 开发规范

### 代码风格

项目使用ESLint和Prettier进行代码规范化：

```bash
# 检查代码风格
npm run lint

# 自动修复代码风格问题
npm run lint:fix

# 格式化代码
npm run format

# 检查代码格式
npm run format:check
```

### 命名规范

#### 文件命名
- **组件文件**: PascalCase，如 `UserProfile.tsx`
- **Hook文件**: camelCase，以use开头，如 `useUserData.ts`
- **工具文件**: camelCase，如 `formatUtils.ts`
- **常量文件**: camelCase，如 `apiConstants.ts`

#### 变量命名
- **组件名**: PascalCase，如 `UserProfile`
- **函数名**: camelCase，如 `getUserData`
- **变量名**: camelCase，如 `userData`
- **常量名**: UPPER_SNAKE_CASE，如 `API_BASE_URL`

#### 类型命名
- **接口**: PascalCase，以I开头，如 `IUserData`
- **类型别名**: PascalCase，如 `UserStatus`
- **枚举**: PascalCase，如 `OrderStatus`

### TypeScript规范

#### 类型定义

```typescript
// 接口定义
interface IApiResponse<T> {
  data: T;
  message: string;
  success: boolean;
  timestamp: string;
}

// 类型别名
type OrderStatus = 'PENDING' | 'FILLED' | 'REJECTED' | 'ERROR';

// 枚举定义
enum SystemModule {
  DATA = 'data',
  STRATEGY = 'strategy',
  EXECUTION = 'execution',
  RISK = 'risk',
  PORTFOLIO = 'portfolio',
  ANALYTICS = 'analytics'
}
```

#### 组件Props类型

```typescript
interface IButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  loading?: boolean;
  type?: 'primary' | 'secondary' | 'danger';
  size?: 'small' | 'medium' | 'large';
}

const Button: React.FC<IButtonProps> = ({
  children,
  onClick,
  disabled = false,
  loading = false,
  type = 'primary',
  size = 'medium'
}) => {
  // 组件实现
};
```

### Git提交规范

使用Conventional Commits规范：

```bash
# 功能添加
git commit -m "feat: add user authentication"

# Bug修复
git commit -m "fix: resolve data loading issue"

# 文档更新
git commit -m "docs: update API documentation"

# 样式修改
git commit -m "style: improve button styling"

# 重构代码
git commit -m "refactor: optimize data fetching logic"

# 性能优化
git commit -m "perf: improve chart rendering performance"

# 测试添加
git commit -m "test: add unit tests for user service"
```

## API集成

### API客户端配置

```typescript
// src/services/api/client.ts
import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: import.meta.env.VITE_API_BASE_URL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // 请求拦截器
    this.client.interceptors.request.use(
      (config) => {
        // 添加认证token
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // 响应拦截器
    this.client.interceptors.response.use(
      (response) => response.data,
      (error) => {
        this.handleApiError(error);
        return Promise.reject(error);
      }
    );
  }

  private handleApiError(error: any) {
    if (error.response?.status === 401) {
      // 处理未授权错误
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
  }

  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    return this.client.get(url, config);
  }

  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    return this.client.post(url, data, config);
  }

  async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    return this.client.put(url, data, config);
  }

  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    return this.client.delete(url, config);
  }
}

export const apiClient = new ApiClient();
```

### API服务定义

```typescript
// src/services/api/endpoints.ts
import { apiClient } from './client';
import { 
  ISystemHealth, 
  ISystemMetrics, 
  IStrategyPerformance,
  IPortfolioSummary,
  IRiskMetrics 
} from './types';

export const systemApi = {
  getHealth: (): Promise<ISystemHealth> => 
    apiClient.get('/health'),
  
  getMetrics: (): Promise<ISystemMetrics> => 
    apiClient.get('/metrics'),
  
  startSystem: (): Promise<void> => 
    apiClient.post('/system/start'),
  
  stopSystem: (): Promise<void> => 
    apiClient.post('/system/stop'),
};

export const strategyApi = {
  getPerformance: (): Promise<IStrategyPerformance> => 
    apiClient.get('/strategy/performance'),
  
  addStrategy: (config: IStrategyConfig): Promise<void> => 
    apiClient.post('/strategy/add', config),
};

export const portfolioApi = {
  getSummary: (): Promise<IPortfolioSummary> => 
    apiClient.get('/portfolio/summary'),
};

export const riskApi = {
  getMetrics: (): Promise<IRiskMetrics> => 
    apiClient.get('/risk/metrics'),
};
```

### React Query集成

```typescript
// src/services/api/hooks.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { systemApi, strategyApi, portfolioApi, riskApi } from './endpoints';

// 系统健康状态查询
export const useSystemHealth = () => {
  return useQuery({
    queryKey: ['system', 'health'],
    queryFn: systemApi.getHealth,
    refetchInterval: 5000, // 5秒刷新
    staleTime: 1000, // 1秒内认为数据是新鲜的
  });
};

// 系统指标查询
export const useSystemMetrics = () => {
  return useQuery({
    queryKey: ['system', 'metrics'],
    queryFn: systemApi.getMetrics,
    refetchInterval: 1000, // 1秒刷新
  });
};

// 策略性能查询
export const useStrategyPerformance = () => {
  return useQuery({
    queryKey: ['strategy', 'performance'],
    queryFn: strategyApi.getPerformance,
    refetchInterval: 5000,
  });
};

// 添加策略变更
export const useAddStrategy = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: strategyApi.addStrategy,
    onSuccess: () => {
      // 刷新策略列表
      queryClient.invalidateQueries({ queryKey: ['strategy'] });
    },
  });
};

// 投资组合摘要查询
export const usePortfolioSummary = () => {
  return useQuery({
    queryKey: ['portfolio', 'summary'],
    queryFn: portfolioApi.getSummary,
    refetchInterval: 1000,
  });
};

// 风险指标查询
export const useRiskMetrics = () => {
  return useQuery({
    queryKey: ['risk', 'metrics'],
    queryFn: riskApi.getMetrics,
    refetchInterval: 1000,
  });
};
```

## 组件开发

### 组件结构

```typescript
// src/components/Common/DataTable.tsx
import React, { useMemo } from 'react';
import { Table, TableProps } from 'antd';
import { ITableColumn, ITableData } from '@/utils/types';

interface IDataTableProps<T extends ITableData> extends Omit<TableProps<T>, 'columns' | 'dataSource'> {
  columns: ITableColumn<T>[];
  data: T[];
  loading?: boolean;
  onRowClick?: (record: T) => void;
  onRowSelect?: (selectedRows: T[]) => void;
}

function DataTable<T extends ITableData>({
  columns,
  data,
  loading = false,
  onRowClick,
  onRowSelect,
  ...tableProps
}: IDataTableProps<T>) {
  const tableColumns = useMemo(() => {
    return columns.map(column => ({
      ...column,
      key: column.key || column.dataIndex,
    }));
  }, [columns]);

  const rowSelection = useMemo(() => {
    if (!onRowSelect) return undefined;
    
    return {
      onChange: (selectedRowKeys: React.Key[], selectedRows: T[]) => {
        onRowSelect(selectedRows);
      },
    };
  }, [onRowSelect]);

  return (
    <Table<T>
      columns={tableColumns}
      dataSource={data}
      loading={loading}
      rowSelection={rowSelection}
      onRow={(record) => ({
        onClick: () => onRowClick?.(record),
        style: { cursor: onRowClick ? 'pointer' : 'default' },
      })}
      {...tableProps}
    />
  );
}

export default DataTable;
```

### 自定义Hook开发

```typescript
// src/hooks/useWebSocket.ts
import { useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';

interface IUseWebSocketOptions {
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Error) => void;
  onMessage?: (data: any) => void;
  autoReconnect?: boolean;
  reconnectInterval?: number;
}

export const useWebSocket = (url: string, options: IUseWebSocketOptions = {}) => {
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const socketRef = useRef<Socket | null>(null);

  const {
    onConnect,
    onDisconnect,
    onError,
    onMessage,
    autoReconnect = true,
    reconnectInterval = 3000,
  } = options;

  useEffect(() => {
    const socket = io(url, {
      autoConnect: true,
      reconnection: autoReconnect,
      reconnectionDelay: reconnectInterval,
    });

    socketRef.current = socket;

    socket.on('connect', () => {
      setIsConnected(true);
      setError(null);
      onConnect?.();
    });

    socket.on('disconnect', () => {
      setIsConnected(false);
      onDisconnect?.();
    });

    socket.on('error', (err: Error) => {
      setError(err);
      onError?.(err);
    });

    if (onMessage) {
      socket.onAny((event, data) => {
        onMessage({ event, data });
      });
    }

    return () => {
      socket.disconnect();
    };
  }, [url, onConnect, onDisconnect, onError, onMessage, autoReconnect, reconnectInterval]);

  const sendMessage = (event: string, data?: any) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit(event, data);
    }
  };

  const subscribe = (event: string, handler: (data: any) => void) => {
    socketRef.current?.on(event, handler);
  };

  const unsubscribe = (event: string, handler?: (data: any) => void) => {
    if (handler) {
      socketRef.current?.off(event, handler);
    } else {
      socketRef.current?.removeAllListeners(event);
    }
  };

  return {
    isConnected,
    error,
    sendMessage,
    subscribe,
    unsubscribe,
  };
};
```

## 状态管理

### Zustand Store示例

```typescript
// src/stores/systemStore.ts
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { ISystemHealth, ISystemMetrics } from '@/services/api/types';

interface ISystemState {
  // 状态
  health: ISystemHealth | null;
  metrics: ISystemMetrics | null;
  isLoading: boolean;
  error: string | null;
  
  // 动作
  setHealth: (health: ISystemHealth) => void;
  setMetrics: (metrics: ISystemMetrics) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

export const useSystemStore = create<ISystemState>()(
  devtools(
    (set) => ({
      // 初始状态
      health: null,
      metrics: null,
      isLoading: false,
      error: null,

      // 动作实现
      setHealth: (health) => set({ health }),
      setMetrics: (metrics) => set({ metrics }),
      setLoading: (isLoading) => set({ isLoading }),
      setError: (error) => set({ error }),
      reset: () => set({
        health: null,
        metrics: null,
        isLoading: false,
        error: null,
      }),
    }),
    {
      name: 'system-store',
    }
  )
);
```

### 状态持久化

```typescript
// src/stores/userStore.ts
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

interface IUserState {
  user: IUser | null;
  preferences: IUserPreferences;
  setUser: (user: IUser | null) => void;
  updatePreferences: (preferences: Partial<IUserPreferences>) => void;
}

export const useUserStore = create<IUserState>()(
  persist(
    (set) => ({
      user: null,
      preferences: {
        theme: 'light',
        language: 'zh-CN',
        autoRefresh: true,
        refreshInterval: 5000,
      },
      
      setUser: (user) => set({ user }),
      updatePreferences: (newPreferences) => 
        set((state) => ({
          preferences: { ...state.preferences, ...newPreferences }
        })),
    }),
    {
      name: 'user-storage',
      storage: createJSONStorage(() => localStorage),
    }
  )
);
```

## 测试指南

### 单元测试

```typescript
// src/components/__tests__/DataTable.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { vi } from 'vitest';
import DataTable from '../Common/DataTable';

const mockData = [
  { id: '1', name: 'Test 1', value: 100 },
  { id: '2', name: 'Test 2', value: 200 },
];

const mockColumns = [
  { title: 'Name', dataIndex: 'name', key: 'name' },
  { title: 'Value', dataIndex: 'value', key: 'value' },
];

describe('DataTable', () => {
  it('renders table with data', () => {
    render(<DataTable columns={mockColumns} data={mockData} />);
    
    expect(screen.getByText('Test 1')).toBeInTheDocument();
    expect(screen.getByText('Test 2')).toBeInTheDocument();
  });

  it('calls onRowClick when row is clicked', () => {
    const onRowClick = vi.fn();
    render(
      <DataTable 
        columns={mockColumns} 
        data={mockData} 
        onRowClick={onRowClick} 
      />
    );
    
    fireEvent.click(screen.getByText('Test 1'));
    expect(onRowClick).toHaveBeenCalledWith(mockData[0]);
  });

  it('shows loading state', () => {
    render(<DataTable columns={mockColumns} data={[]} loading />);
    
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
  });
});
```

### 集成测试

```typescript
// src/pages/__tests__/Dashboard.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { vi } from 'vitest';
import Dashboard from '../Dashboard';
import * as api from '@/services/api/endpoints';

// Mock API
vi.mock('@/services/api/endpoints');
const mockApi = api as vi.Mocked<typeof api>;

const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: { retry: false },
    mutations: { retry: false },
  },
});

const renderWithProviders = (component: React.ReactElement) => {
  const queryClient = createTestQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      {component}
    </QueryClientProvider>
  );
};

describe('Dashboard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('displays system health status', async () => {
    mockApi.systemApi.getHealth.mockResolvedValue({
      systemRunning: true,
      uptimeSeconds: 3600,
      modules: {},
      timestamp: '2025-01-01T00:00:00Z',
    });

    renderWithProviders(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText('运行中')).toBeInTheDocument();
    });
  });

  it('handles API error gracefully', async () => {
    mockApi.systemApi.getHealth.mockRejectedValue(new Error('API Error'));

    renderWithProviders(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText('数据加载失败')).toBeInTheDocument();
    });
  });
});
```

### E2E测试

```typescript
// tests/e2e/dashboard.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
  });

  test('should display system status', async ({ page }) => {
    await expect(page.locator('[data-testid="system-status"]')).toBeVisible();
    await expect(page.locator('text=运行中')).toBeVisible();
  });

  test('should update data in real-time', async ({ page }) => {
    const portfolioValue = page.locator('[data-testid="portfolio-value"]');
    
    // 等待初始数据加载
    await expect(portfolioValue).toBeVisible();
    
    // 等待数据更新
    await page.waitForTimeout(2000);
    
    // 验证数据已更新（具体验证逻辑取决于实际需求）
    await expect(portfolioValue).toBeVisible();
  });

  test('should handle system control', async ({ page }) => {
    // 停止系统
    await page.click('[data-testid="stop-system-btn"]');
    await expect(page.locator('text=已停止')).toBeVisible();
    
    // 启动系统
    await page.click('[data-testid="start-system-btn"]');
    await expect(page.locator('text=运行中')).toBeVisible();
  });
});
```

## 构建和部署

### 开发环境

```bash
# 启动开发服务器
npm run dev

# 类型检查
npm run type-check

# 代码检查
npm run lint

# 运行测试
npm run test
```

### 生产构建

```bash
# 构建生产版本
npm run build

# 预览生产构建
npm run preview

# 构建前进行类型检查
npm run build:check
```

### Docker部署

```dockerfile
# Dockerfile
FROM node:18-alpine as builder

WORKDIR /app

# 复制package文件
COPY package*.json ./

# 安装依赖
RUN npm ci --only=production

# 复制源代码
COPY . .

# 构建应用
RUN npm run build

# 生产镜像
FROM nginx:alpine

# 复制构建产物
COPY --from=builder /app/dist /usr/share/nginx/html

# 复制nginx配置
COPY nginx.conf /etc/nginx/nginx.conf

# 暴露端口
EXPOSE 80

# 启动nginx
CMD ["nginx", "-g", "daemon off;"]
```

### CI/CD配置

```yaml
# .github/workflows/ci.yml
name: CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Run type check
      run: npm run type-check
    
    - name: Run linting
      run: npm run lint
    
    - name: Run tests
      run: npm run test:run
    
    - name: Run E2E tests
      run: npm run test:e2e

  build:
    needs: test
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Build application
      run: npm run build
    
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist
        path: dist/
```

## 故障排除

### 常见问题

#### 1. 依赖安装失败

```bash
# 清除npm缓存
npm cache clean --force

# 删除node_modules和package-lock.json
rm -rf node_modules package-lock.json

# 重新安装
npm install
```

#### 2. 类型错误

```bash
# 重新生成类型定义
npm run type-check

# 检查TypeScript配置
cat tsconfig.json
```

#### 3. 构建失败

```bash
# 检查构建日志
npm run build 2>&1 | tee build.log

# 检查环境变量
cat .env
```

#### 4. 测试失败

```bash
# 运行单个测试文件
npm run test -- src/components/__tests__/DataTable.test.tsx

# 查看测试覆盖率
npm run test:coverage
```

### 调试技巧

#### 1. 使用React DevTools

安装React DevTools浏览器扩展，用于调试组件状态和props。

#### 2. 使用Redux DevTools

安装Redux DevTools扩展，用于调试Zustand状态管理。

#### 3. 网络请求调试

```typescript
// 在API客户端中添加调试日志
this.client.interceptors.request.use(
  (config) => {
    console.log('API Request:', config);
    return config;
  }
);

this.client.interceptors.response.use(
  (response) => {
    console.log('API Response:', response);
    return response;
  }
);
```

#### 4. 性能调试

```typescript
// 使用React Profiler
import { Profiler } from 'react';

function onRenderCallback(id, phase, actualDuration) {
  console.log('Render:', { id, phase, actualDuration });
}

<Profiler id="App" onRender={onRenderCallback}>
  <App />
</Profiler>
```

### 性能优化

#### 1. 代码分割

```typescript
// 使用React.lazy进行代码分割
const Dashboard = React.lazy(() => import('./pages/Dashboard'));
const Strategy = React.lazy(() => import('./pages/Strategy'));

// 使用Suspense包装
<Suspense fallback={<LoadingSpinner />}>
  <Routes>
    <Route path="/dashboard" element={<Dashboard />} />
    <Route path="/strategy" element={<Strategy />} />
  </Routes>
</Suspense>
```

#### 2. 内存优化

```typescript
// 使用useMemo缓存计算结果
const expensiveValue = useMemo(() => {
  return computeExpensiveValue(data);
}, [data]);

// 使用useCallback缓存函数
const handleClick = useCallback(() => {
  // 处理点击事件
}, [dependency]);
```

#### 3. 虚拟滚动

```typescript
// 使用react-window处理大量数据
import { FixedSizeList as List } from 'react-window';

const VirtualizedList = ({ items }) => (
  <List
    height={600}
    itemCount={items.length}
    itemSize={50}
    itemData={items}
  >
    {({ index, style, data }) => (
      <div style={style}>
        {data[index].name}
      </div>
    )}
  </List>
);
```

---

*本指南最后更新时间: 2025年1月*