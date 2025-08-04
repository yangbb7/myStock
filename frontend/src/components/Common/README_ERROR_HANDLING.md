# 前端错误处理系统

这是一个完整的前端错误处理系统，提供了错误捕获、用户通知、离线支持、进度指示和错误报告等功能。

## 功能特性

### 1. 全局错误处理
- **JavaScript错误捕获**: 自动捕获未处理的JavaScript错误
- **Promise拒绝处理**: 处理未捕获的Promise拒绝
- **资源加载错误**: 监控图片、脚本等资源加载失败
- **网络错误处理**: 检测和处理网络连接问题

### 2. 用户友好的错误通知
- **分级通知系统**: 根据错误严重程度显示不同类型的通知
- **智能错误消息**: 将技术错误转换为用户友好的消息
- **重试机制**: 为可重试的错误提供重试选项
- **错误去重**: 避免重复显示相同的错误

### 3. 离线支持和数据降级
- **网络状态监控**: 实时监控网络连接状态
- **数据缓存**: 自动缓存成功的API响应
- **离线数据展示**: 在离线时显示缓存数据
- **数据同步**: 网络恢复时自动同步数据

### 4. 加载状态和进度指示
- **骨架屏**: 为不同类型的内容提供骨架屏
- **进度指示器**: 显示多步骤操作的进度
- **加载状态管理**: 统一的加载状态处理
- **异步操作反馈**: 为异步操作提供用户反馈

### 5. 错误报告和监控
- **错误日志收集**: 自动收集和存储错误信息
- **错误统计**: 提供错误统计和分析
- **错误报告**: 将错误发送到监控服务
- **调试支持**: 开发环境下的详细错误信息

## 组件说明

### ErrorBoundary
React错误边界组件，捕获组件树中的JavaScript错误。

```tsx
import { ErrorBoundary } from './components/Common';

<ErrorBoundary
  onError={(error, errorInfo) => {
    console.error('Error caught:', error, errorInfo);
  }}
>
  <YourComponent />
</ErrorBoundary>
```

### ErrorHandler
错误处理工具类，提供统一的错误处理方法。

```tsx
import { ErrorHandler } from './components/Common';

// 处理API错误
ErrorHandler.handleApiError(apiError, 'user-profile');

// 处理网络错误
ErrorHandler.handleNetworkError(networkError, 'data-fetch');
```

### LoadingState
统一的加载状态组件，支持骨架屏和进度指示。

```tsx
import { LoadingState } from './components/Common';

<LoadingState
  loading={isLoading}
  error={error}
  skeleton={true}
  onRetry={handleRetry}
  showProgress={true}
  progress={progressValue}
>
  <YourContent />
</LoadingState>
```

### DataFallback
数据降级组件，提供离线支持和缓存功能。

```tsx
import { DataFallback } from './components/Common';

<DataFallback
  error={error}
  data={data}
  cacheKey="user-data"
  allowStaleData={true}
  onRetry={refetchData}
>
  <DataDisplay data={data} />
</DataFallback>
```

### ProgressIndicator
进度指示器组件，显示多步骤操作的进度。

```tsx
import { ProgressIndicator, useProgressSteps } from './components/Common';

const { steps, startStep, completeStep, errorStep } = useProgressSteps([
  { id: 'step1', title: '初始化' },
  { id: 'step2', title: '处理数据' },
  { id: 'step3', title: '保存结果' },
]);

<ProgressIndicator
  steps={steps}
  title="操作进度"
  onRetry={handleRetry}
/>
```

### NetworkStatusIndicator
网络状态指示器，显示当前网络连接状态。

```tsx
import { NetworkStatusIndicator } from './components/Common';

<NetworkStatusIndicator
  showDetails={true}
  position="topRight"
/>
```

## Hooks说明

### useErrorHandling
基础错误处理Hook。

```tsx
import { useErrorHandling } from './hooks';

const { handleError, hasError, error, clearError, retry } = useErrorHandling({
  context: 'user-profile',
  component: 'UserProfile',
  onRetry: () => refetchData(),
});
```

### useApiErrorHandling
API错误处理Hook，专门处理API相关错误。

```tsx
import { useApiErrorHandling } from './hooks';

const { handleError } = useApiErrorHandling('api-context');

try {
  const data = await api.fetchData();
} catch (error) {
  handleError(error);
}
```

### useAsyncErrorHandling
异步操作错误处理Hook。

```tsx
import { useAsyncErrorHandling } from './hooks';

const { loading, data, execute } = useAsyncErrorHandling(
  async () => {
    return await api.fetchData();
  },
  {
    onSuccess: (data) => console.log('Success:', data),
    onError: (error) => console.error('Error:', error),
  }
);
```

### useNetworkStatus
网络状态监控Hook。

```tsx
import { useNetworkStatus } from './components/Common';

const { isOnline, lastOnline } = useNetworkStatus();
```

### useDataCache
数据缓存Hook。

```tsx
import { useDataCache } from './components/Common';

const { setCache, getCache, hasCache } = useDataCache('cache-key', 5 * 60 * 1000);
```

## 骨架屏组件

系统提供了多种预定义的骨架屏组件：

```tsx
import { 
  DashboardSkeleton,
  TableSkeleton,
  ChartSkeleton,
  FormSkeleton,
  PortfolioSkeleton,
  OrderSkeleton,
  RiskSkeleton,
  MarketDataSkeleton
} from './components/Common';

// 使用示例
<LoadingState loading={true} skeleton>
  <DashboardSkeleton />
</LoadingState>
```

## 配置和初始化

在应用入口处配置错误处理系统：

```tsx
import { ErrorHandlingProvider } from './utils/errorHandlingSetup';

function App() {
  return (
    <ErrorHandlingProvider
      config={{
        enableErrorReporting: true,
        enableGlobalHandlers: true,
        enableConsoleLogging: process.env.NODE_ENV === 'development',
        userId: 'user-123',
        notificationDuration: 5,
      }}
    >
      <YourApp />
    </ErrorHandlingProvider>
  );
}
```

## 错误报告

系统自动收集错误信息并发送到监控服务：

```tsx
import { errorReporting } from './services/errorReporting';

// 手动报告错误
const errorId = errorReporting.reportError(error, {
  component: 'UserProfile',
  action: 'save-profile',
});

// 获取错误统计
const stats = errorReporting.getErrorStats();
console.log('Total errors:', stats.total);
console.log('Critical errors:', stats.bySeverity.critical);
```

## 最佳实践

### 1. 错误边界使用
在关键组件周围使用错误边界：

```tsx
<ErrorBoundary>
  <CriticalComponent />
</ErrorBoundary>
```

### 2. API错误处理
统一处理API错误：

```tsx
const { handleError } = useApiErrorHandling('api-context');

const fetchData = async () => {
  try {
    const response = await api.getData();
    return response;
  } catch (error) {
    handleError(error);
    throw error;
  }
};
```

### 3. 加载状态管理
为所有异步操作提供加载状态：

```tsx
<LoadingState
  loading={isLoading}
  error={error}
  skeleton={true}
  onRetry={handleRetry}
>
  <DataComponent />
</LoadingState>
```

### 4. 离线支持
为重要数据提供离线支持：

```tsx
<DataFallback
  error={error}
  data={data}
  cacheKey="important-data"
  allowStaleData={true}
>
  <ImportantComponent />
</DataFallback>
```

### 5. 进度指示
为多步骤操作提供进度指示：

```tsx
const handleComplexOperation = async () => {
  startStep('step1');
  await step1();
  completeStep('step1');
  
  startStep('step2');
  try {
    await step2();
    completeStep('step2');
  } catch (error) {
    errorStep('step2', error.message);
  }
};
```

## 调试和监控

### 开发环境
- 详细的控制台日志
- 错误堆栈跟踪
- 网络请求监控

### 生产环境
- 错误报告发送到监控服务
- 用户友好的错误消息
- 性能监控

### 错误统计
```tsx
import { errorHandlingUtils } from './utils/errorHandlingSetup';

// 获取健康检查
const health = errorHandlingUtils.healthCheck();
console.log('System healthy:', health.healthy);

// 导出错误报告
errorHandlingUtils.exportErrorReports();

// 清除所有错误
errorHandlingUtils.clearAllErrors();
```

## 扩展和自定义

### 自定义错误处理器
```tsx
const customErrorHandler = (error: Error, context?: string) => {
  // 自定义错误处理逻辑
  console.log('Custom error handler:', error, context);
  
  // 调用默认处理器
  ErrorHandler.handleSystemError(error, context);
};
```

### 自定义骨架屏
```tsx
const CustomSkeleton: React.FC = () => (
  <div>
    <Skeleton.Input style={{ width: 200 }} active />
    <Skeleton active paragraph={{ rows: 4 }} />
  </div>
);
```

### 自定义错误边界
```tsx
const CustomErrorBoundary: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <ErrorBoundary
    fallback={({ error, retry }) => (
      <CustomErrorDisplay error={error} onRetry={retry} />
    )}
  >
    {children}
  </ErrorBoundary>
);
```

这个错误处理系统提供了完整的前端错误管理解决方案，确保用户在遇到错误时能够获得良好的体验，同时为开发者提供了强大的调试和监控工具。