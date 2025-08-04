# API Services Documentation

This directory contains the API client implementation for the myQuant frontend application.

## Overview

The API services provide a unified interface to communicate with the myQuant backend system, which is built using FastAPI. The implementation includes:

- **Type-safe API client** with comprehensive TypeScript definitions
- **Unified error handling** with retry mechanisms and user-friendly error messages
- **Request/response interceptors** for logging, authentication, and data transformation
- **Modular API organization** by functional domains (system, data, strategy, etc.)
- **React Query integration** through custom hooks for efficient data fetching and caching

## File Structure

```
services/
├── api.ts              # Main API client with axios configuration
├── types.ts            # TypeScript type definitions for all API responses
├── index.ts            # Barrel exports for all services
├── websocket.ts        # WebSocket client for real-time data
├── __tests__/          # Unit tests for API services
│   ├── api.test.ts     # API client tests
│   └── types.test.ts   # Type definition tests
└── README.md           # This documentation
```

## API Client Features

### 1. Unified Configuration

The API client is configured with:
- Base URL from environment variables
- 30-second timeout for requests
- Automatic retry logic (3 attempts with exponential backoff)
- Request/response logging for debugging
- Error transformation and handling

### 2. Modular API Organization

The API is organized into functional modules:

- **System API** (`api.system`): Health checks, metrics, system control
- **Data API** (`api.data`): Market data, tick data, symbols
- **Strategy API** (`api.strategy`): Strategy management and performance
- **Order API** (`api.order`): Order creation, status, history
- **Portfolio API** (`api.portfolio`): Portfolio summary, positions, performance
- **Risk API** (`api.risk`): Risk metrics, alerts, limits
- **Analytics API** (`api.analytics`): Performance analysis, backtesting

### 3. Error Handling

The client includes comprehensive error handling:

```typescript
// Automatic retry for server errors (5xx)
// Timeout handling with custom error messages
// Network error detection and user-friendly messages
// HTTP status code mapping to meaningful error messages
```

### 4. Type Safety

All API responses are fully typed with TypeScript interfaces:

```typescript
// Example usage
const systemHealth: SystemHealth = await api.system.getHealth();
const portfolio: PortfolioSummary = await api.portfolio.getSummary();
```

## Usage Examples

### Basic API Calls

```typescript
import { api } from '@/services';

// Get system health
const health = await api.system.getHealth();

// Get portfolio summary
const portfolio = await api.portfolio.getSummary();

// Create an order
const order = await api.order.createOrder({
  symbol: '000001.SZ',
  side: 'BUY',
  quantity: 1000,
  price: 10.5,
  orderType: 'LIMIT',
});
```

### Using React Query Hooks

```typescript
import { useSystemHealth, usePortfolioSummary } from '@/hooks';

function Dashboard() {
  const { data: health, isLoading: healthLoading } = useSystemHealth();
  const { data: portfolio, isLoading: portfolioLoading } = usePortfolioSummary();

  if (healthLoading || portfolioLoading) {
    return <Spin />;
  }

  return (
    <div>
      <SystemStatus health={health} />
      <PortfolioOverview portfolio={portfolio} />
    </div>
  );
}
```

### Error Handling

```typescript
import { handleApiError } from '@/utils';

try {
  const result = await api.strategy.addStrategy(config);
  message.success('策略添加成功');
} catch (error) {
  handleApiError(error); // Shows user-friendly error message
}
```

## Type Definitions

### Core Types

- `SystemHealth`: System status and module health information
- `PortfolioSummary`: Portfolio value, positions, and P&L
- `RiskMetrics`: Risk indicators and limits
- `StrategyPerformance`: Strategy statistics and performance metrics
- `OrderStatus`: Order information and execution status
- `MarketDataResponse`: Market data records and metadata

### Request Types

- `StrategyConfig`: Configuration for creating strategies
- `OrderRequest`: Parameters for creating orders
- `SystemControlRequest`: System start/stop commands
- `BacktestConfig`: Backtesting configuration parameters

### Utility Types

- `ApiResponse<T>`: Generic API response wrapper
- `PaginationParams`: Pagination parameters for list endpoints
- `DateRangeFilter`: Date range filtering for historical data
- `ApiError`: Standardized error response format

## Configuration

### Environment Variables

```bash
# API Configuration
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000

# Application Configuration
VITE_APP_TITLE=myQuant 量化交易系统
VITE_APP_VERSION=1.0.0
```

### Proxy Configuration (Development)

The Vite development server is configured to proxy API requests:

```typescript
// vite.config.ts
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, ''),
    },
  },
}
```

## Testing

The API services include comprehensive unit tests:

```bash
# Run all tests
npm run test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage
```

### Test Coverage

- Type definition validation
- API client structure verification
- Error handling scenarios
- Mock data integration

## Best Practices

### 1. Use React Query Hooks

Always use the provided React Query hooks instead of calling the API directly in components:

```typescript
// ✅ Good
const { data, isLoading, error } = useSystemHealth();

// ❌ Avoid
useEffect(() => {
  api.system.getHealth().then(setData);
}, []);
```

### 2. Handle Loading and Error States

Always handle loading and error states in your components:

```typescript
const { data, isLoading, error } = usePortfolioSummary();

if (isLoading) return <Spin />;
if (error) return <ErrorMessage error={error} />;
if (!data) return <Empty />;

return <PortfolioDisplay data={data} />;
```

### 3. Use Type Guards

When working with optional data, use type guards:

```typescript
if (portfolio?.positions) {
  Object.entries(portfolio.positions).map(([symbol, position]) => {
    // TypeScript knows position is defined here
  });
}
```

### 4. Leverage Caching

React Query automatically caches API responses. Configure cache times appropriately:

```typescript
// Fast-changing data (1 second cache)
const { data } = usePortfolioSummary({
  staleTime: 1000,
  refetchInterval: 1000,
});

// Slow-changing data (5 minutes cache)
const { data } = useDataSymbols({
  staleTime: 5 * 60 * 1000,
});
```

## Troubleshooting

### Common Issues

1. **Network Errors**: Check if the backend server is running on the correct port
2. **CORS Issues**: Ensure the backend is configured to allow requests from the frontend origin
3. **Type Errors**: Make sure the backend API responses match the TypeScript definitions
4. **Authentication**: Implement authentication headers if required by the backend

### Debug Mode

Enable debug logging by setting the log level:

```bash
VITE_LOG_LEVEL=debug
```

This will show detailed request/response logs in the browser console.

## Contributing

When adding new API endpoints:

1. Add the endpoint to the appropriate API module in `api.ts`
2. Define TypeScript types in `types.ts`
3. Create React Query hooks in `hooks/useApi.ts`
4. Add unit tests in `__tests__/`
5. Update this documentation

## Related Files

- `../hooks/useApi.ts`: React Query hooks for API integration
- `../utils/format.ts`: Data formatting utilities
- `../utils/helpers.ts`: API error handling utilities
- `../utils/constants.ts`: API endpoints and configuration constants