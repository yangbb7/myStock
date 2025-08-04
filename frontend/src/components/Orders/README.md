# Order Management Components

This directory contains the complete order management functionality for the myQuant frontend application.

## Components Overview

### 1. OrderCreateForm
A comprehensive form component for creating new trading orders with built-in risk validation.

**Features:**
- Support for multiple order types (Market, Limit, Stop, Stop-Limit)
- Real-time risk validation and warnings
- Portfolio balance checking
- Position size limit validation
- Estimated order value calculation
- Risk confirmation modal for high-risk orders

**Props:**
```typescript
interface OrderCreateFormProps {
  onSuccess?: (orderId: string) => void;
  onCancel?: () => void;
  initialValues?: Partial<OrderRequest>;
}
```

**Usage:**
```tsx
import { OrderCreateForm } from '../components/Orders';

<OrderCreateForm
  onSuccess={(orderId) => console.log('Order created:', orderId)}
  onCancel={() => setModalVisible(false)}
/>
```

### 2. OrderStatusMonitor
Real-time order monitoring component with filtering and search capabilities.

**Features:**
- Real-time order status updates via WebSocket
- Order filtering by status, side, date range
- Order search by ID or symbol
- Order cancellation functionality
- Detailed order information modal
- Order statistics display
- Active orders alerts

**Props:**
```typescript
interface OrderStatusMonitorProps {
  refreshInterval?: number;
  showFilters?: boolean;
  showStats?: boolean;
}
```

**Usage:**
```tsx
import { OrderStatusMonitor } from '../components/Orders';

<OrderStatusMonitor
  refreshInterval={5000}
  showFilters={true}
  showStats={true}
/>
```

### 3. OrderAnalytics
Comprehensive order analytics and performance visualization component.

**Features:**
- Order volume and success rate trends
- Execution time analysis
- Symbol-based statistics
- Order status distribution charts
- Performance alerts and warnings
- Customizable time range analysis

**Props:**
```typescript
interface OrderAnalyticsProps {
  defaultTimeRange?: [dayjs.Dayjs, dayjs.Dayjs];
}
```

**Usage:**
```tsx
import { OrderAnalytics } from '../components/Orders';
import dayjs from 'dayjs';

<OrderAnalytics
  defaultTimeRange={[dayjs().subtract(30, 'day'), dayjs()]}
/>
```

## API Integration

The components integrate with the following API endpoints:

### Order Management
- `POST /order/create` - Create new order
- `GET /order/status/{orderId}` - Get order status
- `GET /order/history` - Get order history with filters
- `GET /order/active` - Get active orders
- `GET /order/stats` - Get order statistics
- `POST /order/cancel/{orderId}` - Cancel order

### Supporting APIs
- `GET /data/symbols` - Get available symbols
- `GET /portfolio/summary` - Get portfolio information
- `GET /risk/metrics` - Get risk metrics for validation

### WebSocket Events
- `order_update` - Real-time order status updates
- `market_data` - Market data updates for price validation

## Risk Management

The order creation form includes comprehensive risk management features:

1. **Cash Balance Validation**: Ensures sufficient funds for buy orders
2. **Position Size Limits**: Validates against maximum position size settings
3. **Daily Loss Limits**: Checks potential impact on daily P&L limits
4. **Position Validation**: Prevents overselling of current positions
5. **Risk Confirmation**: Requires explicit confirmation for high-risk orders

## Real-time Features

### WebSocket Integration
All components support real-time updates through WebSocket connections:

```typescript
// WebSocket message format
interface OrderUpdateMessage {
  type: 'order_update';
  data: OrderStatus;
  timestamp: string;
}
```

### Auto-refresh
Components automatically refresh data at configurable intervals:
- Order status: Every 1-5 seconds
- Portfolio data: Every 5 seconds
- Market data: Real-time via WebSocket

## Testing

### Unit Tests
Each component includes comprehensive unit tests:

```bash
npm run test -- OrderCreateForm.test.tsx
npm run test -- OrderStatusMonitor.test.tsx
npm run test -- OrderAnalytics.test.tsx
```

### Integration Tests
Page-level integration tests verify component interactions:

```bash
npm run test -- OrdersPage.test.tsx
```

## Performance Considerations

1. **Data Caching**: Uses React Query for intelligent data caching
2. **Virtual Scrolling**: Large order lists use virtual scrolling
3. **Debounced Search**: Search inputs are debounced to reduce API calls
4. **Lazy Loading**: Charts and analytics load on demand
5. **Memory Management**: WebSocket connections are properly cleaned up

## Accessibility

All components follow accessibility best practices:
- Proper ARIA labels and roles
- Keyboard navigation support
- Screen reader compatibility
- High contrast color schemes
- Focus management

## Customization

### Theming
Components support Ant Design theme customization:

```tsx
import { ConfigProvider } from 'antd';

<ConfigProvider
  theme={{
    token: {
      colorPrimary: '#1890ff',
      borderRadius: 6,
    },
  }}
>
  <OrdersPage />
</ConfigProvider>
```

### Styling
Custom styles can be applied via CSS classes or styled-components:

```css
.order-create-form {
  .ant-form-item {
    margin-bottom: 16px;
  }
}
```

## Error Handling

Components include comprehensive error handling:

1. **API Errors**: User-friendly error messages with retry options
2. **Network Errors**: Offline detection and graceful degradation
3. **Validation Errors**: Real-time form validation with clear feedback
4. **WebSocket Errors**: Automatic reconnection with exponential backoff

## Future Enhancements

Planned improvements include:

1. **Advanced Order Types**: Support for more complex order types
2. **Bulk Operations**: Multi-order creation and management
3. **Order Templates**: Save and reuse common order configurations
4. **Advanced Analytics**: More sophisticated performance metrics
5. **Mobile Optimization**: Enhanced mobile experience
6. **Offline Support**: Offline order queuing and synchronization

## Dependencies

Key dependencies used by these components:

- `@tanstack/react-query` - Data fetching and caching
- `antd` - UI components
- `recharts` - Charts and visualizations
- `dayjs` - Date manipulation
- `socket.io-client` - WebSocket communication
- `lodash` - Utility functions

## Contributing

When contributing to these components:

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure accessibility compliance
5. Test with real API integration
6. Consider performance implications