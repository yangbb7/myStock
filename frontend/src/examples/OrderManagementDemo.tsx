import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import OrdersPage from '../pages/Orders/OrdersPage';

// Create a query client for the demo
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

/**
 * Order Management Demo Component
 * 
 * This demo showcases the complete order management functionality including:
 * 1. Manual order creation with risk validation
 * 2. Real-time order status monitoring
 * 3. Order statistics and analytics
 * 
 * Features demonstrated:
 * - Order creation form with validation
 * - Risk checking and warnings
 * - Real-time order status updates
 * - Order filtering and search
 * - Performance analytics and charts
 * - WebSocket integration for live updates
 */
const OrderManagementDemo: React.FC = () => {
  return (
    <ConfigProvider locale={zhCN}>
      <QueryClientProvider client={queryClient}>
        <div style={{ minHeight: '100vh', background: '#f0f2f5' }}>
          <OrdersPage />
        </div>
      </QueryClientProvider>
    </ConfigProvider>
  );
};

export default OrderManagementDemo;