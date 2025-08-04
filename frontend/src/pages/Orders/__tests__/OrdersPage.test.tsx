import React from 'react';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { vi } from 'vitest';
import OrdersPage from '../OrdersPage';

// Mock the API
vi.mock('../../../services/api', () => ({
  api: {
    data: {
      getSymbols: vi.fn().mockResolvedValue(['000001.SZ', '000002.SZ']),
    },
    portfolio: {
      getSummary: vi.fn().mockResolvedValue({
        totalValue: 1000000,
        cashBalance: 500000,
        positions: {},
        unrealizedPnl: 0,
        positionsCount: 0,
      }),
    },
    risk: {
      getMetrics: vi.fn().mockResolvedValue({
        dailyPnl: 0,
        currentDrawdown: 0,
        riskLimits: {
          maxPositionSize: 0.2,
          maxDrawdownLimit: 0.1,
          maxDailyLoss: 50000,
        },
        riskUtilization: {
          dailyLossRatio: 0,
          drawdownRatio: 0,
        },
      }),
    },
    order: {
      getOrderHistory: vi.fn().mockResolvedValue([]),
      getActiveOrders: vi.fn().mockResolvedValue([]),
      getOrderStats: vi.fn().mockResolvedValue({
        totalOrders: 0,
        successRate: 0,
        avgExecutionTime: 0,
      }),
    },
  },
}));

// Mock WebSocket hook
vi.mock('../../../hooks/useWebSocket', () => ({
  useWebSocket: vi.fn(),
}));

describe('OrdersPage', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
  });

  const renderComponent = () => {
    return render(
      <QueryClientProvider client={queryClient}>
        <OrdersPage />
      </QueryClientProvider>
    );
  };

  it('should render orders page with main components', async () => {
    renderComponent();

    // Check page title
    expect(screen.getByRole('heading', { name: /订单管理/ })).toBeInTheDocument();
    
    // Check create order button
    expect(screen.getByRole('button', { name: /创建订单/ })).toBeInTheDocument();
    
    // Check tabs
    expect(screen.getByRole('tab', { name: /订单监控/ })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /统计分析/ })).toBeInTheDocument();
  });

  it('should have correct page structure', () => {
    renderComponent();

    // Check if the main layout elements are present
    expect(screen.getByText('管理和监控所有交易订单，查看执行状态和性能分析')).toBeInTheDocument();
  });
});