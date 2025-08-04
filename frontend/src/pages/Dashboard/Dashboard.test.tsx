import React from 'react';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConfigProvider } from 'antd';
import DashboardPage from './index';

import { vi } from 'vitest';

// Mock the hooks
vi.mock('../../hooks/useApi', () => ({
  useSystemHealth: () => ({
    data: {
      systemRunning: true,
      uptimeSeconds: 3600,
      modules: {
        data: { module: 'data', initialized: true, metrics: {}, timestamp: new Date().toISOString() },
        strategy: { module: 'strategy', initialized: true, metrics: {}, timestamp: new Date().toISOString() },
      },
    },
    isLoading: false,
    error: null,
  }),
  useSystemMetrics: () => ({
    data: {
      system: { running: true, uptime: 3600, modulesCount: 6 },
    },
    isLoading: false,
  }),
  usePortfolioSummary: () => ({
    data: {
      totalValue: 100000,
      cashBalance: 20000,
      positions: {},
      unrealizedPnl: 5000,
      positionsCount: 5,
    },
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  }),
  useRiskMetrics: () => ({
    data: {
      dailyPnl: 1000,
      currentDrawdown: -0.05,
      riskLimits: {
        maxPositionSize: 0.1,
        maxDrawdownLimit: 0.2,
        maxDailyLoss: 5000,
      },
      riskUtilization: {
        dailyLossRatio: 0.2,
        drawdownRatio: 0.25,
      },
    },
    isLoading: false,
    error: null,
  }),
  useSystemControl: () => ({
    startSystem: { mutateAsync: vi.fn(), isPending: false },
    stopSystem: { mutateAsync: vi.fn(), isPending: false },
    restartSystem: { mutateAsync: vi.fn(), isPending: false },
  }),
  usePortfolioHistory: () => ({ data: [] }),
  useRiskAlerts: () => ({ data: [] }),
}));

vi.mock('../../hooks/useRealTime', () => ({
  useSystemStatus: () => ({
    data: null,
    isConnected: true,
    error: null,
    lastUpdated: new Date(),
  }),
  useRiskAlerts: () => ({
    alerts: [],
    latestAlert: null,
    isConnected: true,
    error: null,
    clearAlerts: vi.fn(),
    dismissAlert: vi.fn(),
  }),
}));

describe('DashboardPage', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
  });

  it('renders dashboard components', () => {
    render(
      <QueryClientProvider client={queryClient}>
        <ConfigProvider>
          <DashboardPage />
        </ConfigProvider>
      </QueryClientProvider>
    );

    // Check if main components are rendered
    expect(screen.getByText('系统状态监控')).toBeInTheDocument();
    expect(screen.getByText('投资组合概览')).toBeInTheDocument();
    expect(screen.getByText('风险监控告警')).toBeInTheDocument();
  });

  it('displays system status correctly', () => {
    render(
      <QueryClientProvider client={queryClient}>
        <ConfigProvider>
          <DashboardPage />
        </ConfigProvider>
      </QueryClientProvider>
    );

    expect(screen.getAllByText('运行中')).toHaveLength(4); // Multiple status indicators
  });
});