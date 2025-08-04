import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { vi } from 'vitest';
import RiskMonitoringPage from '../index';
import * as api from '../../../services/api';

// Mock the API
vi.mock('../../../services/api');
const mockApi = api as any;

// Mock the hooks
vi.mock('../../../hooks/useRealTime', () => ({
  useRiskAlerts: () => ({
    alerts: [],
    latestAlert: null,
    isConnected: true,
    error: null,
    clearAlerts: vi.fn(),
    dismissAlert: vi.fn(),
  }),
}));

// Mock ECharts
vi.mock('echarts-for-react', () => ({
  default: function MockECharts() {
    return <div data-testid="risk-chart">Risk Chart</div>;
  }
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <BrowserRouter>
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    </BrowserRouter>
  );
};

describe('RiskMonitoringPage', () => {
  beforeEach(() => {
    // Mock API responses
    mockApi.risk = {
      getMetrics: vi.fn().mockResolvedValue({
        dailyPnl: 1500.50,
        currentDrawdown: 2.5,
        riskLimits: {
          maxPositionSize: 0.3,
          maxDrawdownLimit: 0.1,
          maxDailyLoss: 5000,
        },
        riskUtilization: {
          dailyLossRatio: 0.3,
          drawdownRatio: 0.25,
        },
      }),
      getConfig: vi.fn().mockResolvedValue({
        maxPositionSize: 0.3,
        maxDrawdownLimit: 0.1,
        maxDailyLoss: 5000,
      }),
      getAlerts: vi.fn().mockResolvedValue([]),
      updateLimits: vi.fn().mockResolvedValue({ success: true }),
    };

    mockApi.system = {
      getHealth: vi.fn().mockResolvedValue({
        systemRunning: true,
        uptimeSeconds: 3600,
        modules: {},
        timestamp: new Date().toISOString(),
      }),
      stopSystem: vi.fn().mockResolvedValue({ success: true }),
      startSystem: vi.fn().mockResolvedValue({ success: true }),
      restartSystem: vi.fn().mockResolvedValue({ success: true }),
    };
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should render risk monitoring page with all components', async () => {
    render(<RiskMonitoringPage />, { wrapper: createWrapper() });

    // Check if the main title is rendered
    expect(screen.getByText('风险监控')).toBeInTheDocument();

    // Wait for components to load
    await waitFor(() => {
      // Check if risk dashboard components are rendered
      expect(screen.getByText('风险指标仪表板')).toBeInTheDocument();
      expect(screen.getByText('风险告警系统')).toBeInTheDocument();
      expect(screen.getByText('风险控制操作')).toBeInTheDocument();
    });
  });

  it('should display risk metrics correctly', async () => {
    render(<RiskMonitoringPage />, { wrapper: createWrapper() });

    await waitFor(() => {
      // Check if risk metrics are displayed
      expect(screen.getByText('日盈亏')).toBeInTheDocument();
      expect(screen.getByText('当前回撤')).toBeInTheDocument();
      expect(screen.getByText('风险限制')).toBeInTheDocument();
    });
  });

  it('should show system status correctly', async () => {
    render(<RiskMonitoringPage />, { wrapper: createWrapper() });

    await waitFor(() => {
      // Check if system status is displayed
      expect(screen.getByText(/系统状态.*运行中/)).toBeInTheDocument();
    });
  });

  it('should render control buttons', async () => {
    render(<RiskMonitoringPage />, { wrapper: createWrapper() });

    await waitFor(() => {
      // Check if control buttons are rendered
      expect(screen.getByText('紧急停止')).toBeInTheDocument();
      expect(screen.getByText('系统重启')).toBeInTheDocument();
      expect(screen.getByText('调整风险限制')).toBeInTheDocument();
    });
  });

  it('should handle API errors gracefully', async () => {
    // Mock API error
    mockApi.risk.getMetrics = vi.fn().mockRejectedValue(new Error('API Error'));

    render(<RiskMonitoringPage />, { wrapper: createWrapper() });

    await waitFor(() => {
      // Check if error message is displayed
      expect(screen.getByText('数据加载失败')).toBeInTheDocument();
    });
  });
});