import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { vi } from 'vitest';
import StrategyManagementPage from '../index';
import * as api from '../../../services/api';

// Mock the API
vi.mock('../../../services/api', () => ({
  api: {
    strategy: {
      getPerformance: vi.fn(),
      getStrategies: vi.fn(),
      addStrategy: vi.fn(),
      updateStrategy: vi.fn(),
      startStrategy: vi.fn(),
      stopStrategy: vi.fn(),
      deleteStrategy: vi.fn(),
      getStrategyConfig: vi.fn(),
    },
    data: {
      getSymbols: vi.fn(),
    },
  },
}));

// Mock echarts-for-react
vi.mock('echarts-for-react', () => ({
  default: ({ option }: { option: any }) => (
    <div data-testid="echarts-mock">
      {JSON.stringify(option)}
    </div>
  ),
}));

const mockApi = api.api as any;

describe('StrategyManagementPage', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });

    // Setup default mock responses
    mockApi.strategy.getPerformance.mockResolvedValue({
      'test_strategy_1': {
        signalsGenerated: 100,
        successfulTrades: 60,
        totalPnl: 5000,
        winRate: 60,
        avgWin: 150,
        avgLoss: -80,
        profitFactor: 1.875,
        sharpeRatio: 1.2,
        maxDrawdown: 5,
      },
      'test_strategy_2': {
        signalsGenerated: 80,
        successfulTrades: 40,
        totalPnl: -1000,
        winRate: 50,
        avgWin: 120,
        avgLoss: -100,
        profitFactor: 1.2,
        sharpeRatio: 0.8,
        maxDrawdown: 8,
      },
    });

    mockApi.strategy.getStrategies.mockResolvedValue([
      'test_strategy_1',
      'test_strategy_2',
    ]);

    mockApi.data.getSymbols.mockResolvedValue([
      '000001.SZ',
      '000002.SZ',
      '600000.SH',
      '600036.SH',
    ]);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  const renderWithProviders = (component: React.ReactElement) => {
    return render(
      <QueryClientProvider client={queryClient}>
        {component}
      </QueryClientProvider>
    );
  };

  it('should render strategy management page with overview tab', async () => {
    renderWithProviders(<StrategyManagementPage />);

    // Check if the main title is rendered
    expect(screen.getByText('策略管理中心')).toBeInTheDocument();

    // Check if overview tab is active by default
    expect(screen.getByText('概览')).toBeInTheDocument();

    // Wait for data to load and check statistics
    await waitFor(() => {
      expect(screen.getByText('策略总数')).toBeInTheDocument();
      expect(screen.getByText('活跃策略')).toBeInTheDocument();
      expect(screen.getByText('总盈亏')).toBeInTheDocument();
      expect(screen.getByText('平均胜率')).toBeInTheDocument();
    });
  });

  it('should display correct statistics in overview', async () => {
    renderWithProviders(<StrategyManagementPage />);

    await waitFor(() => {
      // Should show strategy statistics
      expect(screen.getByText('策略总数')).toBeInTheDocument();
      expect(screen.getByText('活跃策略')).toBeInTheDocument();
      
      // Should show best performing strategy
      expect(screen.getByText('最佳表现策略')).toBeInTheDocument();
      expect(screen.getByText('test_strategy_1')).toBeInTheDocument();
      
      // Should show strategy that needs attention
      expect(screen.getByText('需要关注的策略')).toBeInTheDocument();
      expect(screen.getByText('test_strategy_2')).toBeInTheDocument();
    });
  });

  it('should have all required tabs', () => {
    renderWithProviders(<StrategyManagementPage />);

    expect(screen.getByText('概览')).toBeInTheDocument();
    expect(screen.getByText('策略配置')).toBeInTheDocument();
    expect(screen.getByText('性能监控')).toBeInTheDocument();
    expect(screen.getByText('操作管理')).toBeInTheDocument();
  });

  it('should show quick action buttons', async () => {
    renderWithProviders(<StrategyManagementPage />);

    await waitFor(() => {
      expect(screen.getByText('添加新策略')).toBeInTheDocument();
      expect(screen.getByText('查看性能分析')).toBeInTheDocument();
      expect(screen.getByText('策略操作管理')).toBeInTheDocument();
    });
  });

  it('should handle API errors gracefully', async () => {
    // Mock API to return error
    mockApi.strategy.getPerformance.mockRejectedValue(new Error('API Error'));
    mockApi.strategy.getStrategies.mockRejectedValue(new Error('API Error'));

    renderWithProviders(<StrategyManagementPage />);

    // Should still render the page structure
    expect(screen.getByText('策略管理中心')).toBeInTheDocument();
    expect(screen.getByText('概览')).toBeInTheDocument();
  });
});

describe('Strategy Components Integration', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });

    mockApi.strategy.getPerformance.mockResolvedValue({});
    mockApi.strategy.getStrategies.mockResolvedValue([]);
    mockApi.data.getSymbols.mockResolvedValue(['000001.SZ']);
  });

  const renderWithProviders = (component: React.ReactElement) => {
    return render(
      <QueryClientProvider client={queryClient}>
        {component}
      </QueryClientProvider>
    );
  };

  it('should integrate all strategy components without errors', () => {
    expect(() => {
      renderWithProviders(<StrategyManagementPage />);
    }).not.toThrow();
  });
});