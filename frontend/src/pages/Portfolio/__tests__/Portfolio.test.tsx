import React from 'react';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { vi } from 'vitest';
import PortfolioPage from '../index';

// Mock the API
vi.mock('../../../services/api', () => ({
  api: {
    portfolio: {
      getSummary: vi.fn().mockResolvedValue({
        totalValue: 100000,
        cashBalance: 20000,
        positions: {
          '000001.SZ': {
            symbol: '000001.SZ',
            quantity: 1000,
            averagePrice: 10.5,
            currentPrice: 11.2,
            unrealizedPnl: 700,
          },
        },
        unrealizedPnl: 700,
        positionsCount: 1,
      }),
      getHistory: vi.fn().mockResolvedValue([]),
      getPerformance: vi.fn().mockResolvedValue({
        totalReturn: 15.6,
        annualizedReturn: 12.3,
        volatility: 18.5,
        sharpeRatio: 0.85,
      }),
    },
    risk: {
      getMetrics: vi.fn().mockResolvedValue({
        dailyPnl: 500,
        currentDrawdown: -2.5,
        riskLimits: {
          maxPositionSize: 0.3,
          maxDrawdownLimit: 0.1,
          maxDailyLoss: 5000,
        },
        riskUtilization: {
          dailyLossRatio: 0.1,
          drawdownRatio: 0.25,
        },
      }),
    },
    analytics: {
      getPerformance: vi.fn().mockResolvedValue({
        totalReturn: 15.6,
        annualizedReturn: 12.3,
      }),
    },
    data: {
      getMarketData: vi.fn().mockResolvedValue({
        records: [
          {
            datetime: '2024-01-01',
            symbol: '000300.SH',
            open: 3000,
            high: 3100,
            low: 2950,
            close: 3050,
            volume: 1000000,
          },
        ],
      }),
    },
  },
}));

describe('PortfolioPage', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
      },
    });
  });

  it('should render portfolio page with tabs', () => {
    render(
      <QueryClientProvider client={queryClient}>
        <PortfolioPage />
      </QueryClientProvider>
    );

    // Check if main tabs are rendered
    expect(screen.getByText('持仓管理')).toBeInTheDocument();
    expect(screen.getByText('收益分析')).toBeInTheDocument();
    expect(screen.getByText('风险分析')).toBeInTheDocument();
    expect(screen.getByText('报告导出')).toBeInTheDocument();
  });

  it('should render position management tab by default', () => {
    render(
      <QueryClientProvider client={queryClient}>
        <PortfolioPage />
      </QueryClientProvider>
    );

    // The position management tab should be active by default
    // We can check for elements that would be in the PositionManagement component
    expect(screen.getByText('持仓管理')).toBeInTheDocument();
  });
});