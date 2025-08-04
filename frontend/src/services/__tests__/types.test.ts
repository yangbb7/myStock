import { describe, it, expect } from 'vitest';
import {
  SystemHealth,
  PortfolioSummary,
  RiskMetrics,
  StrategyConfig,
  OrderRequest,
  MarketDataResponse,
} from '../types';

describe('Type Definitions', () => {
  it('should have correct SystemHealth interface', () => {
    const systemHealth: SystemHealth = {
      systemRunning: true,
      uptimeSeconds: 3600,
      modules: {
        data: {
          module: 'data',
          initialized: true,
          metrics: {},
          timestamp: '2024-01-01T00:00:00Z',
        },
      },
      timestamp: '2024-01-01T00:00:00Z',
    };

    expect(systemHealth.systemRunning).toBe(true);
    expect(systemHealth.uptimeSeconds).toBe(3600);
    expect(systemHealth.modules.data.initialized).toBe(true);
  });

  it('should have correct PortfolioSummary interface', () => {
    const portfolio: PortfolioSummary = {
      totalValue: 100000,
      cashBalance: 50000,
      positions: {
        '000001.SZ': {
          symbol: '000001.SZ',
          quantity: 1000,
          averagePrice: 10.5,
          currentPrice: 11.0,
          unrealizedPnl: 500,
        },
      },
      unrealizedPnl: 500,
      positionsCount: 1,
    };

    expect(portfolio.totalValue).toBe(100000);
    expect(portfolio.positionsCount).toBe(1);
    expect(portfolio.positions['000001.SZ'].unrealizedPnl).toBe(500);
  });

  it('should have correct OrderRequest interface', () => {
    const orderRequest: OrderRequest = {
      symbol: '000001.SZ',
      side: 'BUY',
      quantity: 1000,
      price: 10.5,
      orderType: 'LIMIT',
    };

    expect(orderRequest.symbol).toBe('000001.SZ');
    expect(orderRequest.side).toBe('BUY');
    expect(orderRequest.quantity).toBe(1000);
  });

  it('should have correct StrategyConfig interface', () => {
    const strategyConfig: StrategyConfig = {
      name: 'Test Strategy',
      symbols: ['000001.SZ', '000002.SZ'],
      initialCapital: 100000,
      riskTolerance: 0.02,
      maxPositionSize: 0.1,
      indicators: {
        sma: { period: 20 },
        rsi: { period: 14 },
      },
    };

    expect(strategyConfig.name).toBe('Test Strategy');
    expect(strategyConfig.symbols).toHaveLength(2);
    expect(strategyConfig.riskTolerance).toBe(0.02);
  });

  it('should have correct MarketDataResponse interface', () => {
    const marketData: MarketDataResponse = {
      records: [
        {
          datetime: '2024-01-01T09:30:00Z',
          symbol: '000001.SZ',
          open: 10.0,
          high: 10.5,
          low: 9.8,
          close: 10.2,
          volume: 1000000,
          adjClose: 10.2,
        },
      ],
      shape: [1, 8],
      columns: ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'adjClose'],
    };

    expect(marketData.records).toHaveLength(1);
    expect(marketData.records[0].symbol).toBe('000001.SZ');
    expect(marketData.shape).toEqual([1, 8]);
  });
});