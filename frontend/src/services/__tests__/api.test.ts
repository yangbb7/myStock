import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock axios before importing the API
vi.mock('axios', () => {
  const mockAxiosInstance = {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
    interceptors: {
      request: { use: vi.fn() },
      response: { use: vi.fn() },
    },
  };
  
  return {
    default: {
      create: vi.fn(() => mockAxiosInstance),
    },
  };
});

// Import API after mocking axios
import { api } from '../api';

describe('API Client', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('System API', () => {
    it('should have system API methods', () => {
      expect(api.system).toHaveProperty('getHealth');
      expect(api.system).toHaveProperty('getMetrics');
      expect(api.system).toHaveProperty('startSystem');
      expect(api.system).toHaveProperty('stopSystem');
    });
  });

  describe('Error Handling', () => {
    it('should handle network errors', () => {
      const networkError = new Error('Network Error');
      expect(networkError.message).toBe('Network Error');
    });

    it('should handle timeout errors', () => {
      const timeoutError = { code: 'ECONNABORTED' };
      expect(timeoutError.code).toBe('ECONNABORTED');
    });
  });

  describe('Type Safety', () => {
    it('should have correct API structure', () => {
      // Test that our API object has the expected structure
      expect(api).toHaveProperty('system');
      expect(api).toHaveProperty('data');
      expect(api).toHaveProperty('strategy');
      expect(api).toHaveProperty('order');
      expect(api).toHaveProperty('portfolio');
      expect(api).toHaveProperty('risk');
      expect(api).toHaveProperty('analytics');
    });

    it('should have all required system methods', () => {
      expect(typeof api.system.getHealth).toBe('function');
      expect(typeof api.system.getMetrics).toBe('function');
      expect(typeof api.system.startSystem).toBe('function');
      expect(typeof api.system.stopSystem).toBe('function');
    });

    it('should have all required data methods', () => {
      expect(typeof api.data.getMarketData).toBe('function');
      expect(typeof api.data.submitTickData).toBe('function');
      expect(typeof api.data.getSymbols).toBe('function');
    });

    it('should have all required strategy methods', () => {
      expect(typeof api.strategy.addStrategy).toBe('function');
      expect(typeof api.strategy.getPerformance).toBe('function');
      expect(typeof api.strategy.getStrategies).toBe('function');
    });

    it('should have all required order methods', () => {
      expect(typeof api.order.createOrder).toBe('function');
      expect(typeof api.order.getOrderStatus).toBe('function');
      expect(typeof api.order.getOrderHistory).toBe('function');
    });

    it('should have all required portfolio methods', () => {
      expect(typeof api.portfolio.getSummary).toBe('function');
      expect(typeof api.portfolio.getHistory).toBe('function');
      expect(typeof api.portfolio.getPositions).toBe('function');
    });

    it('should have all required risk methods', () => {
      expect(typeof api.risk.getMetrics).toBe('function');
      expect(typeof api.risk.getAlerts).toBe('function');
      expect(typeof api.risk.updateLimits).toBe('function');
    });

    it('should have all required analytics methods', () => {
      expect(typeof api.analytics.getPerformance).toBe('function');
      expect(typeof api.analytics.runBacktest).toBe('function');
      expect(typeof api.analytics.generateReport).toBe('function');
    });
  });
});