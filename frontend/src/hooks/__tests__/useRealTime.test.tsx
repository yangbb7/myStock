import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useMarketData, useSystemStatus, useRiskAlerts } from '../useRealTime';
import * as websocketModule from '../useWebSocket';

// Mock the useWebSocket hook
const mockUseWebSocket = vi.fn();
vi.mock('../useWebSocket', () => ({
  useWebSocket: () => mockUseWebSocket(),
}));

describe('useRealTime hooks', () => {
  let mockWebSocketReturn: any;

  beforeEach(() => {
    mockWebSocketReturn = {
      isConnected: true,
      connectionState: 'connected',
      error: null,
      subscribe: vi.fn().mockReturnValue('sub_123'),
      unsubscribe: vi.fn(),
      connect: vi.fn(),
      disconnect: vi.fn(),
    };
    mockUseWebSocket.mockReturnValue(mockWebSocketReturn);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('useMarketData', () => {
    it('should initialize with empty data', () => {
      const { result } = renderHook(() => useMarketData());

      expect(result.current.data).toEqual({});
      expect(result.current.isConnected).toBe(true);
      expect(result.current.error).toBeNull();
    });

    it('should subscribe to market data on mount', () => {
      const symbols = ['AAPL', 'GOOGL'];
      renderHook(() => useMarketData({ symbols, autoSubscribe: true }));

      expect(mockWebSocketReturn.subscribe).toHaveBeenCalledWith(
        'market_data',
        expect.any(Function),
        {
          symbols,
          throttle: 100,
          bufferSize: 10,
        }
      );
    });

    it('should update data when receiving market data', () => {
      const { result } = renderHook(() => useMarketData({ 
        symbols: ['AAPL'], 
        autoSubscribe: true 
      }));

      // Get the callback function passed to subscribe
      expect(mockWebSocketReturn.subscribe).toHaveBeenCalled();
      const subscribeCall = mockWebSocketReturn.subscribe.mock.calls[0];
      const callback = subscribeCall?.[1];

      const marketData = {
        symbol: 'AAPL',
        price: 150.0,
        change: 2.5,
        changePercent: 0.017,
        volume: 1000000,
        timestamp: '2023-01-01T10:00:00Z',
      };

      act(() => {
        callback(marketData);
      });

      expect(result.current.data['AAPL']).toEqual(
        expect.objectContaining({
          ...marketData,
          timestamp: expect.any(String)
        })
      );
    });

    it('should provide getSymbolData function', () => {
      const { result } = renderHook(() => useMarketData({ 
        symbols: ['AAPL'], 
        autoSubscribe: true 
      }));

      // Add some data first
      expect(mockWebSocketReturn.subscribe).toHaveBeenCalled();
      const subscribeCall = mockWebSocketReturn.subscribe.mock.calls[0];
      const callback = subscribeCall?.[1];

      const marketData = {
        symbol: 'AAPL',
        price: 150.0,
        timestamp: '2023-01-01T10:00:00Z',
      };

      act(() => {
        callback(marketData);
      });

      const symbolData = result.current.getSymbolData('AAPL');
      expect(symbolData).toEqual(expect.objectContaining({
        ...marketData,
        timestamp: expect.any(String)
      }));

      const nonExistentData = result.current.getSymbolData('NONEXISTENT');
      expect(nonExistentData).toBeUndefined();
    });

    it('should unsubscribe on unmount', () => {
      const { unmount } = renderHook(() => useMarketData({ 
        symbols: ['AAPL'], 
        autoSubscribe: true 
      }));
      
      unmount();

      // The hook should have called unsubscribe during cleanup
      expect(mockWebSocketReturn.unsubscribe).toHaveBeenCalled();
    });
  });

  describe('useSystemStatus', () => {
    it('should initialize with null data', () => {
      const { result } = renderHook(() => useSystemStatus());

      expect(result.current.data).toBeNull();
      expect(result.current.isConnected).toBe(true);
      expect(result.current.error).toBeNull();
      expect(result.current.lastUpdated).toBeNull();
    });

    it('should subscribe to system status on mount', () => {
      renderHook(() => useSystemStatus({ autoSubscribe: true }));

      expect(mockWebSocketReturn.subscribe).toHaveBeenCalledWith(
        'system_status',
        expect.any(Function)
      );
    });

    it('should update data and timestamp when receiving system status', () => {
      const { result } = renderHook(() => useSystemStatus());

      const subscribeCall = mockWebSocketReturn.subscribe.mock.calls[0];
      const callback = subscribeCall[1];

      const systemHealth = {
        systemRunning: true,
        uptimeSeconds: 3600,
        modules: {
          data: { module: 'data', initialized: true, metrics: {}, timestamp: '2023-01-01T10:00:00Z' },
        },
        timestamp: '2023-01-01T10:00:00Z',
      };

      act(() => {
        callback(systemHealth);
      });

      expect(result.current.data).toEqual(systemHealth);
      expect(result.current.lastUpdated).toBeInstanceOf(Date);
    });
  });

  describe('useRiskAlerts', () => {
    it('should initialize with empty alerts', () => {
      const { result } = renderHook(() => useRiskAlerts());

      expect(result.current.alerts).toEqual([]);
      expect(result.current.latestAlert).toBeNull();
      expect(result.current.isConnected).toBe(true);
      expect(result.current.error).toBeNull();
    });

    it('should add alerts when receiving risk alerts', () => {
      const { result } = renderHook(() => useRiskAlerts());

      const subscribeCall = mockWebSocketReturn.subscribe.mock.calls[0];
      const callback = subscribeCall[1];

      const riskAlert = {
        level: 'warning' as const,
        message: 'Risk threshold exceeded',
        metrics: {
          dailyPnl: -1000,
          currentDrawdown: 5.5,
          riskLimits: {
            maxPositionSize: 0.1,
            maxDrawdownLimit: 10,
            maxDailyLoss: 5000,
          },
          riskUtilization: {
            dailyLossRatio: 0.2,
            drawdownRatio: 0.55,
          },
        },
      };

      act(() => {
        callback(riskAlert);
      });

      expect(result.current.alerts).toHaveLength(1);
      expect(result.current.latestAlert).toEqual(
        expect.objectContaining(riskAlert)
      );
    });

    it('should provide clearAlerts function', () => {
      const { result } = renderHook(() => useRiskAlerts());

      // Add an alert first
      const subscribeCall = mockWebSocketReturn.subscribe.mock.calls[0];
      const callback = subscribeCall[1];

      act(() => {
        callback({
          level: 'warning',
          message: 'Test alert',
          metrics: {} as any,
        });
      });

      expect(result.current.alerts).toHaveLength(1);

      act(() => {
        result.current.clearAlerts();
      });

      expect(result.current.alerts).toHaveLength(0);
      expect(result.current.latestAlert).toBeNull();
    });

    it('should provide dismissAlert function', () => {
      const { result } = renderHook(() => useRiskAlerts());

      const subscribeCall = mockWebSocketReturn.subscribe.mock.calls[0];
      const callback = subscribeCall[1];

      // Add multiple alerts
      act(() => {
        callback({
          level: 'warning',
          message: 'Alert 1',
          metrics: {} as any,
        });
        callback({
          level: 'error',
          message: 'Alert 2',
          metrics: {} as any,
        });
      });

      expect(result.current.alerts).toHaveLength(2);

      act(() => {
        result.current.dismissAlert(0);
      });

      expect(result.current.alerts).toHaveLength(1);
      expect(result.current.alerts[0].message).toBe('Alert 1');
    });

    it('should limit alerts to maxAlertsSize', () => {
      const { result } = renderHook(() => useRiskAlerts({ maxAlertsSize: 2 }));

      const subscribeCall = mockWebSocketReturn.subscribe.mock.calls[0];
      const callback = subscribeCall[1];

      // Add 3 alerts
      act(() => {
        callback({ level: 'warning', message: 'Alert 1', metrics: {} as any });
        callback({ level: 'warning', message: 'Alert 2', metrics: {} as any });
        callback({ level: 'warning', message: 'Alert 3', metrics: {} as any });
      });

      expect(result.current.alerts).toHaveLength(2);
      expect(result.current.alerts[0].message).toBe('Alert 3'); // Latest first
      expect(result.current.alerts[1].message).toBe('Alert 2');
    });
  });

  describe('Error handling', () => {
    it('should handle WebSocket errors', () => {
      const error = new Error('WebSocket connection failed');
      mockWebSocketReturn.error = error;
      mockWebSocketReturn.isConnected = false;

      const { result } = renderHook(() => useMarketData());

      expect(result.current.error).toBe(error);
      expect(result.current.isConnected).toBe(false);
    });
  });
});