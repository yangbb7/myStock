import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { WebSocketService } from '../websocket';
import { io } from 'socket.io-client';

// Mock socket.io-client
vi.mock('socket.io-client', () => ({
  io: vi.fn(),
}));

describe('WebSocketService', () => {
  let mockSocket: any;
  let service: WebSocketService;

  beforeEach(() => {
    mockSocket = {
      connected: false,
      connect: vi.fn(),
      disconnect: vi.fn(),
      on: vi.fn(),
      once: vi.fn(),
      emit: vi.fn(),
    };

    (io as any).mockReturnValue(mockSocket);

    service = new WebSocketService({
      url: 'http://localhost:8000',
      autoConnect: false,
    });
  });

  afterEach(() => {
    service.destroy();
    vi.clearAllMocks();
  });

  describe('Connection Management', () => {
    it('should create WebSocket service with correct config', () => {
      expect(service).toBeDefined();
      expect(service.getConnectionState()).toBe('disconnected');
      expect(service.isConnected()).toBe(false);
    });

    it('should connect to WebSocket server', async () => {
      const connectPromise = service.connect();
      
      // Simulate successful connection
      const connectCallback = mockSocket.once.mock.calls.find(
        (call: any) => call[0] === 'connect'
      )?.[1];
      
      if (connectCallback) {
        connectCallback();
      }

      await expect(connectPromise).resolves.toBeUndefined();
      expect(mockSocket.connect).toHaveBeenCalled();
    });

    it('should handle connection timeout', async () => {
      const connectPromise = service.connect();
      
      // Don't trigger connect callback to simulate timeout
      await expect(connectPromise).rejects.toThrow('Connection timeout');
    }, 10000);

    it('should disconnect from WebSocket server', () => {
      service.disconnect();
      expect(service.getConnectionState()).toBe('disconnected');
    });
  });

  describe('Subscription Management', () => {
    beforeEach(async () => {
      mockSocket.connected = true;
      const connectPromise = service.connect();
      
      const connectCallback = mockSocket.once.mock.calls.find(
        (call: any) => call[0] === 'connect'
      )?.[1];
      
      if (connectCallback) {
        connectCallback();
      }
      
      await connectPromise;
    });

    it('should subscribe to market data', () => {
      const callback = vi.fn();
      const subscriptionId = service.subscribeToMarketData(
        ['AAPL', 'GOOGL'], 
        callback,
        { throttle: 100 }
      );

      expect(subscriptionId).toBeDefined();
      expect(typeof subscriptionId).toBe('string');
      expect(mockSocket.emit).toHaveBeenCalledWith('subscribe', {
        type: 'market_data',
        options: {
          symbols: ['AAPL', 'GOOGL'],
          throttle: 100,
          bufferSize: 10,
        },
      });
    });

    it('should subscribe to system status', () => {
      const callback = vi.fn();
      const subscriptionId = service.subscribeToSystemStatus(callback);

      expect(subscriptionId).toBeDefined();
      expect(mockSocket.emit).toHaveBeenCalledWith('subscribe', {
        type: 'system_status',
        options: undefined,
      });
    });

    it('should unsubscribe from events', () => {
      const callback = vi.fn();
      const subscriptionId = service.subscribeToSystemStatus(callback);
      
      service.unsubscribe(subscriptionId);
      
      expect(mockSocket.emit).toHaveBeenCalledWith('unsubscribe', {
        type: 'system_status',
        subscriptionId,
      });
    });
  });

  describe('Message Handling', () => {
    let callback: any;
    let subscriptionId: string;

    beforeEach(async () => {
      mockSocket.connected = true;
      const connectPromise = service.connect();
      
      const connectCallback = mockSocket.once.mock.calls.find(
        (call: any) => call[0] === 'connect'
      )?.[1];
      
      if (connectCallback) {
        connectCallback();
      }
      
      await connectPromise;

      callback = vi.fn();
      subscriptionId = service.subscribeToMarketData(['AAPL'], callback);
    });

    it('should handle market data messages', () => {
      const marketData = {
        symbol: 'AAPL',
        price: 150.0,
        change: 2.5,
        changePercent: 0.017,
        volume: 1000000,
        timestamp: '2023-01-01T10:00:00Z',
      };

      // Find the market_data event handler
      const marketDataHandler = mockSocket.on.mock.calls.find(
        (call: any) => call[0] === 'market_data'
      )?.[1];

      if (marketDataHandler) {
        marketDataHandler({ data: marketData });
      }

      // Due to throttling, we need to wait
      setTimeout(() => {
        expect(callback).toHaveBeenCalledWith(marketData);
      }, 150);
    });

    it('should filter messages by symbol', () => {
      const marketData1 = {
        symbol: 'AAPL',
        price: 150.0,
        timestamp: '2023-01-01T10:00:00Z',
      };

      const marketData2 = {
        symbol: 'GOOGL',
        price: 2500.0,
        timestamp: '2023-01-01T10:00:00Z',
      };

      const marketDataHandler = mockSocket.on.mock.calls.find(
        (call: any) => call[0] === 'market_data'
      )?.[1];

      if (marketDataHandler) {
        marketDataHandler({ data: marketData1 });
        marketDataHandler({ data: marketData2 });
      }

      setTimeout(() => {
        expect(callback).toHaveBeenCalledWith(marketData1);
        expect(callback).not.toHaveBeenCalledWith(marketData2);
      }, 150);
    });
  });

  describe('Buffering and Throttling', () => {
    it('should buffer and throttle messages', (done) => {
      const callback = vi.fn();
      
      service.subscribeToMarketData(['AAPL'], callback, {
        throttle: 100,
        bufferSize: 5,
      });

      const marketDataHandler = mockSocket.on.mock.calls.find(
        (call: any) => call[0] === 'market_data'
      )?.[1];

      if (marketDataHandler) {
        // Send multiple messages quickly
        for (let i = 0; i < 3; i++) {
          marketDataHandler({
            data: {
              symbol: 'AAPL',
              price: 150 + i,
              timestamp: new Date().toISOString(),
            },
          });
        }
      }

      // Check that callback is called only once after throttle period
      setTimeout(() => {
        expect(callback).toHaveBeenCalledTimes(1);
        // Should receive the latest data
        expect(callback).toHaveBeenCalledWith(
          expect.objectContaining({
            symbol: 'AAPL',
            price: 152,
          })
        );
        done();
      }, 150);
    });
  });

  describe('Error Handling', () => {
    it('should handle connection errors', async () => {
      const stateChangeCallback = vi.fn();
      service.onStateChange(stateChangeCallback);

      // First connect to set up event handlers
      await service.connect().catch(() => {});

      const errorHandler = mockSocket.on.mock.calls.find(
        (call: any) => call[0] === 'connect_error'
      )?.[1];

      if (errorHandler) {
        errorHandler(new Error('Connection failed'));
      }

      expect(stateChangeCallback).toHaveBeenCalledWith('error');
    });

    it('should handle disconnection', async () => {
      const stateChangeCallback = vi.fn();
      service.onStateChange(stateChangeCallback);

      // First connect to set up event handlers
      await service.connect().catch(() => {});

      const disconnectHandler = mockSocket.on.mock.calls.find(
        (call: any) => call[0] === 'disconnect'
      )?.[1];

      if (disconnectHandler) {
        disconnectHandler('io server disconnect');
      }

      expect(stateChangeCallback).toHaveBeenCalledWith('disconnected');
    });
  });
});