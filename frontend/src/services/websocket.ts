import { io, Socket } from 'socket.io-client';
import { 
  WebSocketMessage, 
  MarketDataMessage, 
  SystemStatusMessage, 
  OrderUpdateMessage, 
  RiskAlertMessage,
  MarketData,
  SystemHealth,
  OrderStatus
} from './types';
import { tokenManager } from '@/utils/auth';

export type WebSocketEventType = 
  | 'market_data' 
  | 'system_status' 
  | 'order_update' 
  | 'risk_alert'
  | 'connect'
  | 'disconnect'
  | 'error';

export interface WebSocketConfig {
  url: string;
  autoConnect?: boolean;
  reconnection?: boolean;
  reconnectionAttempts?: number;
  reconnectionDelay?: number;
  timeout?: number;
}

export interface SubscriptionOptions {
  symbols?: string[];
  throttle?: number;
  bufferSize?: number;
}

export interface WebSocketSubscription {
  id: string;
  type: WebSocketEventType;
  options?: SubscriptionOptions;
  callback: (data: any) => void;
  active: boolean;
}

export class WebSocketService {
  private socket: Socket | null = null;
  private subscriptions = new Map<string, WebSocketSubscription>();
  private config: WebSocketConfig;
  private connectionState: 'disconnected' | 'connecting' | 'connected' | 'error' = 'disconnected';
  private reconnectAttempts = 0;
  private messageBuffer = new Map<string, any[]>();
  private bufferTimers = new Map<string, NodeJS.Timeout>();
  private eventListeners = new Map<string, Set<(data: any) => void>>();

  constructor(config: WebSocketConfig) {
    this.config = {
      autoConnect: true,
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
      timeout: 20000,
      ...config,
    };

    if (this.config.autoConnect) {
      this.connect();
    }
  }

  public connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.socket?.connected) {
        resolve();
        return;
      }

      this.connectionState = 'connecting';
      this.notifyStateChange();

      // Get authentication token
      const token = tokenManager.getAccessToken();

      this.socket = io(this.config.url, {
        autoConnect: false,
        reconnection: this.config.reconnection,
        reconnectionAttempts: this.config.reconnectionAttempts,
        reconnectionDelay: this.config.reconnectionDelay,
        timeout: this.config.timeout,
        auth: {
          token: token || undefined
        },
        transportOptions: {
          polling: {
            extraHeaders: {
              'Authorization': token ? `Bearer ${token}` : ''
            }
          }
        }
      });

      this.setupEventHandlers();

      this.socket.connect();

      const connectTimeout = setTimeout(() => {
        this.connectionState = 'error';
        this.notifyStateChange();
        reject(new Error('Connection timeout'));
      }, this.config.timeout);

      this.socket.once('connect', () => {
        clearTimeout(connectTimeout);
        this.connectionState = 'connected';
        this.reconnectAttempts = 0;
        this.notifyStateChange();
        this.resubscribeAll();
        resolve();
      });

      this.socket.once('connect_error', (error) => {
        clearTimeout(connectTimeout);
        this.connectionState = 'error';
        this.notifyStateChange();
        reject(error);
      });
    });
  }

  public disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.connectionState = 'disconnected';
    this.clearAllBuffers();
    this.notifyStateChange();
  }

  public subscribe<T = any>(
    type: WebSocketEventType,
    callback: (data: T) => void,
    options?: SubscriptionOptions
  ): string {
    const id = this.generateSubscriptionId();
    const subscription: WebSocketSubscription = {
      id,
      type,
      callback,
      options,
      active: true,
    };

    this.subscriptions.set(id, subscription);

    if (this.socket?.connected) {
      this.activateSubscription(subscription);
    }

    return id;
  }

  public unsubscribe(subscriptionId: string): void {
    const subscription = this.subscriptions.get(subscriptionId);
    if (subscription) {
      subscription.active = false;
      this.deactivateSubscription(subscription);
      this.subscriptions.delete(subscriptionId);
    }
  }

  public subscribeToMarketData(
    symbols: string[],
    callback: (data: MarketData) => void,
    options?: { throttle?: number; bufferSize?: number }
  ): string {
    return this.subscribe('market_data', callback, {
      symbols,
      throttle: options?.throttle || 100, // 100ms throttle by default
      bufferSize: options?.bufferSize || 10,
    });
  }

  public subscribeToSystemStatus(
    callback: (data: SystemHealth) => void
  ): string {
    return this.subscribe('system_status', callback);
  }

  public subscribeToOrderUpdates(
    callback: (data: OrderStatus) => void
  ): string {
    return this.subscribe('order_update', callback);
  }

  public subscribeToRiskAlerts(
    callback: (data: any) => void
  ): string {
    return this.subscribe('risk_alert', callback);
  }

  public getConnectionState(): string {
    return this.connectionState;
  }

  public isConnected(): boolean {
    return this.connectionState === 'connected' && this.socket?.connected === true;
  }

  public onStateChange(callback: (state: string) => void): () => void {
    const listeners = this.eventListeners.get('stateChange') || new Set();
    listeners.add(callback);
    this.eventListeners.set('stateChange', listeners);

    return () => {
      listeners.delete(callback);
    };
  }

  private setupEventHandlers(): void {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      this.connectionState = 'connected';
      this.reconnectAttempts = 0;
      this.notifyStateChange();
    });

    this.socket.on('disconnect', (reason) => {
      this.connectionState = 'disconnected';
      this.notifyStateChange();
      
      if (reason === 'io server disconnect') {
        // Server initiated disconnect, try to reconnect
        this.handleReconnection();
      }
    });

    this.socket.on('connect_error', (error: any) => {
      this.connectionState = 'error';
      this.notifyStateChange();
      
      // Handle authentication errors
      if (error.type === 'auth' || error.message?.includes('auth') || error.message?.includes('unauthorized')) {
        console.error('WebSocket authentication failed:', error);
        // Don't attempt reconnection for auth errors
        return;
      }
      
      this.handleReconnection();
    });

    // Handle different message types
    this.socket.on('market_data', (message: MarketDataMessage) => {
      this.handleMessage('market_data', message.data);
    });

    this.socket.on('system_status', (message: SystemStatusMessage) => {
      this.handleMessage('system_status', message.data);
    });

    this.socket.on('order_update', (message: OrderUpdateMessage) => {
      this.handleMessage('order_update', message.data);
    });

    this.socket.on('risk_alert', (message: RiskAlertMessage) => {
      this.handleMessage('risk_alert', message.data);
    });
  }

  private handleMessage(type: WebSocketEventType, data: any): void {
    const relevantSubscriptions = Array.from(this.subscriptions.values())
      .filter(sub => sub.type === type && sub.active);

    for (const subscription of relevantSubscriptions) {
      if (this.shouldProcessMessage(subscription, data)) {
        if (subscription.options?.throttle || subscription.options?.bufferSize) {
          this.bufferMessage(subscription, data);
        } else {
          subscription.callback(data);
        }
      }
    }
  }

  private shouldProcessMessage(subscription: WebSocketSubscription, data: any): boolean {
    // Filter by symbols if specified
    if (subscription.options?.symbols && data.symbol) {
      return subscription.options.symbols.includes(data.symbol);
    }
    return true;
  }

  private bufferMessage(subscription: WebSocketSubscription, data: any): void {
    const bufferId = subscription.id;
    
    if (!this.messageBuffer.has(bufferId)) {
      this.messageBuffer.set(bufferId, []);
    }

    const buffer = this.messageBuffer.get(bufferId)!;
    buffer.push(data);

    // Remove duplicates for market data
    if (subscription.type === 'market_data' && data.symbol) {
      const filtered = buffer.filter((item, index, arr) => {
        // Find the last occurrence of this symbol
        for (let i = arr.length - 1; i >= 0; i--) {
          if (arr[i].symbol === item.symbol) {
            return i === index;
          }
        }
        return false;
      });
      this.messageBuffer.set(bufferId, filtered);
    }

    // Limit buffer size
    const maxSize = subscription.options?.bufferSize || 10;
    if (buffer.length > maxSize) {
      buffer.splice(0, buffer.length - maxSize);
    }

    // Setup or reset throttle timer
    if (this.bufferTimers.has(bufferId)) {
      clearTimeout(this.bufferTimers.get(bufferId)!);
    }

    const throttleDelay = subscription.options?.throttle || 100;
    const timer = setTimeout(() => {
      this.flushBuffer(subscription);
    }, throttleDelay);

    this.bufferTimers.set(bufferId, timer);
  }

  private flushBuffer(subscription: WebSocketSubscription): void {
    const bufferId = subscription.id;
    const buffer = this.messageBuffer.get(bufferId);
    
    if (buffer && buffer.length > 0) {
      // For market data, send only the latest data for each symbol
      if (subscription.type === 'market_data') {
        const latestData = new Map();
        buffer.forEach(item => {
          if (item.symbol) {
            latestData.set(item.symbol, item);
          }
        });
        Array.from(latestData.values()).forEach(data => {
          subscription.callback(data);
        });
      } else {
        // For other types, send all buffered messages
        buffer.forEach(data => subscription.callback(data));
      }
      
      this.messageBuffer.set(bufferId, []);
    }

    this.bufferTimers.delete(bufferId);
  }

  private activateSubscription(subscription: WebSocketSubscription): void {
    if (!this.socket?.connected) return;

    // Send subscription request to server
    this.socket.emit('subscribe', {
      type: subscription.type,
      options: subscription.options,
    });
  }

  private deactivateSubscription(subscription: WebSocketSubscription): void {
    if (!this.socket?.connected) return;

    // Send unsubscription request to server
    this.socket.emit('unsubscribe', {
      type: subscription.type,
      subscriptionId: subscription.id,
    });

    // Clear buffers
    this.clearBuffer(subscription.id);
  }

  private resubscribeAll(): void {
    for (const subscription of this.subscriptions.values()) {
      if (subscription.active) {
        this.activateSubscription(subscription);
      }
    }
  }

  private handleReconnection(): void {
    if (this.reconnectAttempts >= (this.config.reconnectionAttempts || 5)) {
      return;
    }

    this.reconnectAttempts++;
    const delay = (this.config.reconnectionDelay || 1000) * this.reconnectAttempts;

    setTimeout(() => {
      if (this.connectionState !== 'connected') {
        this.connect().catch(() => {
          // Reconnection failed, will try again if attempts remain
        });
      }
    }, delay);
  }

  private clearBuffer(subscriptionId: string): void {
    this.messageBuffer.delete(subscriptionId);
    const timer = this.bufferTimers.get(subscriptionId);
    if (timer) {
      clearTimeout(timer);
      this.bufferTimers.delete(subscriptionId);
    }
  }

  private clearAllBuffers(): void {
    this.messageBuffer.clear();
    for (const timer of this.bufferTimers.values()) {
      clearTimeout(timer);
    }
    this.bufferTimers.clear();
  }

  private generateSubscriptionId(): string {
    return `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private notifyStateChange(): void {
    const listeners = this.eventListeners.get('stateChange');
    if (listeners) {
      listeners.forEach(callback => callback(this.connectionState));
    }
  }

  public updateAuthentication(): void {
    if (this.socket) {
      const token = tokenManager.getAccessToken();
      
      // Update auth for future reconnections
      this.socket.auth = {
        token: token || undefined
      };
      
      // If connected, disconnect and reconnect with new auth
      if (this.socket.connected) {
        this.socket.disconnect();
        this.connect().catch(error => {
          console.error('Failed to reconnect with new authentication:', error);
        });
      }
    }
  }

  public destroy(): void {
    this.clearAllBuffers();
    this.subscriptions.clear();
    this.eventListeners.clear();
    this.disconnect();
  }
}

// Singleton instance
let webSocketService: WebSocketService | null = null;

export const createWebSocketService = (config: WebSocketConfig): WebSocketService => {
  if (webSocketService) {
    webSocketService.destroy();
  }
  webSocketService = new WebSocketService(config);
  return webSocketService;
};

export const getWebSocketService = (): WebSocketService | null => {
  return webSocketService;
};

export default WebSocketService;