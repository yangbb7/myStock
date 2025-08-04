import { useEffect, useRef, useCallback, useState } from 'react';
import { WebSocketService, WebSocketEventType, SubscriptionOptions, createWebSocketService, getWebSocketService } from '../services/websocket';
import { useUserStore } from '@/stores/userStore';

export interface UseWebSocketOptions {
  url?: string;
  autoConnect?: boolean;
  reconnection?: boolean;
  reconnectionAttempts?: number;
  reconnectionDelay?: number;
  timeout?: number;
}

export interface UseWebSocketReturn {
  isConnected: boolean;
  connectionState: string;
  error: Error | null;
  subscribe: <T = any>(
    type: WebSocketEventType,
    callback: (data: T) => void,
    options?: SubscriptionOptions
  ) => string;
  unsubscribe: (subscriptionId: string) => void;
  connect: () => Promise<void>;
  disconnect: () => void;
}

export const useWebSocket = (options: UseWebSocketOptions = {}): UseWebSocketReturn => {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionState, setConnectionState] = useState<string>('disconnected');
  const [error, setError] = useState<Error | null>(null);
  
  const serviceRef = useRef<WebSocketService | null>(null);
  const subscriptionsRef = useRef<Set<string>>(new Set());
  const { isAuthenticated } = useUserStore();

  const defaultOptions: UseWebSocketOptions = {
    url: import.meta.env.VITE_WS_BASE_URL || 'http://localhost:8000',
    autoConnect: true,
    reconnection: true,
    reconnectionAttempts: 5,
    reconnectionDelay: 1000,
    timeout: 20000,
  };

  const config = { ...defaultOptions, ...options };

  useEffect(() => {
    // Try to get existing service or create new one
    let service = getWebSocketService();
    
    if (!service) {
      service = createWebSocketService({
        url: config.url!,
        autoConnect: config.autoConnect,
        reconnection: config.reconnection,
        reconnectionAttempts: config.reconnectionAttempts,
        reconnectionDelay: config.reconnectionDelay,
        timeout: config.timeout,
      });
    }

    serviceRef.current = service;

    // Set initial state
    setConnectionState(service.getConnectionState());
    setIsConnected(service.isConnected());

    // Listen to state changes
    const unsubscribeStateChange = service.onStateChange((state: string) => {
      setConnectionState(state);
      setIsConnected(state === 'connected');
      
      if (state === 'error') {
        setError(new Error('WebSocket connection error'));
      } else {
        setError(null);
      }
    });

    return () => {
      unsubscribeStateChange();
      
      // Clean up subscriptions
      subscriptionsRef.current.forEach(subId => {
        service?.unsubscribe(subId);
      });
      subscriptionsRef.current.clear();
    };
  }, [config.url, config.autoConnect, config.reconnection, config.reconnectionAttempts, config.reconnectionDelay, config.timeout]);

  // Update WebSocket authentication when auth state changes
  useEffect(() => {
    if (serviceRef.current) {
      serviceRef.current.updateAuthentication();
    }
  }, [isAuthenticated]);

  const subscribe = useCallback(<T = any>(
    type: WebSocketEventType,
    callback: (data: T) => void,
    subscriptionOptions?: SubscriptionOptions
  ): string => {
    if (!serviceRef.current) {
      throw new Error('WebSocket service not initialized');
    }

    const subscriptionId = serviceRef.current.subscribe(type, callback, subscriptionOptions);
    subscriptionsRef.current.add(subscriptionId);
    
    return subscriptionId;
  }, []);

  const unsubscribe = useCallback((subscriptionId: string): void => {
    if (serviceRef.current) {
      serviceRef.current.unsubscribe(subscriptionId);
      subscriptionsRef.current.delete(subscriptionId);
    }
  }, []);

  const connect = useCallback(async (): Promise<void> => {
    if (!serviceRef.current) {
      throw new Error('WebSocket service not initialized');
    }

    try {
      await serviceRef.current.connect();
      setError(null);
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Connection failed');
      setError(error);
      throw error;
    }
  }, []);

  const disconnect = useCallback((): void => {
    if (serviceRef.current) {
      serviceRef.current.disconnect();
    }
  }, []);

  return {
    isConnected,
    connectionState,
    error,
    subscribe,
    unsubscribe,
    connect,
    disconnect,
  };
};

export default useWebSocket;