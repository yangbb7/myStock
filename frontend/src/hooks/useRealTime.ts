import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocket } from './useWebSocket';
import { 
  MarketData, 
  SystemHealth, 
  OrderStatus, 
  RiskMetrics 
} from '../services/types';

// Market Data Hook
export interface UseMarketDataOptions {
  symbols?: string[];
  throttle?: number;
  bufferSize?: number;
  autoSubscribe?: boolean;
}

export interface UseMarketDataReturn {
  data: Record<string, MarketData>;
  isConnected: boolean;
  error: Error | null;
  subscribe: (symbols: string[]) => void;
  unsubscribe: () => void;
  getSymbolData: (symbol: string) => MarketData | undefined;
}

export const useMarketData = (options: UseMarketDataOptions = {}): UseMarketDataReturn => {
  const [data, setData] = useState<Record<string, MarketData>>({});
  const subscriptionIdRef = useRef<string | null>(null);
  
  const { 
    symbols = [], 
    throttle = 100, 
    bufferSize = 10, 
    autoSubscribe = true 
  } = options;

  const { isConnected, error, subscribe: wsSubscribe, unsubscribe: wsUnsubscribe } = useWebSocket();

  const handleMarketDataUpdate = useCallback((marketData: MarketData) => {
    setData(prevData => ({
      ...prevData,
      [marketData.symbol]: {
        ...prevData[marketData.symbol],
        ...marketData,
        timestamp: new Date().toISOString(),
      }
    }));
  }, []);

  const subscribe = useCallback((symbolsToSubscribe: string[]) => {
    if (subscriptionIdRef.current) {
      wsUnsubscribe(subscriptionIdRef.current);
    }

    if (symbolsToSubscribe.length > 0) {
      subscriptionIdRef.current = wsSubscribe(
        'market_data',
        handleMarketDataUpdate,
        {
          symbols: symbolsToSubscribe,
          throttle,
          bufferSize,
        }
      );
    }
  }, [wsSubscribe, wsUnsubscribe, handleMarketDataUpdate, throttle, bufferSize]);

  const unsubscribe = useCallback(() => {
    if (subscriptionIdRef.current) {
      wsUnsubscribe(subscriptionIdRef.current);
      subscriptionIdRef.current = null;
    }
  }, [wsUnsubscribe]);

  const getSymbolData = useCallback((symbol: string): MarketData | undefined => {
    return data[symbol];
  }, [data]);

  useEffect(() => {
    if (autoSubscribe && symbols.length > 0 && isConnected) {
      subscribe(symbols);
    }

    return () => {
      unsubscribe();
    };
  }, [symbols, isConnected, autoSubscribe, subscribe, unsubscribe]);

  return {
    data,
    isConnected,
    error,
    subscribe,
    unsubscribe,
    getSymbolData,
  };
};

// System Status Hook
export interface UseSystemStatusOptions {
  autoSubscribe?: boolean;
  pollInterval?: number;
}

export interface UseSystemStatusReturn {
  data: SystemHealth | null;
  isConnected: boolean;
  error: Error | null;
  subscribe: () => void;
  unsubscribe: () => void;
  lastUpdated: Date | null;
}

export const useSystemStatus = (options: UseSystemStatusOptions = {}): UseSystemStatusReturn => {
  const [data, setData] = useState<SystemHealth | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const subscriptionIdRef = useRef<string | null>(null);
  
  const { autoSubscribe = true } = options;

  const { isConnected, error, subscribe: wsSubscribe, unsubscribe: wsUnsubscribe } = useWebSocket();

  const handleSystemStatusUpdate = useCallback((systemHealth: SystemHealth) => {
    setData(systemHealth);
    setLastUpdated(new Date());
  }, []);

  const subscribe = useCallback(() => {
    if (subscriptionIdRef.current) {
      wsUnsubscribe(subscriptionIdRef.current);
    }

    subscriptionIdRef.current = wsSubscribe(
      'system_status',
      handleSystemStatusUpdate
    );
  }, [wsSubscribe, wsUnsubscribe, handleSystemStatusUpdate]);

  const unsubscribe = useCallback(() => {
    if (subscriptionIdRef.current) {
      wsUnsubscribe(subscriptionIdRef.current);
      subscriptionIdRef.current = null;
    }
  }, [wsUnsubscribe]);

  useEffect(() => {
    if (autoSubscribe && isConnected) {
      subscribe();
    }

    return () => {
      unsubscribe();
    };
  }, [isConnected, autoSubscribe, subscribe, unsubscribe]);

  return {
    data,
    isConnected,
    error,
    subscribe,
    unsubscribe,
    lastUpdated,
  };
};

// Order Updates Hook
export interface UseOrderUpdatesOptions {
  autoSubscribe?: boolean;
  maxHistorySize?: number;
}

export interface UseOrderUpdatesReturn {
  currentOrders: Record<string, OrderStatus>;
  orderHistory: OrderStatus[];
  isConnected: boolean;
  error: Error | null;
  subscribe: () => void;
  unsubscribe: () => void;
  getOrderById: (orderId: string) => OrderStatus | undefined;
  clearHistory: () => void;
}

export const useOrderUpdates = (options: UseOrderUpdatesOptions = {}): UseOrderUpdatesReturn => {
  const [currentOrders, setCurrentOrders] = useState<Record<string, OrderStatus>>({});
  const [orderHistory, setOrderHistory] = useState<OrderStatus[]>([]);
  const subscriptionIdRef = useRef<string | null>(null);
  
  const { autoSubscribe = true, maxHistorySize = 100 } = options;

  const { isConnected, error, subscribe: wsSubscribe, unsubscribe: wsUnsubscribe } = useWebSocket();

  const handleOrderUpdate = useCallback((orderStatus: OrderStatus) => {
    setCurrentOrders(prevOrders => ({
      ...prevOrders,
      [orderStatus.orderId]: orderStatus,
    }));

    setOrderHistory(prevHistory => {
      const newHistory = [orderStatus, ...prevHistory];
      return newHistory.slice(0, maxHistorySize);
    });
  }, [maxHistorySize]);

  const subscribe = useCallback(() => {
    if (subscriptionIdRef.current) {
      wsUnsubscribe(subscriptionIdRef.current);
    }

    subscriptionIdRef.current = wsSubscribe(
      'order_update',
      handleOrderUpdate
    );
  }, [wsSubscribe, wsUnsubscribe, handleOrderUpdate]);

  const unsubscribe = useCallback(() => {
    if (subscriptionIdRef.current) {
      wsUnsubscribe(subscriptionIdRef.current);
      subscriptionIdRef.current = null;
    }
  }, [wsUnsubscribe]);

  const getOrderById = useCallback((orderId: string): OrderStatus | undefined => {
    return currentOrders[orderId];
  }, [currentOrders]);

  const clearHistory = useCallback(() => {
    setOrderHistory([]);
  }, []);

  useEffect(() => {
    if (autoSubscribe && isConnected) {
      subscribe();
    }

    return () => {
      unsubscribe();
    };
  }, [isConnected, autoSubscribe, subscribe, unsubscribe]);

  return {
    currentOrders,
    orderHistory,
    isConnected,
    error,
    subscribe,
    unsubscribe,
    getOrderById,
    clearHistory,
  };
};

// Risk Alerts Hook
export interface RiskAlert {
  level: 'warning' | 'error' | 'critical';
  message: string;
  metrics: RiskMetrics;
  timestamp: string;
}

export interface UseRiskAlertsOptions {
  autoSubscribe?: boolean;
  maxAlertsSize?: number;
}

export interface UseRiskAlertsReturn {
  alerts: RiskAlert[];
  latestAlert: RiskAlert | null;
  isConnected: boolean;
  error: Error | null;
  subscribe: () => void;
  unsubscribe: () => void;
  clearAlerts: () => void;
  dismissAlert: (index: number) => void;
}

export const useRiskAlerts = (options: UseRiskAlertsOptions = {}): UseRiskAlertsReturn => {
  const [alerts, setAlerts] = useState<RiskAlert[]>([]);
  const [latestAlert, setLatestAlert] = useState<RiskAlert | null>(null);
  const subscriptionIdRef = useRef<string | null>(null);
  
  const { autoSubscribe = true, maxAlertsSize = 50 } = options;

  const { isConnected, error, subscribe: wsSubscribe, unsubscribe: wsUnsubscribe } = useWebSocket();

  const handleRiskAlert = useCallback((alertData: RiskAlert) => {
    const alert: RiskAlert = {
      ...alertData,
      timestamp: new Date().toISOString(),
    };

    setLatestAlert(alert);
    setAlerts(prevAlerts => {
      const newAlerts = [alert, ...prevAlerts];
      return newAlerts.slice(0, maxAlertsSize);
    });
  }, [maxAlertsSize]);

  const subscribe = useCallback(() => {
    if (subscriptionIdRef.current) {
      wsUnsubscribe(subscriptionIdRef.current);
    }

    subscriptionIdRef.current = wsSubscribe(
      'risk_alert',
      handleRiskAlert
    );
  }, [wsSubscribe, wsUnsubscribe, handleRiskAlert]);

  const unsubscribe = useCallback(() => {
    if (subscriptionIdRef.current) {
      wsUnsubscribe(subscriptionIdRef.current);
      subscriptionIdRef.current = null;
    }
  }, [wsUnsubscribe]);

  const clearAlerts = useCallback(() => {
    setAlerts([]);
    setLatestAlert(null);
  }, []);

  const dismissAlert = useCallback((index: number) => {
    setAlerts(prevAlerts => prevAlerts.filter((_, i) => i !== index));
  }, []);

  useEffect(() => {
    if (autoSubscribe && isConnected) {
      subscribe();
    }

    return () => {
      unsubscribe();
    };
  }, [isConnected, autoSubscribe, subscribe, unsubscribe]);

  return {
    alerts,
    latestAlert,
    isConnected,
    error,
    subscribe,
    unsubscribe,
    clearAlerts,
    dismissAlert,
  };
};

// Combined Real-time Hook
export interface UseRealTimeOptions {
  marketData?: UseMarketDataOptions;
  systemStatus?: UseSystemStatusOptions;
  orderUpdates?: UseOrderUpdatesOptions;
  riskAlerts?: UseRiskAlertsOptions;
}

export interface UseRealTimeReturn {
  marketData: UseMarketDataReturn;
  systemStatus: UseSystemStatusReturn;
  orderUpdates: UseOrderUpdatesReturn;
  riskAlerts: UseRiskAlertsReturn;
  isConnected: boolean;
  error: Error | null;
}

export const useRealTime = (options: UseRealTimeOptions = {}): UseRealTimeReturn => {
  const marketData = useMarketData(options.marketData);
  const systemStatus = useSystemStatus(options.systemStatus);
  const orderUpdates = useOrderUpdates(options.orderUpdates);
  const riskAlerts = useRiskAlerts(options.riskAlerts);

  // Use the connection state from any of the hooks (they should all be the same)
  const isConnected = marketData.isConnected;
  const error = marketData.error || systemStatus.error || orderUpdates.error || riskAlerts.error;

  return {
    marketData,
    systemStatus,
    orderUpdates,
    riskAlerts,
    isConnected,
    error,
  };
};