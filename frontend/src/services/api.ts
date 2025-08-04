import {
  ApiResponse,
  SystemHealth,
  SystemMetrics,
  SystemControlRequest,
  SystemControlResponse,
  PortfolioSummary,
  RiskMetrics,
  StrategyConfig,
  StrategyPerformance,
  OrderRequest,
  OrderStatus,
  MarketDataResponse,
  TickData,
  PerformanceMetrics,
  BacktestConfig,
  BacktestResult,
  ApiError,
  PaginationParams,
  OrderFilter,
  DateRangeFilter,
} from './types';
import request from '@/utils/request';
import { AxiosResponse } from 'axios';

// Helper function to generate request ID
const generateRequestId = (): string => {
  return `${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
};

// Helper function to handle API response
const formatApiResponse = <T>(response: AxiosResponse<T>): T => {
  return response.data;
};

// Helper function to build query parameters
const buildQueryParams = (params: Record<string, any>): string => {
  const queryParams = new URLSearchParams();
  
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      if (Array.isArray(value)) {
        value.forEach(v => queryParams.append(key, String(v)));
      } else {
        queryParams.append(key, String(value));
      }
    }
  });
  
  return queryParams.toString();
};

// Helper function to handle errors
const handleApiError = (error: any): never => {
  const apiError: ApiError = {
    message: error.response?.data?.message || error.message || 'An unknown error occurred',
    code: error.response?.data?.code || error.code || 'UNKNOWN_ERROR',
    details: error.response?.data?.details || {},
    timestamp: new Date().toISOString(),
  };
  
  throw apiError;
};

// Format system health response to handle different formats
const formatSystemHealthResponse = (response: any): SystemHealth => {
  console.log('[API] formatSystemHealthResponse input:', response);
  
  if (!response) {
    throw new Error('No system health data received');
  }

  // The response from the backend is wrapped in a SystemResponse object
  // Extract the actual health data from the data property
  const healthData = response.data || response;
  
  console.log('[API] Raw health data:', healthData);
  
  const formattedResponse = {
    systemRunning: healthData.system_running ?? healthData.systemRunning ?? false,
    uptimeSeconds: healthData.uptime_seconds ?? healthData.uptimeSeconds ?? 0,
    modules: healthData.modules || {},
    timestamp: healthData.timestamp || new Date().toISOString(),
  };
  
  console.log('[API] Formatted health response:', formattedResponse);
  return formattedResponse;
};

// System Management API
export const systemApi = {
  // Get system health status
  getHealth: async (): Promise<SystemHealth> => {
    const response = await request.get('/health');
    console.log('[API] Health response:', response);
    console.log('[API] Health response.data:', response.data);
    return formatSystemHealthResponse(response.data);
  },

  // Get system metrics
  getMetrics: async (): Promise<SystemMetrics> => {
    const response = await request.get('/metrics');
    console.log('[API] Raw metrics data:', response.data);
    
    // Handle the wrapped response format
    const metricsData = response.data.data || response.data;
    
    const formattedResponse: SystemMetrics = {
      system: {
        running: metricsData.system?.running ?? false,
        uptime: metricsData.system?.uptime ?? 0,
        modulesCount: metricsData.system?.modules_count ?? metricsData.system?.modulesCount ?? 0,
      },
      data: metricsData.data || {},
      strategy: metricsData.strategy || {},
      execution: metricsData.execution || {},
      risk: metricsData.risk || {},
      portfolio: metricsData.portfolio || {},
      analytics: metricsData.analytics || {},
      portfolioSummary: metricsData.portfolio_summary || metricsData.portfolioSummary,
      riskMetrics: metricsData.risk_metrics || metricsData.riskMetrics,
      performance: metricsData.performance || {},
    };
    
    console.log('[API] Formatted metrics response:', formattedResponse);
    return formattedResponse;
  },

  // Start system
  startSystem: async (modules?: string[]): Promise<SystemControlResponse> => {
    console.log('[API] Starting system with modules:', modules);
    
    try {
      const response = await request.post<SystemControlResponse>('/system/start', {});
      console.log('[API] System start response:', response.data);
      
      return {
        success: true,
        message: response.data?.message || 'System started successfully',
        timestamp: new Date().toISOString(),
        ...response.data
      };
    } catch (error) {
      console.error('[API] System start failed:', error);
      throw error;
    }
  },

  // Stop system
  stopSystem: async (modules?: string[]): Promise<SystemControlResponse> => {
    console.log('[API] Stopping system with modules:', modules);
    
    try {
      const response = await request.post<SystemControlResponse>('/system/stop', {});
      console.log('[API] System stop response:', response.data);
      
      return {
        success: true,
        message: response.data?.message || 'System stopped successfully',
        timestamp: new Date().toISOString(),
        ...response.data
      };
    } catch (error) {
      console.error('[API] System stop failed:', error);
      throw error;
    }
  },

  // Restart system
  restartSystem: async (modules?: string[]): Promise<SystemControlResponse> => {
    console.log('[API] Restarting system with modules:', modules);
    
    try {
      const response = await request.post<SystemControlResponse>('/system/restart', {});
      console.log('[API] System restart response:', response.data);
      
      return {
        success: true,
        message: response.data?.message || 'System restarted successfully',
        timestamp: new Date().toISOString(),
        ...response.data
      };
    } catch (error) {
      console.error('[API] System restart failed:', error);
      throw error;
    }
  },
};

// Data Management API
export const dataApi = {
  // Get market data for a symbol
  getMarketData: async (symbol: string, timeframe?: string, startDate?: string, endDate?: string): Promise<MarketDataResponse> => {
    const response = await request.get<MarketDataResponse>(`/data/market/${symbol}`, {
      params: { 
        period: timeframe || '1d',
        start_date: startDate,
        end_date: endDate
      },
    });
    return formatApiResponse(response);
  },

  // Get available symbols
  getSymbols: async (): Promise<string[]> => {
    const response = await request.get<ApiResponse<string[]>>('/data/symbols');
    return formatApiResponse(response);
  },

  // Process tick data
  processTick: async (tickData: TickData): Promise<ApiResponse<void>> => {
    const response = await request.post<ApiResponse<void>>('/data/tick', tickData);
    return formatApiResponse(response);
  },

  // Get data processing status
  getDataStatus: async (): Promise<any> => {
    const response = await request.get('/data/status');
    return formatApiResponse(response);
  },

  // Get stock info
  getStockInfo: async (symbol: string): Promise<any> => {
    const response = await request.get(`/stock/info/${symbol}`);
    return formatApiResponse(response);
  },

  // Search stocks
  searchStocks: async (keyword: string = '', limit: number = 20): Promise<any> => {
    const response = await request.get(`/stock/search?keyword=${encodeURIComponent(keyword)}&limit=${limit}`);
    return formatApiResponse(response);
  },

  // Get real-time price for a symbol
  getRealTimePrice: async (symbol: string): Promise<any> => {
    const response = await request.get(`/data/realtime/${symbol}`);
    return formatApiResponse(response);
  },

  // Get real-time prices for multiple symbols (batch)
  getRealTimePricesBatch: async (symbols: string[]): Promise<any> => {
    const response = await request.post(`/data/realtime/batch`, symbols);
    return formatApiResponse(response);
  },
};

// Strategy Management API
export const strategyApi = {
  // List all strategies (alias for getStrategies)
  getStrategies: async (): Promise<string[]> => {
    const response = await request.get<ApiResponse<string[]>>('/strategy/list');
    return formatApiResponse(response);
  },

  // List all strategies
  listStrategies: async (): Promise<ApiResponse<StrategyConfig[]>> => {
    const response = await request.get<ApiResponse<StrategyConfig[]>>('/strategy/list');
    return formatApiResponse(response);
  },

  // Get strategy by ID
  getStrategy: async (strategyId: string): Promise<ApiResponse<StrategyConfig>> => {
    const response = await request.get<ApiResponse<StrategyConfig>>(`/strategy/${strategyId}`);
    return formatApiResponse(response);
  },

  // Get strategy config by name
  getStrategyConfig: async (strategyName: string): Promise<StrategyConfig> => {
    const response = await request.get<ApiResponse<StrategyConfig>>(`/strategy/${strategyName}/config`);
    return formatApiResponse(response);
  },

  // Add new strategy
  addStrategy: async (config: StrategyConfig): Promise<ApiResponse<StrategyConfig>> => {
    const response = await request.post<ApiResponse<StrategyConfig>>('/strategy/create', config);
    return formatApiResponse(response);
  },

  // Create new strategy (alias for addStrategy)
  createStrategy: async (config: StrategyConfig): Promise<ApiResponse<StrategyConfig>> => {
    const response = await request.post<ApiResponse<StrategyConfig>>('/strategy/create', config);
    return formatApiResponse(response);
  },

  // Update strategy
  updateStrategy: async (strategyName: string, config: Partial<StrategyConfig>): Promise<ApiResponse<StrategyConfig>> => {
    const response = await request.put<ApiResponse<StrategyConfig>>(`/strategy/${strategyName}`, config);
    return formatApiResponse(response);
  },

  // Delete strategy
  deleteStrategy: async (strategyName: string): Promise<ApiResponse<void>> => {
    const response = await request.delete<ApiResponse<void>>(`/strategy/${strategyName}`);
    return formatApiResponse(response);
  },

  // Start strategy
  startStrategy: async (strategyName: string): Promise<ApiResponse<void>> => {
    const response = await request.post<ApiResponse<void>>(`/strategy/${strategyName}/start`);
    return formatApiResponse(response);
  },

  // Stop strategy
  stopStrategy: async (strategyName: string): Promise<ApiResponse<void>> => {
    const response = await request.post<ApiResponse<void>>(`/strategy/${strategyName}/stop`);
    return formatApiResponse(response);
  },

  // Get strategy performance by ID
  getPerformance: async (): Promise<StrategyPerformance> => {
    const response = await request.get<ApiResponse<StrategyPerformance>>('/strategy/performance');
    return formatApiResponse(response);
  },

  // Get all strategies performance
  getAllPerformance: async (): Promise<ApiResponse<StrategyPerformance>> => {
    const response = await request.get<ApiResponse<StrategyPerformance>>('/strategy/performance');
    return formatApiResponse(response);
  },
};

// Order Management API
export const orderApi = {
  // Create order
  createOrder: async (order: OrderRequest): Promise<ApiResponse<OrderStatus>> => {
    const response = await request.post<ApiResponse<OrderStatus>>('/order/create', order);
    return formatApiResponse(response);
  },

  // Get order status
  getOrderStatus: async (orderId: string): Promise<OrderStatus> => {
    const response = await request.get<ApiResponse<OrderStatus>>(`/order/${orderId}`);
    return formatApiResponse(response);
  },

  // Get order history
  getOrderHistory: async (filter?: OrderFilter & PaginationParams): Promise<OrderStatus[]> => {
    const queryString = filter ? buildQueryParams(filter) : '';
    const response = await request.get<ApiResponse<OrderStatus[]>>(`/order/history${queryString ? `?${queryString}` : ''}`);
    return formatApiResponse(response);
  },

  // Get active orders
  getActiveOrders: async (): Promise<OrderStatus[]> => {
    const response = await request.get<ApiResponse<OrderStatus[]>>('/order/active');
    return formatApiResponse(response);
  },

  // Get order statistics
  getOrderStats: async (filter?: DateRangeFilter): Promise<Record<string, any>> => {
    const queryString = filter ? buildQueryParams(filter) : '';
    const response = await request.get<ApiResponse<Record<string, any>>>(`/order/stats${queryString ? `?${queryString}` : ''}`);
    return formatApiResponse(response);
  },

  // List orders with filters
  listOrders: async (filters?: OrderFilter & PaginationParams): Promise<ApiResponse<OrderStatus[]>> => {
    const queryString = filters ? buildQueryParams(filters) : '';
    const response = await request.get<ApiResponse<OrderStatus[]>>(`/order/list${queryString ? `?${queryString}` : ''}`);
    return formatApiResponse(response);
  },

  // Cancel order
  cancelOrder: async (orderId: string): Promise<ApiResponse<void>> => {
    const response = await request.post<ApiResponse<void>>(`/order/${orderId}/cancel`);
    return formatApiResponse(response);
  },
};

// Portfolio Management API
export const portfolioApi = {
  // Get portfolio summary
  getSummary: async (): Promise<PortfolioSummary> => {
    const response = await request.get('/portfolio/summary');
    console.log('[API] Raw portfolio data:', response.data);
    
    // Handle the wrapped response format
    const portfolioData = response.data.data || response.data;
    
    const formattedResponse: PortfolioSummary = {
      totalValue: portfolioData.total_value ?? portfolioData.totalValue ?? 0,
      cashBalance: portfolioData.cash_balance ?? portfolioData.cashBalance ?? 0,
      positions: portfolioData.positions ?? {},
      unrealizedPnl: portfolioData.unrealized_pnl ?? portfolioData.unrealizedPnl ?? 0,
      positionsCount: portfolioData.positions_count ?? portfolioData.positionsCount ?? 0,
      timestamp: portfolioData.timestamp || new Date().toISOString(),
    };
    
    console.log('[API] Formatted portfolio response:', formattedResponse);
    return formattedResponse;
  },

  // Get portfolio history
  getHistory: async (filter?: DateRangeFilter): Promise<any[]> => {
    const queryString = filter ? buildQueryParams(filter) : '';
    const response = await request.get<ApiResponse<any[]>>(`/portfolio/history${queryString ? `?${queryString}` : ''}`);
    return formatApiResponse(response);
  },

  // Get portfolio positions
  getPositions: async (): Promise<Record<string, any>> => {
    const response = await request.get<ApiResponse<Record<string, any>>>('/portfolio/positions');
    return formatApiResponse(response);
  },

  // Get portfolio performance
  getPerformance: async (filter?: DateRangeFilter): Promise<PerformanceMetrics> => {
    const queryString = filter ? buildQueryParams(filter) : '';
    const response = await request.get<ApiResponse<PerformanceMetrics>>(`/portfolio/performance${queryString ? `?${queryString}` : ''}`);
    return formatApiResponse(response);
  },
};

// Risk Management API
export const riskApi = {
  // Get risk metrics
  getMetrics: async (): Promise<RiskMetrics> => {
    const response = await request.get('/risk/metrics');
    console.log('[API] Raw risk data:', response.data);
    
    // Handle the wrapped response format
    const riskData = response.data.data || response.data;
    
    const formattedResponse: RiskMetrics = {
      dailyPnl: riskData.daily_pnl ?? riskData.dailyPnl ?? 0,
      currentDrawdown: riskData.current_drawdown ?? riskData.currentDrawdown ?? 0,
      riskLimits: riskData.risk_limits ?? riskData.riskLimits ?? {
        maxPositionSize: 0.1,
        maxDrawdownLimit: 0.2,
        maxDailyLoss: 0.05
      },
      riskUtilization: riskData.risk_utilization ?? riskData.riskUtilization ?? {
        dailyLossRatio: 0.0,
        drawdownRatio: 0.0
      },
      timestamp: riskData.timestamp || new Date().toISOString(),
    };
    
    console.log('[API] Formatted risk response:', formattedResponse);
    return formattedResponse;
  },

  // Get risk alerts
  getAlerts: async (filter?: DateRangeFilter): Promise<any[]> => {
    const queryString = filter ? buildQueryParams(filter) : '';
    const response = await request.get<ApiResponse<any[]>>(`/risk/alerts${queryString ? `?${queryString}` : ''}`);
    return formatApiResponse(response);
  },

  // Update risk parameters
  updateParameters: async (params: Partial<RiskMetrics>): Promise<ApiResponse<RiskMetrics>> => {
    const response = await request.put<ApiResponse<RiskMetrics>>('/risk/parameters', params);
    return formatApiResponse(response);
  },
  
  // Get risk configuration
  getConfig: async (): Promise<any> => {
    const response = await request.get('/risk/config');
    return formatApiResponse(response);
  },
  
  // Update risk limits
  updateLimits: async (limits: any): Promise<any> => {
    const response = await request.put('/risk/limits', limits);
    return formatApiResponse(response);
  },
};

// Performance Analytics API
export const analyticsApi = {
  // Get performance analytics
  getPerformance: async (filter?: DateRangeFilter): Promise<PerformanceMetrics> => {
    const queryString = filter ? buildQueryParams(filter) : '';
    const response = await request.get<ApiResponse<PerformanceMetrics>>(`/analytics/performance${queryString ? `?${queryString}` : ''}`);
    return formatApiResponse(response);
  },

  // Get backtest history
  getBacktestHistory: async (): Promise<any[]> => {
    const response = await request.get<ApiResponse<any[]>>('/analytics/backtest/history');
    return formatApiResponse(response);
  },

  // Run backtest
  runBacktest: async (config: any): Promise<ApiResponse<any>> => {
    const response = await request.post<ApiResponse<any>>('/analytics/backtest/run', config);
    return formatApiResponse(response);
  },

  // Generate report
  generateReport: async (type: string, filter?: DateRangeFilter): Promise<ApiResponse<any>> => {
    const queryString = filter ? buildQueryParams(filter) : '';
    const response = await request.post<ApiResponse<any>>(`/analytics/report/${type}${queryString ? `?${queryString}` : ''}`);
    return formatApiResponse(response);
  },

  // Export data
  exportData: async (type: string, format: string, filter?: any): Promise<ApiResponse<any>> => {
    const queryString = filter ? buildQueryParams(filter) : '';
    const response = await request.post<ApiResponse<any>>(`/analytics/export/${type}/${format}${queryString ? `?${queryString}` : ''}`);
    return formatApiResponse(response);
  },
};

// Backtesting API
export const backtestApi = {
  // Run backtest
  runBacktest: async (config: BacktestConfig): Promise<ApiResponse<BacktestResult>> => {
    const response = await request.post<ApiResponse<BacktestResult>>('/backtest/run', config);
    return formatApiResponse(response);
  },

  // Get backtest result
  getResult: async (backtestId: string): Promise<ApiResponse<BacktestResult>> => {
    const response = await request.get<ApiResponse<BacktestResult>>(`/backtest/${backtestId}`);
    return formatApiResponse(response);
  },

  // List backtest results
  listResults: async (pagination?: PaginationParams): Promise<ApiResponse<BacktestResult[]>> => {
    const queryString = pagination ? buildQueryParams(pagination) : '';
    const response = await request.get<ApiResponse<BacktestResult[]>>(`/backtest/list${queryString ? `?${queryString}` : ''}`);
    return formatApiResponse(response);
  },
};

// Stock API for quick access
export const stockApi = {
  // Get stock info
  getInfo: async (symbol: string): Promise<any> => {
    const response = await request.get(`/stock/info/${symbol}`);
    return formatApiResponse(response);
  },

  // Search stocks
  search: async (keyword: string = '', limit: number = 20): Promise<any> => {
    const response = await request.get(`/stock/search?keyword=${encodeURIComponent(keyword)}&limit=${limit}`);
    return formatApiResponse(response);
  },
};

// Combined API export for backward compatibility
export const api = {
  system: systemApi,
  data: dataApi,
  strategy: strategyApi,
  order: orderApi,
  portfolio: portfolioApi,
  risk: riskApi,
  analytics: analyticsApi,
  backtest: backtestApi,
  stock: stockApi,
};