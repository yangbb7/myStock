import React from 'react';
import { useQuery, useMutation, useQueryClient, UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { message } from 'antd';
import { api } from '../services/api';
import { useSystemStore } from '../stores/systemStore';
import {
  SystemHealth,
  SystemMetrics,
  PortfolioSummary,
  RiskMetrics,
  StrategyPerformance,
  StrategyConfig,
  OrderStatus,
  OrderRequest,
  MarketDataResponse,
  PerformanceMetrics,
  ApiResponse,
  SystemControlRequest,
  SystemControlResponse,
  DateRangeFilter,
  OrderFilter,
  PaginationParams,
} from '../services/types';
import { REFRESH_INTERVALS } from '../utils/constants';
import { handleApiError, handleApiSuccess } from '../utils/helpers';

// Query keys
export const QUERY_KEYS = {
  SYSTEM_HEALTH: ['system', 'health'],
  SYSTEM_METRICS: ['system', 'metrics'],
  PORTFOLIO_SUMMARY: ['portfolio', 'summary'],
  PORTFOLIO_HISTORY: ['portfolio', 'history'],
  PORTFOLIO_POSITIONS: ['portfolio', 'positions'],
  PORTFOLIO_PERFORMANCE: ['portfolio', 'performance'],
  RISK_METRICS: ['risk', 'metrics'],
  RISK_ALERTS: ['risk', 'alerts'],
  STRATEGY_PERFORMANCE: ['strategy', 'performance'],
  STRATEGY_LIST: ['strategy', 'list'],
  STRATEGY_CONFIG: ['strategy', 'config'],
  ORDER_HISTORY: ['order', 'history'],
  ORDER_ACTIVE: ['order', 'active'],
  ORDER_STATS: ['order', 'stats'],
  MARKET_DATA: ['data', 'market'],
  DATA_SYMBOLS: ['data', 'symbols'],
  DATA_STATUS: ['data', 'status'],
  ANALYTICS_PERFORMANCE: ['analytics', 'performance'],
  BACKTEST_HISTORY: ['analytics', 'backtest', 'history'],
} as const;

// System API hooks
export const useSystemHealth = (options?: Partial<UseQueryOptions<SystemHealth>>) => {
  const { setSystemStatus } = useSystemStore();
  
  const query = useQuery({
    queryKey: QUERY_KEYS.SYSTEM_HEALTH,
    queryFn: async () => {
      console.log('[useApi] Calling api.system.getHealth()');
      const result = await api.system.getHealth();
      console.log('[useApi] getHealth result:', result);
      return result;
    },
    refetchInterval: REFRESH_INTERVALS.NORMAL,
    staleTime: REFRESH_INTERVALS.FAST,
    ...options,
  });

  // Log query state
  React.useEffect(() => {
    console.log('[useApi] Query state:', {
      data: query.data,
      isLoading: query.isLoading,
      isError: query.isError,
      error: query.error,
    });
  }, [query.data, query.isLoading, query.isError, query.error]);

  // Update system store when health data changes
  React.useEffect(() => {
    if (query.data) {
      console.log('[useApi] Updating system store with:', query.data);
      setSystemStatus({
        isRunning: query.data.systemRunning,
        uptime: query.data.uptimeSeconds,
        modules: query.data.modules,
      });
    }
  }, [query.data, setSystemStatus]);

  return query;
};

export const useSystemMetrics = (options?: Partial<UseQueryOptions<SystemMetrics>>) => {
  return useQuery({
    queryKey: QUERY_KEYS.SYSTEM_METRICS,
    queryFn: () => api.system.getMetrics(),
    refetchInterval: REFRESH_INTERVALS.NORMAL,
    staleTime: REFRESH_INTERVALS.FAST,
    ...options,
  });
};

export const useSystemControl = () => {
  const queryClient = useQueryClient();
  
  const startSystem = useMutation({
    mutationFn: (modules?: string[]) => api.system.startSystem(modules),
    onSuccess: () => {
      handleApiSuccess('SYSTEM_STARTED');
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.SYSTEM_HEALTH });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.SYSTEM_METRICS });
    },
    onError: handleApiError,
  });
  
  const stopSystem = useMutation({
    mutationFn: (modules?: string[]) => api.system.stopSystem(modules),
    onSuccess: () => {
      handleApiSuccess('SYSTEM_STOPPED');
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.SYSTEM_HEALTH });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.SYSTEM_METRICS });
    },
    onError: handleApiError,
  });
  
  const restartSystem = useMutation({
    mutationFn: (modules?: string[]) => api.system.restartSystem(modules),
    onSuccess: () => {
      message.success('系统重启成功');
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.SYSTEM_HEALTH });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.SYSTEM_METRICS });
    },
    onError: handleApiError,
  });
  
  return {
    startSystem,
    stopSystem,
    restartSystem,
  };
};

// Portfolio API hooks
export const usePortfolioSummary = (options?: Partial<UseQueryOptions<PortfolioSummary>>) => {
  return useQuery({
    queryKey: QUERY_KEYS.PORTFOLIO_SUMMARY,
    queryFn: () => api.portfolio.getSummary(),
    refetchInterval: REFRESH_INTERVALS.FAST,
    staleTime: REFRESH_INTERVALS.REALTIME,
    ...options,
  });
};

export const usePortfolioHistory = (filter?: DateRangeFilter, options?: Partial<UseQueryOptions<any[]>>) => {
  return useQuery({
    queryKey: [...QUERY_KEYS.PORTFOLIO_HISTORY, filter],
    queryFn: () => api.portfolio.getHistory(filter),
    enabled: !!filter,
    staleTime: REFRESH_INTERVALS.NORMAL,
    ...options,
  });
};

export const usePortfolioPositions = (options?: Partial<UseQueryOptions<Record<string, any>>>) => {
  return useQuery({
    queryKey: QUERY_KEYS.PORTFOLIO_POSITIONS,
    queryFn: () => api.portfolio.getPositions(),
    refetchInterval: REFRESH_INTERVALS.FAST,
    staleTime: REFRESH_INTERVALS.REALTIME,
    ...options,
  });
};

export const usePortfolioPerformance = (filter?: DateRangeFilter, options?: Partial<UseQueryOptions<PerformanceMetrics>>) => {
  return useQuery({
    queryKey: [...QUERY_KEYS.PORTFOLIO_PERFORMANCE, filter],
    queryFn: () => api.portfolio.getPerformance(filter),
    enabled: !!filter,
    staleTime: REFRESH_INTERVALS.NORMAL,
    ...options,
  });
};

// Risk API hooks
export const useRiskMetrics = (options?: Partial<UseQueryOptions<RiskMetrics>>) => {
  return useQuery({
    queryKey: QUERY_KEYS.RISK_METRICS,
    queryFn: () => api.risk.getMetrics(),
    refetchInterval: REFRESH_INTERVALS.FAST,
    staleTime: REFRESH_INTERVALS.REALTIME,
    ...options,
  });
};

export const useRiskAlerts = (filter?: DateRangeFilter, options?: Partial<UseQueryOptions<any[]>>) => {
  return useQuery({
    queryKey: [...QUERY_KEYS.RISK_ALERTS, filter],
    queryFn: () => api.risk.getAlerts(filter),
    staleTime: REFRESH_INTERVALS.NORMAL,
    ...options,
  });
};

export const useRiskConfig = (options?: Partial<UseQueryOptions<Record<string, any>>>) => {
  return useQuery({
    queryKey: ['risk', 'config'],
    queryFn: () => api.risk.getConfig(),
    staleTime: REFRESH_INTERVALS.SLOW,
    ...options,
  });
};

export const useUpdateRiskLimits = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (limits: Partial<any>) => api.risk.updateLimits(limits),
    onSuccess: () => {
      message.success('风险限制更新成功');
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.RISK_METRICS });
      queryClient.invalidateQueries({ queryKey: ['risk', 'config'] });
    },
    onError: handleApiError,
  });
};

// Strategy API hooks
export const useStrategyPerformance = (options?: Partial<UseQueryOptions<StrategyPerformance>>) => {
  return useQuery({
    queryKey: QUERY_KEYS.STRATEGY_PERFORMANCE,
    queryFn: () => api.strategy.getPerformance(),
    refetchInterval: REFRESH_INTERVALS.NORMAL,
    staleTime: REFRESH_INTERVALS.FAST,
    ...options,
  });
};

export const useStrategyList = (options?: Partial<UseQueryOptions<string[]>>) => {
  return useQuery({
    queryKey: QUERY_KEYS.STRATEGY_LIST,
    queryFn: () => api.strategy.getStrategies(),
    staleTime: REFRESH_INTERVALS.SLOW,
    ...options,
  });
};

export const useStrategyConfig = (strategyName: string, options?: Partial<UseQueryOptions<StrategyConfig>>) => {
  return useQuery({
    queryKey: [...QUERY_KEYS.STRATEGY_CONFIG, strategyName],
    queryFn: () => api.strategy.getStrategyConfig(strategyName),
    enabled: !!strategyName,
    staleTime: REFRESH_INTERVALS.SLOW,
    ...options,
  });
};

export const useStrategyMutations = () => {
  const queryClient = useQueryClient();
  
  const addStrategy = useMutation({
    mutationFn: (config: StrategyConfig) => api.strategy.addStrategy(config),
    onSuccess: () => {
      handleApiSuccess('STRATEGY_ADDED');
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.STRATEGY_PERFORMANCE });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.STRATEGY_LIST });
    },
    onError: handleApiError,
  });
  
  const updateStrategy = useMutation({
    mutationFn: ({ name, config }: { name: string; config: Partial<StrategyConfig> }) => 
      api.strategy.updateStrategy(name, config),
    onSuccess: () => {
      message.success('策略更新成功');
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.STRATEGY_PERFORMANCE });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.STRATEGY_CONFIG });
    },
    onError: handleApiError,
  });
  
  const startStrategy = useMutation({
    mutationFn: (strategyName: string) => api.strategy.startStrategy(strategyName),
    onSuccess: () => {
      message.success('策略启动成功');
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.STRATEGY_PERFORMANCE });
    },
    onError: handleApiError,
  });
  
  const stopStrategy = useMutation({
    mutationFn: (strategyName: string) => api.strategy.stopStrategy(strategyName),
    onSuccess: () => {
      message.success('策略停止成功');
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.STRATEGY_PERFORMANCE });
    },
    onError: handleApiError,
  });
  
  const deleteStrategy = useMutation({
    mutationFn: (strategyName: string) => api.strategy.deleteStrategy(strategyName),
    onSuccess: () => {
      message.success('策略删除成功');
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.STRATEGY_PERFORMANCE });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.STRATEGY_LIST });
    },
    onError: handleApiError,
  });
  
  return {
    addStrategy,
    updateStrategy,
    startStrategy,
    stopStrategy,
    deleteStrategy,
  };
};

// Order API hooks
export const useOrderHistory = (filter?: OrderFilter & PaginationParams, options?: Partial<UseQueryOptions<OrderStatus[]>>) => {
  return useQuery({
    queryKey: [...QUERY_KEYS.ORDER_HISTORY, filter],
    queryFn: () => api.order.getOrderHistory(filter),
    staleTime: REFRESH_INTERVALS.NORMAL,
    ...options,
  });
};

export const useActiveOrders = (options?: Partial<UseQueryOptions<OrderStatus[]>>) => {
  return useQuery({
    queryKey: QUERY_KEYS.ORDER_ACTIVE,
    queryFn: () => api.order.getActiveOrders(),
    refetchInterval: REFRESH_INTERVALS.FAST,
    staleTime: REFRESH_INTERVALS.REALTIME,
    ...options,
  });
};

export const useOrderStats = (filter?: DateRangeFilter, options?: Partial<UseQueryOptions<Record<string, any>>>) => {
  return useQuery({
    queryKey: [...QUERY_KEYS.ORDER_STATS, filter],
    queryFn: () => api.order.getOrderStats(filter),
    staleTime: REFRESH_INTERVALS.NORMAL,
    ...options,
  });
};

export const useOrderStatus = (orderId: string, options?: Partial<UseQueryOptions<OrderStatus>>) => {
  return useQuery({
    queryKey: ['order', 'status', orderId],
    queryFn: () => api.order.getOrderStatus(orderId),
    enabled: !!orderId,
    refetchInterval: REFRESH_INTERVALS.FAST,
    staleTime: REFRESH_INTERVALS.REALTIME,
    ...options,
  });
};

export const useOrderMutations = () => {
  const queryClient = useQueryClient();
  
  const createOrder = useMutation({
    mutationFn: (orderRequest: OrderRequest) => api.order.createOrder(orderRequest),
    onSuccess: () => {
      handleApiSuccess('ORDER_CREATED');
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.ORDER_HISTORY });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.ORDER_ACTIVE });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.PORTFOLIO_SUMMARY });
    },
    onError: handleApiError,
  });
  
  const cancelOrder = useMutation({
    mutationFn: (orderId: string) => api.order.cancelOrder(orderId),
    onSuccess: () => {
      message.success('订单取消成功');
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.ORDER_HISTORY });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.ORDER_ACTIVE });
    },
    onError: handleApiError,
  });
  
  return {
    createOrder,
    cancelOrder,
  };
};

// Data API hooks
export const useMarketData = (
  symbol: string,
  timeframe: string = '1d',
  startDate?: string,
  endDate?: string,
  options?: Partial<UseQueryOptions<MarketDataResponse>>
) => {
  return useQuery({
    queryKey: [...QUERY_KEYS.MARKET_DATA, symbol, timeframe, startDate, endDate],
    queryFn: () => api.data.getMarketData(symbol, timeframe, startDate, endDate),
    enabled: !!symbol,
    staleTime: REFRESH_INTERVALS.NORMAL,
    ...options,
  });
};

export const useDataSymbols = (options?: Partial<UseQueryOptions<string[]>>) => {
  return useQuery({
    queryKey: QUERY_KEYS.DATA_SYMBOLS,
    queryFn: () => api.data.getSymbols(),
    staleTime: REFRESH_INTERVALS.VERY_SLOW,
    ...options,
  });
};

export const useDataStatus = (options?: Partial<UseQueryOptions<Record<string, any>>>) => {
  return useQuery({
    queryKey: QUERY_KEYS.DATA_STATUS,
    queryFn: () => api.data.getDataStatus(),
    refetchInterval: REFRESH_INTERVALS.NORMAL,
    staleTime: REFRESH_INTERVALS.FAST,
    ...options,
  });
};

// Analytics API hooks
export const useAnalyticsPerformance = (filter?: DateRangeFilter, options?: Partial<UseQueryOptions<PerformanceMetrics>>) => {
  return useQuery({
    queryKey: [...QUERY_KEYS.ANALYTICS_PERFORMANCE, filter],
    queryFn: () => api.analytics.getPerformance(filter),
    staleTime: REFRESH_INTERVALS.NORMAL,
    ...options,
  });
};

export const useBacktestHistory = (options?: Partial<UseQueryOptions<any[]>>) => {
  return useQuery({
    queryKey: QUERY_KEYS.BACKTEST_HISTORY,
    queryFn: () => api.analytics.getBacktestHistory(),
    staleTime: REFRESH_INTERVALS.SLOW,
    ...options,
  });
};

export const useAnalyticsMutations = () => {
  const queryClient = useQueryClient();
  
  const runBacktest = useMutation({
    mutationFn: (config: any) => api.analytics.runBacktest(config),
    onSuccess: () => {
      message.success('回测运行成功');
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.BACKTEST_HISTORY });
    },
    onError: handleApiError,
  });
  
  const generateReport = useMutation({
    mutationFn: ({ type, filter }: { type: string; filter?: DateRangeFilter }) => 
      api.analytics.generateReport(type, filter),
    onSuccess: () => {
      message.success('报告生成成功');
    },
    onError: handleApiError,
  });
  
  const exportData = useMutation({
    mutationFn: ({ type, format, filter }: { type: string; format: string; filter?: any }) => 
      api.analytics.exportData(type, format, filter),
    onSuccess: () => {
      handleApiSuccess('DATA_EXPORTED');
    },
    onError: handleApiError,
  });
  
  return {
    runBacktest,
    generateReport,
    exportData,
  };
};

// Combined hook for dashboard data
export const useDashboardData = () => {
  const systemHealth = useSystemHealth();
  const systemMetrics = useSystemMetrics();
  const portfolioSummary = usePortfolioSummary();
  const riskMetrics = useRiskMetrics();
  const strategyPerformance = useStrategyPerformance();
  
  return {
    systemHealth,
    systemMetrics,
    portfolioSummary,
    riskMetrics,
    strategyPerformance,
    isLoading: systemHealth.isLoading || systemMetrics.isLoading || portfolioSummary.isLoading,
    isError: systemHealth.isError || systemMetrics.isError || portfolioSummary.isError,
    refetchAll: () => {
      systemHealth.refetch();
      systemMetrics.refetch();
      portfolioSummary.refetch();
      riskMetrics.refetch();
      strategyPerformance.refetch();
    },
  };
};

// Export all hooks
export default {
  useSystemHealth,
  useSystemMetrics,
  useSystemControl,
  usePortfolioSummary,
  usePortfolioHistory,
  usePortfolioPositions,
  usePortfolioPerformance,
  useRiskMetrics,
  useRiskAlerts,
  useRiskConfig,
  useUpdateRiskLimits,
  useStrategyPerformance,
  useStrategyList,
  useStrategyConfig,
  useStrategyMutations,
  useOrderHistory,
  useActiveOrders,
  useOrderStats,
  useOrderStatus,
  useOrderMutations,
  useMarketData,
  useDataSymbols,
  useDataStatus,
  useAnalyticsPerformance,
  useBacktestHistory,
  useAnalyticsMutations,
  useDashboardData,
};