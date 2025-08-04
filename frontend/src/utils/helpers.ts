import { message } from 'antd';
import { ApiError, MarketDataRecord, OrderStatus, Position } from '../services/types';
import { ERROR_MESSAGES, SUCCESS_MESSAGES } from './constants';

// Error handling utilities
export const handleApiError = (error: ApiError | any): void => {
  console.error('API Error:', error);
  
  if (error?.code) {
    const errorMessage = ERROR_MESSAGES[error.code as keyof typeof ERROR_MESSAGES] || error.message;
    message.error(errorMessage);
  } else if (error?.message) {
    message.error(error.message);
  } else {
    message.error(ERROR_MESSAGES.UNKNOWN_ERROR);
  }
};

export const handleApiSuccess = (messageKey: keyof typeof SUCCESS_MESSAGES): void => {
  message.success(SUCCESS_MESSAGES[messageKey]);
};

// Data processing utilities
export const processMarketData = (data: MarketDataRecord[]): MarketDataRecord[] => {
  return data
    .filter(record => record && record.symbol && record.datetime)
    .map(record => ({
      ...record,
      open: Number(record.open) || 0,
      high: Number(record.high) || 0,
      low: Number(record.low) || 0,
      close: Number(record.close) || 0,
      volume: Number(record.volume) || 0,
      adjClose: Number(record.adjClose) || record.close,
    }))
    .sort((a, b) => new Date(a.datetime).getTime() - new Date(b.datetime).getTime());
};

export const calculateTechnicalIndicators = (data: MarketDataRecord[], period: number = 20): {
  sma: number[];
  ema: number[];
  bollinger: { upper: number[]; middle: number[]; lower: number[] };
  rsi: number[];
} => {
  const closes = data.map(d => d.close);
  
  // Simple Moving Average
  const sma = calculateSMA(closes, period);
  
  // Exponential Moving Average
  const ema = calculateEMA(closes, period);
  
  // Bollinger Bands
  const bollinger = calculateBollingerBands(closes, period);
  
  // RSI
  const rsi = calculateRSI(closes, 14);
  
  return { sma, ema, bollinger, rsi };
};

const calculateSMA = (data: number[], period: number): number[] => {
  const result: number[] = [];
  for (let i = period - 1; i < data.length; i++) {
    const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
    result.push(sum / period);
  }
  return result;
};

const calculateEMA = (data: number[], period: number): number[] => {
  const result: number[] = [];
  const multiplier = 2 / (period + 1);
  
  // First EMA is SMA
  const firstSMA = data.slice(0, period).reduce((a, b) => a + b, 0) / period;
  result.push(firstSMA);
  
  for (let i = period; i < data.length; i++) {
    const ema = (data[i] - result[result.length - 1]) * multiplier + result[result.length - 1];
    result.push(ema);
  }
  
  return result;
};

const calculateBollingerBands = (data: number[], period: number): {
  upper: number[];
  middle: number[];
  lower: number[];
} => {
  const sma = calculateSMA(data, period);
  const upper: number[] = [];
  const middle: number[] = [];
  const lower: number[] = [];
  
  for (let i = period - 1; i < data.length; i++) {
    const slice = data.slice(i - period + 1, i + 1);
    const mean = slice.reduce((a, b) => a + b, 0) / period;
    const variance = slice.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / period;
    const stdDev = Math.sqrt(variance);
    
    const smaIndex = i - period + 1;
    middle.push(sma[smaIndex]);
    upper.push(sma[smaIndex] + 2 * stdDev);
    lower.push(sma[smaIndex] - 2 * stdDev);
  }
  
  return { upper, middle, lower };
};

const calculateRSI = (data: number[], period: number = 14): number[] => {
  const result: number[] = [];
  const gains: number[] = [];
  const losses: number[] = [];
  
  // Calculate gains and losses
  for (let i = 1; i < data.length; i++) {
    const change = data[i] - data[i - 1];
    gains.push(change > 0 ? change : 0);
    losses.push(change < 0 ? Math.abs(change) : 0);
  }
  
  // Calculate RSI
  for (let i = period - 1; i < gains.length; i++) {
    const avgGain = gains.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
    const avgLoss = losses.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
    
    if (avgLoss === 0) {
      result.push(100);
    } else {
      const rs = avgGain / avgLoss;
      const rsi = 100 - (100 / (1 + rs));
      result.push(rsi);
    }
  }
  
  return result;
};

// Portfolio utilities
export const calculatePortfolioMetrics = (positions: Record<string, Position>): {
  totalValue: number;
  totalCost: number;
  totalPnl: number;
  totalPnlPercent: number;
  positionCount: number;
  topGainer: Position | null;
  topLoser: Position | null;
} => {
  const positionArray = Object.values(positions);
  
  if (positionArray.length === 0) {
    return {
      totalValue: 0,
      totalCost: 0,
      totalPnl: 0,
      totalPnlPercent: 0,
      positionCount: 0,
      topGainer: null,
      topLoser: null,
    };
  }
  
  const totalValue = positionArray.reduce((sum, pos) => sum + pos.quantity * pos.currentPrice, 0);
  const totalCost = positionArray.reduce((sum, pos) => sum + pos.quantity * pos.averagePrice, 0);
  const totalPnl = totalValue - totalCost;
  const totalPnlPercent = totalCost > 0 ? totalPnl / totalCost : 0;
  
  const sortedByPnl = [...positionArray].sort((a, b) => b.unrealizedPnl - a.unrealizedPnl);
  const topGainer = sortedByPnl[0]?.unrealizedPnl > 0 ? sortedByPnl[0] : null;
  const topLoser = sortedByPnl[sortedByPnl.length - 1]?.unrealizedPnl < 0 ? sortedByPnl[sortedByPnl.length - 1] : null;
  
  return {
    totalValue,
    totalCost,
    totalPnl,
    totalPnlPercent,
    positionCount: positionArray.length,
    topGainer,
    topLoser,
  };
};

// Order utilities
export const groupOrdersByStatus = (orders: OrderStatus[]): Record<string, OrderStatus[]> => {
  return orders.reduce((groups, order) => {
    const status = order.status;
    if (!groups[status]) {
      groups[status] = [];
    }
    groups[status].push(order);
    return groups;
  }, {} as Record<string, OrderStatus[]>);
};

export const calculateOrderStats = (orders: OrderStatus[]): {
  total: number;
  filled: number;
  pending: number;
  rejected: number;
  fillRate: number;
  avgExecutionTime: number;
} => {
  const total = orders.length;
  const filled = orders.filter(o => o.status === 'FILLED').length;
  const pending = orders.filter(o => o.status === 'PENDING').length;
  const rejected = orders.filter(o => o.status === 'REJECTED').length;
  const fillRate = total > 0 ? filled / total : 0;
  
  // Calculate average execution time for filled orders
  const filledOrders = orders.filter(o => o.status === 'FILLED' && o.executedPrice);
  const avgExecutionTime = filledOrders.length > 0 
    ? filledOrders.reduce((sum, order) => {
        const created = new Date(order.timestamp).getTime();
        // Assuming execution time is stored somewhere, using timestamp as placeholder
        return sum + 1000; // 1 second placeholder
      }, 0) / filledOrders.length
    : 0;
  
  return {
    total,
    filled,
    pending,
    rejected,
    fillRate,
    avgExecutionTime,
  };
};

// Chart utilities
export const generateChartOptions = (type: 'line' | 'candlestick' | 'bar', data: any[], options?: any) => {
  const baseOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 300,
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross',
      },
    },
    legend: {
      show: true,
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true,
    },
    ...options,
  };
  
  switch (type) {
    case 'line':
      return {
        ...baseOptions,
        xAxis: {
          type: 'category',
          data: data.map(d => d.time),
        },
        yAxis: {
          type: 'value',
        },
        series: [{
          type: 'line',
          data: data.map(d => d.value),
          smooth: true,
        }],
      };
      
    case 'candlestick':
      return {
        ...baseOptions,
        xAxis: {
          type: 'category',
          data: data.map(d => d.time),
        },
        yAxis: {
          type: 'value',
          scale: true,
        },
        series: [{
          type: 'candlestick',
          data: data.map(d => [d.open, d.close, d.low, d.high]),
        }],
      };
      
    case 'bar':
      return {
        ...baseOptions,
        xAxis: {
          type: 'category',
          data: data.map(d => d.name),
        },
        yAxis: {
          type: 'value',
        },
        series: [{
          type: 'bar',
          data: data.map(d => d.value),
        }],
      };
      
    default:
      return baseOptions;
  }
};

// Utility functions
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout;
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
};

export const throttle = <T extends (...args: any[]) => any>(
  func: T,
  limit: number
): ((...args: Parameters<T>) => void) => {
  let inThrottle: boolean;
  
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
};

export const deepClone = <T>(obj: T): T => {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }
  
  if (obj instanceof Date) {
    return new Date(obj.getTime()) as unknown as T;
  }
  
  if (obj instanceof Array) {
    return obj.map(item => deepClone(item)) as unknown as T;
  }
  
  if (typeof obj === 'object') {
    const clonedObj = {} as T;
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        clonedObj[key] = deepClone(obj[key]);
      }
    }
    return clonedObj;
  }
  
  return obj;
};

export const generateId = (): string => {
  return `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

export const sleep = (ms: number): Promise<void> => {
  return new Promise(resolve => setTimeout(resolve, ms));
};

export const retry = async <T>(
  fn: () => Promise<T>,
  attempts: number = 3,
  delay: number = 1000
): Promise<T> => {
  try {
    return await fn();
  } catch (error) {
    if (attempts <= 1) {
      throw error;
    }
    
    await sleep(delay);
    return retry(fn, attempts - 1, delay * 2);
  }
};

// Local storage utilities
export const storage = {
  get: <T>(key: string, defaultValue?: T): T | null => {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue || null;
    } catch (error) {
      console.error('Error reading from localStorage:', error);
      return defaultValue || null;
    }
  },
  
  set: <T>(key: string, value: T): void => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error('Error writing to localStorage:', error);
    }
  },
  
  remove: (key: string): void => {
    try {
      localStorage.removeItem(key);
    } catch (error) {
      console.error('Error removing from localStorage:', error);
    }
  },
  
  clear: (): void => {
    try {
      localStorage.clear();
    } catch (error) {
      console.error('Error clearing localStorage:', error);
    }
  },
};

// URL utilities
export const buildUrl = (base: string, params: Record<string, any>): string => {
  const url = new URL(base);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      url.searchParams.append(key, String(value));
    }
  });
  return url.toString();
};

export const parseQueryString = (search: string): Record<string, string> => {
  const params = new URLSearchParams(search);
  const result: Record<string, string> = {};
  
  for (const [key, value] of params.entries()) {
    result[key] = value;
  }
  
  return result;
};

// Export all utilities
export default {
  handleApiError,
  handleApiSuccess,
  processMarketData,
  calculateTechnicalIndicators,
  calculatePortfolioMetrics,
  groupOrdersByStatus,
  calculateOrderStats,
  generateChartOptions,
  debounce,
  throttle,
  deepClone,
  generateId,
  sleep,
  retry,
  storage,
  buildUrl,
  parseQueryString,
};