import dayjs from 'dayjs';
import duration from 'dayjs/plugin/duration';
import relativeTime from 'dayjs/plugin/relativeTime';
import { MarketDataRecord, Position, OrderStatus, PerformanceMetrics } from '../services/types';

// Extend dayjs with plugins
dayjs.extend(duration);
dayjs.extend(relativeTime);

// Number formatting utilities
export const formatCurrency = (value: number, currency: string = '¥', precision: number = 2): string => {
  if (isNaN(value)) return `${currency}0.00`;
  return `${currency}${value.toLocaleString('zh-CN', { 
    minimumFractionDigits: precision, 
    maximumFractionDigits: precision 
  })}`;
};

export const formatPercent = (value: number, precision: number = 2): string => {
  if (isNaN(value)) return '0.00%';
  return `${(value * 100).toFixed(precision)}%`;
};

export const formatNumber = (value: number, precision: number = 0): string => {
  if (isNaN(value)) return '0';
  return value.toLocaleString('zh-CN', { 
    minimumFractionDigits: precision, 
    maximumFractionDigits: precision 
  });
};

export const formatLargeNumber = (value: number): string => {
  if (isNaN(value)) return '0';
  
  const abs = Math.abs(value);
  const sign = value < 0 ? '-' : '';
  
  if (abs >= 1e8) {
    return `${sign}${(abs / 1e8).toFixed(2)}亿`;
  } else if (abs >= 1e4) {
    return `${sign}${(abs / 1e4).toFixed(2)}万`;
  } else {
    return `${sign}${abs.toFixed(2)}`;
  }
};

export const formatBytes = (bytes: number, precision: number = 2): string => {
  if (isNaN(bytes) || bytes === 0) return '0 B';
  
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(precision))} ${sizes[i]}`;
};

// Date and time formatting utilities
export const formatDateTime = (timestamp: string | number | Date, format: string = 'YYYY-MM-DD HH:mm:ss'): string => {
  if (!timestamp) return '';
  return dayjs(timestamp).format(format);
};

export const formatDate = (timestamp: string | number | Date): string => {
  return formatDateTime(timestamp, 'YYYY-MM-DD');
};

export const formatTime = (timestamp: string | number | Date): string => {
  return formatDateTime(timestamp, 'HH:mm:ss');
};

export const formatRelativeTime = (timestamp: string | number | Date): string => {
  if (!timestamp) return '';
  return dayjs(timestamp).fromNow();
};

export const formatUptime = (seconds: number): string => {
  if (isNaN(seconds) || seconds < 0) return '0秒';
  
  const duration = dayjs.duration(seconds, 'seconds');
  const days = Math.floor(duration.asDays());
  const hours = duration.hours();
  const minutes = duration.minutes();
  const secs = duration.seconds();
  
  if (days > 0) {
    return `${days}天${hours}小时${minutes}分钟`;
  } else if (hours > 0) {
    return `${hours}小时${minutes}分钟`;
  } else if (minutes > 0) {
    return `${minutes}分钟${secs}秒`;
  } else {
    return `${secs}秒`;
  }
};

// Market data formatting
export const formatPrice = (price: number, precision: number = 2): string => {
  return formatCurrency(price, '¥', precision);
};

export const formatPriceChange = (change: number, precision: number = 2): string => {
  if (isNaN(change)) return '¥0.00';
  const sign = change >= 0 ? '+' : '';
  return `${sign}¥${change.toFixed(precision)}`;
};

export const formatPercentChange = (change: number, precision: number = 2): string => {
  if (isNaN(change)) return '0.00%';
  const sign = change >= 0 ? '+' : '';
  return `${sign}${(change * 100).toFixed(precision)}%`;
};

export const formatVolume = (volume: number): string => {
  return formatLargeNumber(volume);
};

// Portfolio formatting
export const formatPosition = (position: Position): {
  symbol: string;
  quantity: string;
  averagePrice: string;
  currentPrice: string;
  marketValue: string;
  unrealizedPnl: string;
  unrealizedPnlPercent: string;
  pnlColor: string;
} => {
  const marketValue = position.quantity * position.currentPrice;
  const unrealizedPnlPercent = position.averagePrice > 0 
    ? (position.currentPrice - position.averagePrice) / position.averagePrice 
    : 0;
  const pnlColor = position.unrealizedPnl >= 0 ? '#3f8600' : '#cf1322';

  return {
    symbol: position.symbol,
    quantity: formatNumber(position.quantity),
    averagePrice: formatPrice(position.averagePrice),
    currentPrice: formatPrice(position.currentPrice),
    marketValue: formatPrice(marketValue),
    unrealizedPnl: formatPriceChange(position.unrealizedPnl),
    unrealizedPnlPercent: formatPercentChange(unrealizedPnlPercent),
    pnlColor,
  };
};

// Order formatting
export const formatOrderStatus = (status: OrderStatus['status']): {
  text: string;
  color: string;
} => {
  const statusMap = {
    PENDING: { text: '待处理', color: '#1890ff' },
    FILLED: { text: '已成交', color: '#52c41a' },
    PARTIALLY_FILLED: { text: '部分成交', color: '#faad14' },
    REJECTED: { text: '已拒绝', color: '#f5222d' },
    CANCELLED: { text: '已取消', color: '#8c8c8c' },
    ERROR: { text: '错误', color: '#f5222d' },
  };
  
  return statusMap[status] || { text: status, color: '#8c8c8c' };
};

export const formatOrderSide = (side: 'BUY' | 'SELL'): {
  text: string;
  color: string;
} => {
  return side === 'BUY' 
    ? { text: '买入', color: '#f5222d' }
    : { text: '卖出', color: '#52c41a' };
};

// Performance metrics formatting
export const formatPerformanceMetrics = (metrics: PerformanceMetrics): {
  totalReturn: string;
  totalReturnPercent: string;
  annualizedReturn: string;
  volatility: string;
  sharpeRatio: string;
  maxDrawdown: string;
  calmarRatio: string;
  winRate: string;
  profitFactor: string;
  totalTrades: string;
} => {
  return {
    totalReturn: formatCurrency(metrics.totalReturn),
    totalReturnPercent: formatPercent(metrics.totalReturnPercent),
    annualizedReturn: formatPercent(metrics.annualizedReturn),
    volatility: formatPercent(metrics.volatility),
    sharpeRatio: metrics.sharpeRatio.toFixed(3),
    maxDrawdown: formatPercent(metrics.maxDrawdown),
    calmarRatio: metrics.calmarRatio.toFixed(3),
    winRate: formatPercent(metrics.winRate),
    profitFactor: metrics.profitFactor.toFixed(3),
    totalTrades: formatNumber(metrics.totalTrades),
  };
};

// Risk metrics formatting
export const formatRiskLevel = (level: number): {
  text: string;
  color: string;
} => {
  if (level <= 0.3) {
    return { text: '低风险', color: '#52c41a' };
  } else if (level <= 0.7) {
    return { text: '中风险', color: '#faad14' };
  } else {
    return { text: '高风险', color: '#f5222d' };
  }
};

// Chart data formatting
export const formatCandlestickData = (records: MarketDataRecord[]): any[] => {
  return records.map(record => ({
    time: record.datetime,
    open: record.open,
    high: record.high,
    low: record.low,
    close: record.close,
    volume: record.volume,
  }));
};

export const formatLineChartData = (records: MarketDataRecord[], field: keyof MarketDataRecord = 'close'): any[] => {
  return records.map(record => ({
    time: record.datetime,
    value: record[field] as number,
  }));
};

// Validation utilities
export const validateNumber = (value: any): boolean => {
  return typeof value === 'number' && !isNaN(value) && isFinite(value);
};

export const validatePositiveNumber = (value: any): boolean => {
  return validateNumber(value) && value > 0;
};

export const validatePercentage = (value: any): boolean => {
  return validateNumber(value) && value >= 0 && value <= 1;
};

export const validateSymbol = (symbol: string): boolean => {
  return typeof symbol === 'string' && /^[A-Z0-9]{6}\.(SH|SZ)$/.test(symbol);
};

export const validateDateRange = (startDate: string, endDate: string): boolean => {
  const start = dayjs(startDate);
  const end = dayjs(endDate);
  return start.isValid() && end.isValid() && start.isBefore(end);
};

// Data transformation utilities
export const transformMarketData = (data: MarketDataRecord[]): MarketDataRecord[] => {
  return data.map(record => ({
    ...record,
    datetime: dayjs(record.datetime).toISOString(),
    open: Number(record.open) || 0,
    high: Number(record.high) || 0,
    low: Number(record.low) || 0,
    close: Number(record.close) || 0,
    volume: Number(record.volume) || 0,
    adjClose: Number(record.adjClose) || record.close,
  }));
};

export const calculatePriceChange = (current: number, previous: number): {
  change: number;
  changePercent: number;
} => {
  const change = current - previous;
  const changePercent = previous > 0 ? change / previous : 0;
  
  return {
    change,
    changePercent,
  };
};

// Color utilities for financial data
export const getPnlColor = (value: number): string => {
  return value >= 0 ? '#3f8600' : '#cf1322';
};

export const getChangeColor = (value: number): string => {
  return value > 0 ? '#f5222d' : value < 0 ? '#52c41a' : '#8c8c8c';
};

// Export all utilities
export default {
  formatCurrency,
  formatPercent,
  formatNumber,
  formatLargeNumber,
  formatBytes,
  formatDateTime,
  formatDate,
  formatTime,
  formatRelativeTime,
  formatUptime,
  formatPrice,
  formatPriceChange,
  formatPercentChange,
  formatVolume,
  formatPosition,
  formatOrderStatus,
  formatOrderSide,
  formatPerformanceMetrics,
  formatRiskLevel,
  formatCandlestickData,
  formatLineChartData,
  validateNumber,
  validatePositiveNumber,
  validatePercentage,
  validateSymbol,
  validateDateRange,
  transformMarketData,
  calculatePriceChange,
  getPnlColor,
  getChangeColor,
};