// API Response Types for myQuant Frontend

// Base API Response
export interface ApiResponse<T = any> {
  data?: T;
  message?: string;
  success?: boolean;
  timestamp?: string;
}

// System Health Types
export interface ModuleHealth {
  module: string;
  initialized: boolean;
  metrics: Record<string, any>;
  timestamp: string;
}

export interface SystemHealth {
  systemRunning: boolean;
  uptimeSeconds: number;
  modules: Record<string, ModuleHealth>;
  timestamp: string;
}

// System Metrics Types
export interface SystemMetrics {
  system: {
    running: boolean;
    uptime: number;
    modulesCount: number;
  };
  data?: Record<string, any>;
  strategy?: Record<string, any>;
  execution?: Record<string, any>;
  risk?: Record<string, any>;
  portfolio?: Record<string, any>;
  analytics?: Record<string, any>;
  portfolioSummary?: PortfolioSummary;
  riskMetrics?: RiskMetrics;
  performance?: PerformanceMetrics;
}

// Portfolio Types
export interface Position {
  symbol: string;
  quantity: number;
  averagePrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  marketValue?: number;
  weight?: number;
}

export interface PortfolioSummary {
  totalValue: number;
  cashBalance: number;
  positions: Record<string, Position>;
  unrealizedPnl: number;
  positionsCount: number;
  realizedPnl?: number;
  totalReturn?: number;
  totalReturnPercent?: number;
  timestamp?: string;
}

// Risk Management Types
export interface RiskLimits {
  maxPositionSize: number;
  maxDrawdownLimit: number;
  maxDailyLoss: number;
}

export interface RiskUtilization {
  dailyLossRatio: number;
  drawdownRatio: number;
  positionSizeRatio?: number;
}

export interface RiskMetrics {
  dailyPnl: number;
  currentDrawdown: number;
  riskLimits: RiskLimits;
  riskUtilization: RiskUtilization;
  var?: number; // Value at Risk
  volatility?: number;
  sharpeRatio?: number;
  maxDrawdown?: number;
  timestamp?: string;
}

// Strategy Types
export interface StrategyConfig {
  name: string;
  symbols: string[];
  initialCapital: number;
  riskTolerance: number;
  maxPositionSize: number;
  stopLoss?: number;
  takeProfit?: number;
  indicators: Record<string, any>;
  parameters?: Record<string, any>;
}

export interface StrategyPerformanceData {
  signalsGenerated: number;
  successfulTrades: number;
  totalPnl: number;
  winRate?: number;
  avgWin?: number;
  avgLoss?: number;
  profitFactor?: number;
  maxDrawdown?: number;
  sharpeRatio?: number;
}

export interface StrategyPerformance {
  [strategyName: string]: StrategyPerformanceData;
}

// Order Types
export type OrderSide = 'BUY' | 'SELL';
export type OrderType = 'MARKET' | 'LIMIT' | 'STOP' | 'STOP_LIMIT';
export type OrderStatusType = 'PENDING' | 'FILLED' | 'PARTIALLY_FILLED' | 'REJECTED' | 'CANCELLED' | 'ERROR';

export interface OrderRequest {
  symbol: string;
  side: OrderSide;
  quantity: number;
  price?: number;
  orderType?: OrderType;
  stopPrice?: number;
  timeInForce?: string;
}

export interface OrderStatus {
  orderId: string;
  symbol: string;
  side: OrderSide;
  quantity: number;
  price?: number;
  orderType: OrderType;
  status: OrderStatusType;
  timestamp: string;
  executedPrice?: number;
  executedQuantity?: number;
  remainingQuantity?: number;
  avgExecutionPrice?: number;
  commission?: number;
  errorMessage?: string;
}

// Market Data Types
export interface MarketDataRecord {
  datetime: string;
  symbol: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  adjClose: number;
  amount?: number;
  turnover?: number;
}

export interface MarketDataResponse {
  records: MarketDataRecord[];
  shape: [number, number];
  columns: string[];
  symbol?: string;
  name?: string;  // 添加股票名称字段
  timeframe?: string;
  startDate?: string;
  endDate?: string;
}

export interface TickData {
  symbol: string;
  price: number;
  volume: number;
  timestamp: string;
  bid?: number;
  ask?: number;
  bidSize?: number;
  askSize?: number;
}

export interface MarketData {
  symbol: string;
  name?: string;  // 添加股票名称字段
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: string;
  high?: number;
  low?: number;
  open?: number;
  previousClose?: number;
}

// Analytics Types
export interface PerformanceMetrics {
  totalReturn: number;
  totalReturnPercent: number;
  annualizedReturn: number;
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  calmarRatio: number;
  winRate: number;
  profitFactor: number;
  avgWin: number;
  avgLoss: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
}

export interface BacktestConfig {
  strategyName: string;
  symbols: string[];
  startDate: string;
  endDate: string;
  initialCapital: number;
  commission: number;
  slippage: number;
  parameters?: Record<string, any>;
}

export interface BacktestResult {
  config: BacktestConfig;
  performance: PerformanceMetrics;
  trades: TradeRecord[];
  equity: EquityPoint[];
  drawdown: DrawdownPoint[];
  positions: PositionRecord[];
}

export interface TradeRecord {
  entryTime: string;
  exitTime: string;
  symbol: string;
  side: OrderSide;
  quantity: number;
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  pnlPercent: number;
  commission: number;
  duration: number;
}

export interface EquityPoint {
  timestamp: string;
  equity: number;
  cash: number;
  positions: number;
}

export interface DrawdownPoint {
  timestamp: string;
  drawdown: number;
  drawdownPercent: number;
}

export interface PositionRecord {
  timestamp: string;
  symbol: string;
  quantity: number;
  price: number;
  value: number;
}

// System Control Types
export interface SystemControlRequest {
  action: 'start' | 'stop' | 'restart';
  modules?: string[];
}

export interface SystemControlResponse {
  success: boolean;
  message: string;
  timestamp: string;
  affectedModules?: string[];
}

// WebSocket Message Types
export interface WebSocketMessage<T = any> {
  type: string;
  data: T;
  timestamp: string;
}

export interface MarketDataMessage extends WebSocketMessage<MarketData> {
  type: 'market_data';
}

export interface OrderUpdateMessage extends WebSocketMessage<OrderStatus> {
  type: 'order_update';
}

export interface SystemStatusMessage extends WebSocketMessage<SystemHealth> {
  type: 'system_status';
}

export interface RiskAlertMessage extends WebSocketMessage<{
  level: 'warning' | 'error' | 'critical';
  message: string;
  metrics: RiskMetrics;
}> {
  type: 'risk_alert';
}

// Error Types
export interface ApiError {
  code: string;
  message: string;
  details?: any;
  timestamp: string;
}

// Pagination Types
export interface PaginationParams {
  page?: number;
  pageSize?: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

// Filter Types
export interface DateRangeFilter {
  startDate?: string;
  endDate?: string;
}

export interface SymbolFilter {
  symbols?: string[];
}

export interface OrderFilter extends DateRangeFilter, SymbolFilter {
  status?: OrderStatusType[];
  side?: OrderSide[];
}

export interface TradeFilter extends DateRangeFilter, SymbolFilter {
  minPnl?: number;
  maxPnl?: number;
}