// Application constants
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
export const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || 'http://localhost:8000';
export const APP_TITLE = import.meta.env.VITE_APP_TITLE || 'myQuant 量化交易系统';
export const APP_VERSION = import.meta.env.VITE_APP_VERSION || '1.0.0';

// API endpoints
export const API_ENDPOINTS = {
  // System endpoints
  HEALTH: '/health',
  METRICS: '/metrics',
  SYSTEM_START: '/system/start',
  SYSTEM_STOP: '/system/stop',
  SYSTEM_RESTART: '/system/restart',
  
  // Strategy endpoints
  STRATEGY_ADD: '/strategy/add',
  STRATEGY_PERFORMANCE: '/strategy/performance',
  STRATEGY_LIST: '/strategy/list',
  STRATEGY_CONFIG: '/strategy/config',
  STRATEGY_UPDATE: '/strategy/update',
  STRATEGY_START: '/strategy/start',
  STRATEGY_STOP: '/strategy/stop',
  STRATEGY_DELETE: '/strategy/delete',
  
  // Data endpoints
  DATA_MARKET: '/data/market',
  DATA_TICK: '/data/tick',
  DATA_SYMBOLS: '/data/symbols',
  DATA_STATUS: '/data/status',
  
  // Order endpoints
  ORDER_CREATE: '/order/create',
  ORDER_STATUS: '/order/status',
  ORDER_HISTORY: '/order/history',
  ORDER_CANCEL: '/order/cancel',
  ORDER_ACTIVE: '/order/active',
  ORDER_STATS: '/order/stats',
  
  // Portfolio endpoints
  PORTFOLIO_SUMMARY: '/portfolio/summary',
  PORTFOLIO_HISTORY: '/portfolio/history',
  PORTFOLIO_POSITIONS: '/portfolio/positions',
  PORTFOLIO_PERFORMANCE: '/portfolio/performance',
  
  // Risk endpoints
  RISK_METRICS: '/risk/metrics',
  RISK_ALERTS: '/risk/alerts',
  RISK_LIMITS: '/risk/limits',
  RISK_CONFIG: '/risk/config',
  
  // Analytics endpoints
  ANALYTICS_PERFORMANCE: '/analytics/performance',
  ANALYTICS_BACKTEST: '/analytics/backtest',
  ANALYTICS_REPORT: '/analytics/report',
  ANALYTICS_EXPORT: '/analytics/export',
} as const;

// WebSocket events
export const WS_EVENTS = {
  MARKET_DATA: 'market_data',
  SYSTEM_STATUS: 'system_status',
  ORDER_UPDATE: 'order_update',
  RISK_ALERT: 'risk_alert',
  PORTFOLIO_UPDATE: 'portfolio_update',
  STRATEGY_SIGNAL: 'strategy_signal',
} as const;

// UI constants
export const REFRESH_INTERVALS = {
  REALTIME: 500,   // 0.5 seconds - for real-time data
  FAST: 1000,      // 1 second - for frequently changing data
  NORMAL: 5000,    // 5 seconds - for regular updates
  SLOW: 30000,     // 30 seconds - for slow changing data
  VERY_SLOW: 60000, // 1 minute - for rarely changing data
} as const;

// Chart constants
export const CHART_COLORS = {
  PRIMARY: '#1890ff',
  SUCCESS: '#52c41a',
  WARNING: '#faad14',
  ERROR: '#f5222d',
  UP: '#f5222d',      // 上涨红色
  DOWN: '#52c41a',    // 下跌绿色
  NEUTRAL: '#8c8c8c',
  VOLUME: '#7fbe9e',
  BACKGROUND: '#fafafa',
} as const;

// Order constants
export const ORDER_SIDES = {
  BUY: 'BUY',
  SELL: 'SELL',
} as const;

export const ORDER_TYPES = {
  MARKET: 'MARKET',
  LIMIT: 'LIMIT',
  STOP: 'STOP',
  STOP_LIMIT: 'STOP_LIMIT',
} as const;

export const ORDER_STATUS = {
  PENDING: 'PENDING',
  FILLED: 'FILLED',
  PARTIALLY_FILLED: 'PARTIALLY_FILLED',
  REJECTED: 'REJECTED',
  CANCELLED: 'CANCELLED',
  ERROR: 'ERROR',
} as const;

// Risk levels
export const RISK_LEVELS = {
  LOW: 'LOW',
  MEDIUM: 'MEDIUM',
  HIGH: 'HIGH',
  CRITICAL: 'CRITICAL',
} as const;

// System modules
export const SYSTEM_MODULES = {
  DATA: 'data',
  STRATEGY: 'strategy',
  EXECUTION: 'execution',
  RISK: 'risk',
  PORTFOLIO: 'portfolio',
  ANALYTICS: 'analytics',
} as const;

// Time frames
export const TIME_FRAMES = {
  '1m': '1分钟',
  '5m': '5分钟',
  '15m': '15分钟',
  '30m': '30分钟',
  '1h': '1小时',
  '4h': '4小时',
  '1d': '1天',
  '1w': '1周',
  '1M': '1月',
} as const;

// Date formats
export const DATE_FORMATS = {
  DATE: 'YYYY-MM-DD',
  TIME: 'HH:mm:ss',
  DATETIME: 'YYYY-MM-DD HH:mm:ss',
  DATETIME_SHORT: 'MM-DD HH:mm',
  TIME_SHORT: 'HH:mm',
} as const;

// Pagination defaults
export const PAGINATION = {
  DEFAULT_PAGE_SIZE: 20,
  PAGE_SIZE_OPTIONS: ['10', '20', '50', '100'],
  SHOW_SIZE_CHANGER: true,
  SHOW_QUICK_JUMPER: true,
} as const;

// Table constants
export const TABLE_SCROLL = {
  X: 'max-content',
  Y: 400,
} as const;

// Form validation rules
export const VALIDATION_RULES = {
  REQUIRED: { required: true, message: '此字段为必填项' },
  POSITIVE_NUMBER: {
    validator: (_: any, value: number) => {
      if (value && value <= 0) {
        return Promise.reject(new Error('必须是正数'));
      }
      return Promise.resolve();
    },
  },
  PERCENTAGE: {
    validator: (_: any, value: number) => {
      if (value && (value < 0 || value > 1)) {
        return Promise.reject(new Error('必须在0-1之间'));
      }
      return Promise.resolve();
    },
  },
  SYMBOL: {
    pattern: /^[A-Z0-9]{6}\.(SH|SZ)$/,
    message: '股票代码格式不正确',
  },
} as const;

// Local storage keys
export const STORAGE_KEYS = {
  USER_PREFERENCES: 'myquant_user_preferences',
  THEME: 'myquant_theme',
  LANGUAGE: 'myquant_language',
  DASHBOARD_LAYOUT: 'myquant_dashboard_layout',
  CHART_SETTINGS: 'myquant_chart_settings',
} as const;

// Error codes
export const ERROR_CODES = {
  NETWORK_ERROR: 'NETWORK_ERROR',
  TIMEOUT: 'TIMEOUT',
  UNAUTHORIZED: 'UNAUTHORIZED',
  FORBIDDEN: 'FORBIDDEN',
  NOT_FOUND: 'NOT_FOUND',
  SERVER_ERROR: 'SERVER_ERROR',
  VALIDATION_ERROR: 'VALIDATION_ERROR',
} as const;

// Success messages
export const SUCCESS_MESSAGES = {
  STRATEGY_ADDED: '策略添加成功',
  ORDER_CREATED: '订单创建成功',
  SYSTEM_STARTED: '系统启动成功',
  SYSTEM_STOPPED: '系统停止成功',
  DATA_EXPORTED: '数据导出成功',
  SETTINGS_SAVED: '设置保存成功',
} as const;

// Error messages
export const ERROR_MESSAGES = {
  NETWORK_ERROR: '网络连接失败，请检查网络设置',
  TIMEOUT: '请求超时，请稍后重试',
  UNAUTHORIZED: '未授权访问，请重新登录',
  FORBIDDEN: '访问被拒绝，权限不足',
  NOT_FOUND: '请求的资源不存在',
  SERVER_ERROR: '服务器内部错误，请稍后重试',
  VALIDATION_ERROR: '数据验证失败，请检查输入',
  UNKNOWN_ERROR: '发生未知错误，请联系管理员',
} as const;

// Menu items
export const MENU_ITEMS = [
  {
    key: 'dashboard',
    label: '系统仪表板',
    icon: 'DashboardOutlined',
    path: '/dashboard',
  },
  {
    key: 'strategy',
    label: '策略管理',
    icon: 'SettingOutlined',
    path: '/strategy',
  },
  {
    key: 'data',
    label: '实时数据',
    icon: 'LineChartOutlined',
    path: '/data',
  },
  {
    key: 'orders',
    label: '订单管理',
    icon: 'ShoppingOutlined',
    path: '/orders',
  },
  {
    key: 'portfolio',
    label: '投资组合',
    icon: 'PieChartOutlined',
    path: '/portfolio',
  },
  {
    key: 'risk',
    label: '风险监控',
    icon: 'AlertOutlined',
    path: '/risk',
  },
  {
    key: 'backtest',
    label: '回测分析',
    icon: 'BarChartOutlined',
    path: '/backtest',
  },
  {
    key: 'system',
    label: '系统管理',
    icon: 'ControlOutlined',
    path: '/system',
  },
] as const;
