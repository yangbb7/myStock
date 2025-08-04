import request from '@/utils/request';

export interface MarketData {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  amount: number;
  high: number;
  low: number;
  open: number;
  close: number;
  timestamp: string;
}

export interface MarketOverview {
  indexList: Array<{
    code: string;
    name: string;
    price: number;
    change: number;
    changePercent: number;
  }>;
  marketStats: {
    totalVolume: number;
    totalAmount: number;
    advanceCount: number;
    declineCount: number;
    unchangedCount: number;
  };
}

export interface MarketCondition {
  trend: 'bullish' | 'bearish' | 'sideways';
  volatility: 'low' | 'medium' | 'high';
  volume_trend: 'increasing' | 'decreasing' | 'stable';
  confidence: number;
}

export const marketApi = {
  // 获取市场概览
  getMarketOverview: () => {
    return request.get<MarketOverview>('/api/v1/market/overview');
  },

  // 获取实时行情
  getRealtimeQuotes: (symbols: string[]) => {
    return request.post<MarketData[]>('/api/v1/market/realtime', { symbols });
  },

  // 获取K线数据
  getKlineData: (symbol: string, period: string, count: number = 100) => {
    return request.get(`/api/v1/market/kline/${symbol}`, {
      params: { period, count },
    });
  },

  // 获取市场深度
  getMarketDepth: (symbol: string) => {
    return request.get(`/api/v1/market/depth/${symbol}`);
  },

  // 获取分时数据
  getTimelineData: (symbol: string) => {
    return request.get(`/api/v1/market/timeline/${symbol}`);
  },

  // 搜索股票
  searchStocks: (keyword: string) => {
    return request.get('/api/v1/market/search', {
      params: { keyword },
    });
  },

  // 获取板块行情
  getSectorQuotes: () => {
    return request.get('/api/v1/market/sectors');
  },

  // 获取涨跌幅排行
  getRankList: (type: 'gainers' | 'losers', limit: number = 10) => {
    return request.get(`/api/v1/market/rank/${type}`, {
      params: { limit },
    });
  },

  // 获取市场状态分析
  analyzeMarketCondition: (symbols: string[], period: string = '1d') => {
    return request.post<{
      overall: MarketCondition;
      individual: Record<string, MarketCondition>;
      analysis_time: string;
    }>('/api/v1/ai-assistant/analyze-market', {
      symbols,
      period,
      lookback_days: 60,
    });
  },

  // 获取市场洞察
  getMarketInsights: () => {
    return request.get('/api/v1/ai-assistant/market-insights');
  },

  // 订阅实时行情
  subscribeRealtimeData: (symbols: string[], onData: (data: MarketData) => void) => {
    // WebSocket订阅逻辑
    const ws = new WebSocket(`${process.env.REACT_APP_WS_URL}/market/realtime`);
    
    ws.onopen = () => {
      ws.send(JSON.stringify({ action: 'subscribe', symbols }));
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onData(data);
    };
    
    return () => {
      ws.close();
    };
  },

  // 获取历史行情
  getHistoricalData: (
    symbol: string,
    startDate: string,
    endDate: string,
    period: string = '1d'
  ) => {
    return request.get(`/api/v1/market/historical/${symbol}`, {
      params: {
        start_date: startDate,
        end_date: endDate,
        period,
      },
    });
  },

  // 获取股票基本信息
  getStockInfo: (symbol: string) => {
    return request.get(`/api/v1/market/stock-info/${symbol}`);
  },

  // 获取财务数据
  getFinancialData: (symbol: string) => {
    return request.get(`/api/v1/market/financial/${symbol}`);
  },

  // 获取资金流向
  getMoneyFlow: (symbol: string) => {
    return request.get(`/api/v1/market/money-flow/${symbol}`);
  },

  // 获取大单成交
  getLargeOrders: (symbol: string, minAmount: number = 1000000) => {
    return request.get(`/api/v1/market/large-orders/${symbol}`, {
      params: { min_amount: minAmount },
    });
  },
};