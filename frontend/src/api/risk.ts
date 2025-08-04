import request from '@/utils/request';

export interface RiskAlert {
  alert_id: string;
  timestamp: string;
  level: 'low' | 'medium' | 'high' | 'critical';
  type: string;
  message: string;
  position_id?: string;
  suggested_action?: string;
  metadata?: Record<string, any>;
}

export interface RiskConfig {
  max_position_size: number;
  max_position_percent: number;
  max_total_exposure: number;
  max_sector_exposure: number;
  max_correlation: number;
  stop_loss_enabled: boolean;
  stop_loss_percent: number;
  trailing_stop_enabled: boolean;
  trailing_stop_percent: number;
  take_profit_enabled: boolean;
  take_profit_levels: Array<{
    percent: number;
    close_ratio: number;
  }>;
}

export interface RiskMetrics {
  total_positions: number;
  risk_level: string;
  active_alerts: number;
  risk_metrics: {
    max_drawdown: number;
    value_at_risk: number;
    expected_shortfall: number;
    portfolio_volatility: number;
  };
  position_distribution: Record<string, number>;
  recommendations: string[];
}

export interface RiskRule {
  rule_id: string;
  name: string;
  condition: string;
  action: string;
  params: Record<string, any>;
  priority: number;
  enabled: boolean;
}

export const riskApi = {
  // 获取风险配置
  getRiskConfig: () => {
    return request.get<RiskConfig>('/api/v1/risk/config');
  },

  // 更新风险配置
  updateRiskConfig: (config: Partial<RiskConfig>) => {
    return request.put('/api/v1/risk/config', config);
  },

  // 获取风险警报
  getRiskAlerts: (params?: {
    level?: string;
    type?: string;
    start_date?: string;
    end_date?: string;
    limit?: number;
  }) => {
    return request.get<RiskAlert[]>('/api/v1/risk/alerts', { params });
  },

  // 执行风险警报建议
  executeRiskAlert: (alertId: string) => {
    return request.post(`/api/v1/risk/alerts/${alertId}/execute`);
  },

  // 忽略风险警报
  dismissRiskAlert: (alertId: string) => {
    return request.post(`/api/v1/risk/alerts/${alertId}/dismiss`);
  },

  // 获取风险指标
  getRiskMetrics: () => {
    return request.get<RiskMetrics>('/api/v1/risk/metrics');
  },

  // 获取历史风险数据
  getRiskHistory: (period: string = '1M') => {
    return request.get('/api/v1/risk/history', {
      params: { period },
    });
  },

  // 获取风险规则
  getRiskRules: () => {
    return request.get<RiskRule[]>('/api/v1/risk/rules');
  },

  // 创建风险规则
  createRiskRule: (rule: Omit<RiskRule, 'rule_id'>) => {
    return request.post('/api/v1/risk/rules', rule);
  },

  // 更新风险规则
  updateRiskRule: (ruleId: string, rule: Partial<RiskRule>) => {
    return request.put(`/api/v1/risk/rules/${ruleId}`, rule);
  },

  // 删除风险规则
  deleteRiskRule: (ruleId: string) => {
    return request.delete(`/api/v1/risk/rules/${ruleId}`);
  },

  // 启用/禁用风险规则
  toggleRiskRule: (ruleId: string, enabled: boolean) => {
    return request.post(`/api/v1/risk/rules/${ruleId}/toggle`, { enabled });
  },

  // 测试风险规则
  testRiskRule: (rule: Partial<RiskRule>) => {
    return request.post('/api/v1/risk/rules/test', rule);
  },

  // 获取仓位风险分析
  getPositionRiskAnalysis: (positionId: string) => {
    return request.get(`/api/v1/risk/positions/${positionId}/analysis`);
  },

  // 获取投资组合风险分析
  getPortfolioRiskAnalysis: () => {
    return request.get('/api/v1/risk/portfolio/analysis');
  },

  // 运行风险压力测试
  runStressTest: (scenarios: Array<{
    name: string;
    market_change: number;
    volatility_change: number;
  }>) => {
    return request.post('/api/v1/risk/stress-test', { scenarios });
  },

  // 获取风险报告
  getRiskReport: (date?: string) => {
    return request.get('/api/v1/risk/report', {
      params: { date },
    });
  },

  // 订阅实时风险警报
  subscribeRiskAlerts: (onAlert: (alert: RiskAlert) => void) => {
    const ws = new WebSocket(`${process.env.REACT_APP_WS_URL}/risk/alerts`);
    
    ws.onmessage = (event) => {
      const alert = JSON.parse(event.data);
      onAlert(alert);
    };
    
    return () => {
      ws.close();
    };
  },

  // 设置风险警报通知
  setAlertNotifications: (settings: {
    email_enabled: boolean;
    sms_enabled: boolean;
    webhook_enabled: boolean;
    webhook_url?: string;
    alert_levels: string[];
  }) => {
    return request.post('/api/v1/risk/notifications', settings);
  },

  // 获取风险限额使用情况
  getRiskLimitsUsage: () => {
    return request.get('/api/v1/risk/limits/usage');
  },

  // 获取风险归因分析
  getRiskAttribution: () => {
    return request.get('/api/v1/risk/attribution');
  },
};