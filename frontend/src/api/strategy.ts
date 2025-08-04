import request from '@/utils/request';

export interface VisualStrategyData {
  nodes: Array<{
    id: string;
    type: string;
    params: Record<string, any>;
    position: { x: number; y: number };
  }>;
  edges: Array<{
    source: string;
    target: string;
    targetHandle?: string;
  }>;
  name?: string;
  symbols?: string[];
}

export interface StrategyTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  difficulty: string;
  strategy_data: any;
  performance_metrics?: {
    annual_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
  };
}

export const strategyApi = {
  // 保存可视化策略
  saveVisualStrategy: (data: VisualStrategyData) => {
    return request.post('/api/v1/visual-strategy/save', data);
  },

  // 获取用户策略列表
  listUserStrategies: () => {
    return request.get('/api/v1/visual-strategy/list');
  },

  // 加载策略详情
  loadStrategy: (strategyId: string) => {
    return request.get(`/api/v1/visual-strategy/load/${strategyId}`);
  },

  // 删除策略
  deleteStrategy: (strategyId: string) => {
    return request.delete(`/api/v1/visual-strategy/${strategyId}`);
  },

  // 获取策略模板
  getStrategyTemplates: () => {
    return request.get<{ templates: StrategyTemplate[] }>('/api/v1/visual-strategy/templates');
  },

  // 从模板创建策略
  createFromTemplate: (templateId: string, name: string, symbols: string[]) => {
    return request.post(`/api/v1/visual-strategy/create-from-template/${templateId}`, {
      name,
      symbols,
    });
  },

  // 测试运行策略
  testRunStrategy: (strategyId: string, startDate?: string, endDate?: string) => {
    return request.post(`/api/v1/visual-strategy/test-run/${strategyId}`, {
      start_date: startDate,
      end_date: endDate,
    });
  },

  // 获取指标参数
  getIndicatorParams: (indicatorType: string) => {
    return request.get(`/api/v1/visual-strategy/indicator-params/${indicatorType}`);
  },

  // 获取策略性能报告
  getStrategyPerformance: (strategyId: string, period: string = '1M') => {
    return request.get(`/api/v1/strategy/performance/${strategyId}`, {
      params: { period },
    });
  },

  // 获取策略信号历史
  getStrategySignals: (strategyId: string, startDate?: string, endDate?: string) => {
    return request.get(`/api/v1/strategy/signals/${strategyId}`, {
      params: { start_date: startDate, end_date: endDate },
    });
  },

  // 获取策略回测结果
  getBacktestResults: (strategyId: string) => {
    return request.get(`/api/v1/strategy/backtest/${strategyId}`);
  },

  // 启动策略
  startStrategy: (strategyId: string) => {
    return request.post(`/api/v1/strategy/start/${strategyId}`);
  },

  // 停止策略
  stopStrategy: (strategyId: string) => {
    return request.post(`/api/v1/strategy/stop/${strategyId}`);
  },

  // 获取策略状态
  getStrategyStatus: (strategyId: string) => {
    return request.get(`/api/v1/strategy/status/${strategyId}`);
  },

  // 更新策略参数
  updateStrategyParams: (strategyId: string, params: Record<string, any>) => {
    return request.put(`/api/v1/strategy/params/${strategyId}`, params);
  },

  // 复制策略
  cloneStrategy: (strategyId: string, newName: string) => {
    return request.post(`/api/v1/strategy/clone/${strategyId}`, { name: newName });
  },

  // 导出策略
  exportStrategy: (strategyId: string) => {
    return request.get(`/api/v1/strategy/export/${strategyId}`, {
      responseType: 'blob',
    });
  },

  // 导入策略
  importStrategy: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return request.post('/api/v1/strategy/import', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
};