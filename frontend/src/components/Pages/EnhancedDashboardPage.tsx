import React, { useEffect, useState } from 'react';
import { Card, Row, Col, Alert, Spin, Button, message } from 'antd';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { StatisticCard } from '../Common/StatisticCard';
import { api } from '../../services/api';
import { createWebSocketService, getWebSocketService } from '../../services/websocket';
import { SystemHealth, SystemMetrics, PortfolioSummary, RiskMetrics } from '../../services/types';

const EnhancedDashboardPage: React.FC = () => {
  const queryClient = useQueryClient();
  const [wsConnected, setWsConnected] = useState(false);
  const [systemControlLoading, setSystemControlLoading] = useState(false);

  // Initialize WebSocket connection
  useEffect(() => {
    const wsService = createWebSocketService({
      url: import.meta.env.VITE_WS_BASE_URL || 'http://localhost:8000',
      autoConnect: true,
    });

    const unsubscribeStateChange = wsService.onStateChange((state) => {
      setWsConnected(state === 'connected');
    });

    // Subscribe to real-time system status updates
    const systemStatusSub = wsService.subscribeToSystemStatus((data: SystemHealth) => {
      queryClient.setQueryData(['system', 'health'], data);
    });

    return () => {
      unsubscribeStateChange();
      wsService.unsubscribe(systemStatusSub);
    };
  }, [queryClient]);

  // Fetch system health
  const { 
    data: systemHealth, 
    isLoading: healthLoading, 
    error: healthError,
    refetch: refetchHealth 
  } = useQuery<SystemHealth>({
    queryKey: ['system', 'health'],
    queryFn: api.system.getHealth,
    refetchInterval: 10000, // Refetch every 10 seconds
  });

  // Fetch system metrics
  const { 
    data: systemMetrics, 
    isLoading: metricsLoading, 
    error: metricsError 
  } = useQuery<SystemMetrics>({
    queryKey: ['system', 'metrics'],
    queryFn: api.system.getMetrics,
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  // Fetch portfolio summary
  const { 
    data: portfolioSummary, 
    isLoading: portfolioLoading, 
    error: portfolioError 
  } = useQuery<PortfolioSummary>({
    queryKey: ['portfolio', 'summary'],
    queryFn: api.portfolio.getSummary,
    refetchInterval: 5000, // Refetch every 5 seconds
  });

  // Fetch risk metrics
  const { 
    data: riskMetrics, 
    isLoading: riskLoading, 
    error: riskError 
  } = useQuery<RiskMetrics>({
    queryKey: ['risk', 'metrics'],
    queryFn: api.risk.getMetrics,
    refetchInterval: 5000, // Refetch every 5 seconds
  });

  // Handle system control
  const handleSystemControl = async (action: 'start' | 'stop') => {
    setSystemControlLoading(true);
    try {
      if (action === 'start') {
        await api.system.startSystem();
        message.success('系统启动成功');
      } else {
        await api.system.stopSystem();
        message.success('系统停止成功');
      }
      // Refetch system health after control action
      refetchHealth();
    } catch (error: any) {
      message.error(`系统${action === 'start' ? '启动' : '停止'}失败: ${error.message}`);
    } finally {
      setSystemControlLoading(false);
    }
  };

  // Calculate derived metrics
  const isSystemRunning = systemHealth?.data?.system_running ?? false;
  const totalModules = systemHealth?.data?.modules ? Object.keys(systemHealth.data.modules).length : 0;
  const runningModules = systemHealth?.data?.modules 
    ? Object.values(systemHealth.data.modules).filter(module => module.initialized).length 
    : 0;

  const totalValue = portfolioSummary?.data?.total_value ?? 0;
  const cashBalance = portfolioSummary?.data?.cash_balance ?? 0;
  const positionsValue = totalValue - cashBalance;
  const positionsCount = portfolioSummary?.data?.positions ? Object.keys(portfolioSummary.data.positions).length : 0;

  const dailyPnL = riskMetrics?.data?.daily_pnl ?? 0;
  const dailyPnLPercent = totalValue > 0 ? (dailyPnL / totalValue) * 100 : 0;

  return (
    <div style={{ padding: '24px' }}>
      {/* Header */}
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0 }}>系统仪表板</h2>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <div style={{ 
            padding: '4px 8px',
            background: wsConnected ? '#f6ffed' : '#fff2f0',
            border: `1px solid ${wsConnected ? '#b7eb8f' : '#ffccc7'}`,
            borderRadius: '4px',
            fontSize: '12px',
            color: wsConnected ? '#52c41a' : '#ff4d4f'
          }}>
            {wsConnected ? '🟢 实时连接' : '🔴 连接断开'}
          </div>
          <Button 
            type={isSystemRunning ? 'default' : 'primary'}
            loading={systemControlLoading}
            onClick={() => handleSystemControl(isSystemRunning ? 'stop' : 'start')}
            danger={isSystemRunning}
          >
            {isSystemRunning ? '停止系统' : '启动系统'}
          </Button>
        </div>
      </div>

      {/* Error Alerts */}
      {(healthError || metricsError || portfolioError || riskError) && (
        <Alert
          message="数据加载错误"
          description="部分数据无法加载，请检查系统状态或稍后重试"
          type="warning"
          showIcon
          closable
          style={{ marginBottom: '24px' }}
        />
      )}

      {/* Main Statistics */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Spin spinning={healthLoading}>
            <StatisticCard
              title="系统状态"
              value={isSystemRunning ? '运行中' : '已停止'}
              valueStyle={{ color: isSystemRunning ? '#52c41a' : '#ff4d4f' }}
              trend={isSystemRunning ? 'up' : 'down'}
              trendValue={`${runningModules}/${totalModules} 模块`}
            />
          </Spin>
        </Col>
        <Col span={6}>
          <Spin spinning={portfolioLoading}>
            <StatisticCard
              title="总资产"
              value={totalValue}
              precision={2}
              prefix="¥"
              trend={dailyPnL >= 0 ? 'up' : 'down'}
              trendValue={`${dailyPnL >= 0 ? '+' : ''}${dailyPnLPercent.toFixed(2)}%`}
            />
          </Spin>
        </Col>
        <Col span={6}>
          <Spin spinning={portfolioLoading}>
            <StatisticCard
              title="持仓数量"
              value={positionsCount}
              suffix="只"
              trend="neutral"
              trendValue={`市值 ¥${positionsValue.toLocaleString()}`}
            />
          </Spin>
        </Col>
        <Col span={6}>
          <Spin spinning={riskLoading}>
            <StatisticCard
              title="今日盈亏"
              value={dailyPnL}
              precision={2}
              prefix="¥"
              trend={dailyPnL >= 0 ? 'up' : 'down'}
              trendValue={`${dailyPnL >= 0 ? '+' : ''}${dailyPnLPercent.toFixed(2)}%`}
              valueStyle={{ color: dailyPnL >= 0 ? '#52c41a' : '#ff4d4f' }}
            />
          </Spin>
        </Col>
      </Row>

      {/* System Modules Status */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col span={24}>
          <Card title="系统模块状态" loading={healthLoading}>
            {systemHealth?.data?.modules && (
              <Row gutter={[16, 16]}>
                {Object.entries(systemHealth.data.modules).map(([moduleName, moduleData]) => (
                  <Col span={4} key={moduleName}>
                    <div style={{ 
                      textAlign: 'center',
                      padding: '16px',
                      border: '1px solid #f0f0f0',
                      borderRadius: '8px',
                      background: moduleData.initialized ? '#f6ffed' : '#fff2f0'
                    }}>
                      <div style={{ 
                        fontSize: '16px', 
                        fontWeight: 'bold',
                        color: moduleData.initialized ? '#52c41a' : '#ff4d4f',
                        marginBottom: '8px'
                      }}>
                        {moduleData.initialized ? '●' : '○'}
                      </div>
                      <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '4px' }}>
                        {moduleData.module}
                      </div>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        {moduleData.initialized ? '运行中' : '未启动'}
                      </div>
                      {moduleData.metrics && Object.keys(moduleData.metrics).length > 0 && (
                        <div style={{ fontSize: '12px', color: '#999', marginTop: '4px' }}>
                          {Object.entries(moduleData.metrics).map(([key, value]) => (
                            <div key={key}>{key}: {String(value)}</div>
                          ))}
                        </div>
                      )}
                    </div>
                  </Col>
                ))}
              </Row>
            )}
          </Card>
        </Col>
      </Row>

      {/* Additional Metrics */}
      <Row gutter={[16, 16]}>
        <Col span={12}>
          <Card title="系统性能" loading={metricsLoading}>
            {systemMetrics?.data && (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                  <span>运行时间:</span>
                  <span>{Math.floor((systemMetrics.data.uptime_seconds || 0) / 3600)}小时</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                  <span>系统状态:</span>
                  <span style={{ color: isSystemRunning ? '#52c41a' : '#ff4d4f' }}>
                    {isSystemRunning ? '正常运行' : '已停止'}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>最后更新:</span>
                  <span>{new Date(systemMetrics.data.timestamp || '').toLocaleString()}</span>
                </div>
              </div>
            )}
          </Card>
        </Col>
        <Col span={12}>
          <Card title="风险概览" loading={riskLoading}>
            {riskMetrics?.data && (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                  <span>当前回撤:</span>
                  <span style={{ color: (riskMetrics.data.current_drawdown || 0) < -0.05 ? '#ff4d4f' : '#52c41a' }}>
                    {((riskMetrics.data.current_drawdown || 0) * 100).toFixed(2)}%
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                  <span>风险利用率:</span>
                  <span style={{ color: (riskMetrics.data.risk_utilization || 0) > 0.8 ? '#ff4d4f' : '#52c41a' }}>
                    {((riskMetrics.data.risk_utilization || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>最后检查:</span>
                  <span>{new Date(riskMetrics.data.timestamp || '').toLocaleString()}</span>
                </div>
              </div>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default EnhancedDashboardPage;