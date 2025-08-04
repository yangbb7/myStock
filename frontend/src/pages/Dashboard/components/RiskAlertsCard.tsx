import React, { useMemo } from 'react';
import { 
  Card, 
  Alert, 
  List, 
  Badge, 
  Typography, 
  Space, 
  Button, 
  Tag,
  Statistic,
  Row,
  Col,
  Empty,
  Progress,
  Tooltip,
  Spin
} from 'antd';
import { 
  ExclamationCircleOutlined, 
  WarningOutlined, 
  CloseCircleOutlined,
  DeleteOutlined,
  ClearOutlined,
  SafetyCertificateOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import { useRiskMetrics, useRiskAlerts } from '../../../hooks/useApi';
import { useRiskAlerts as useRealTimeRiskAlerts } from '../../../hooks/useRealTime';
import { formatCurrency, formatPercent } from '../../../utils/format';
import { RiskMetrics } from '../../../services/types';

const { Text, Title } = Typography;

interface RiskAlertDisplay {
  id: string;
  level: 'critical' | 'error' | 'warning';
  message: string;
  timestamp: string;
  metrics?: RiskMetrics;
}

const RiskAlertsCard: React.FC = () => {
  const { data: riskMetrics, isLoading: metricsLoading, error: metricsError } = useRiskMetrics();
  const { data: alertsHistory } = useRiskAlerts();
  const { 
    alerts: realtimeAlerts, 
    latestAlert, 
    isConnected, 
    error: realtimeError,
    clearAlerts,
    dismissAlert 
  } = useRealTimeRiskAlerts();

  // Combine real-time and historical alerts
  const allAlerts = useMemo(() => {
    const alerts: RiskAlertDisplay[] = [];
    
    // Add real-time alerts
    realtimeAlerts.forEach((alert, index) => {
      alerts.push({
        id: `rt-${index}`,
        level: alert.level,
        message: alert.message,
        timestamp: alert.timestamp,
        metrics: alert.metrics
      });
    });
    
    // Add historical alerts if no real-time alerts
    if (alertsHistory && realtimeAlerts.length === 0) {
      alertsHistory.forEach((alert: any, index: number) => {
        alerts.push({
          id: `hist-${index}`,
          level: alert.level || 'warning',
          message: alert.message || '历史风险告警',
          timestamp: alert.timestamp || new Date().toISOString(),
          metrics: alert.metrics
        });
      });
    }
    
    return alerts.slice(0, 10); // Show max 10 alerts
  }, [realtimeAlerts, alertsHistory]);

  // Calculate risk statistics
  const riskStats = useMemo(() => {
    if (!riskMetrics) return null;

    const dailyPnl = riskMetrics.dailyPnl || 0;
    const currentDrawdown = riskMetrics.currentDrawdown || 0;
    const riskLimits = riskMetrics.riskLimits || {};
    const riskUtilization = riskMetrics.riskUtilization || {};

    // Calculate risk levels
    const dailyLossRatio = riskUtilization.dailyLossRatio || 0;
    const drawdownRatio = riskUtilization.drawdownRatio || 0;
    const positionSizeRatio = riskUtilization.positionSizeRatio || 0;

    // Determine overall risk level
    const maxRatio = Math.max(dailyLossRatio, drawdownRatio, positionSizeRatio);
    let riskLevel: 'low' | 'medium' | 'high' | 'critical';
    let riskColor: string;

    if (maxRatio >= 0.9) {
      riskLevel = 'critical';
      riskColor = '#ff4d4f';
    } else if (maxRatio >= 0.7) {
      riskLevel = 'high';
      riskColor = '#ff7a45';
    } else if (maxRatio >= 0.5) {
      riskLevel = 'medium';
      riskColor = '#faad14';
    } else {
      riskLevel = 'low';
      riskColor = '#52c41a';
    }

    return {
      dailyPnl,
      currentDrawdown,
      riskLimits,
      riskUtilization,
      dailyLossRatio,
      drawdownRatio,
      positionSizeRatio,
      riskLevel,
      riskColor,
      maxRatio
    };
  }, [riskMetrics]);

  const getAlertIcon = (level: RiskAlertDisplay['level']) => {
    switch (level) {
      case 'critical':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'error':
        return <ExclamationCircleOutlined style={{ color: '#ff7a45' }} />;
      case 'warning':
        return <WarningOutlined style={{ color: '#faad14' }} />;
      default:
        return <ExclamationCircleOutlined />;
    }
  };

  const getAlertColor = (level: RiskAlertDisplay['level']): string => {
    switch (level) {
      case 'critical':
        return 'error';
      case 'error':
        return 'warning';
      case 'warning':
        return 'default';
      default:
        return 'default';
    }
  };

  const getRiskLevelText = (level: string): string => {
    switch (level) {
      case 'critical':
        return '严重';
      case 'high':
        return '高';
      case 'medium':
        return '中等';
      case 'low':
        return '低';
      default:
        return '未知';
    }
  };

  if (metricsLoading) {
    return (
      <Card title="风险监控告警" loading>
        <div style={{ height: '400px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          <Spin size="large" />
        </div>
      </Card>
    );
  }

  if (metricsError && !riskStats) {
    return (
      <Card title="风险监控告警">
        <Alert
          message="数据加载失败"
          description={metricsError.message}
          type="error"
          showIcon
        />
      </Card>
    );
  }

  const criticalAlerts = allAlerts.filter(alert => alert.level === 'critical').length;
  const errorAlerts = allAlerts.filter(alert => alert.level === 'error').length;
  const warningAlerts = allAlerts.filter(alert => alert.level === 'warning').length;

  return (
    <Card 
      title={
        <Space>
          <SafetyCertificateOutlined />
          <Title level={4} style={{ margin: 0 }}>风险监控告警</Title>
          {!isConnected && (
            <Badge status="error" text="离线" />
          )}
        </Space>
      }
      extra={
        <Space>
          <Button 
            size="small" 
            icon={<ClearOutlined />} 
            onClick={clearAlerts}
            disabled={allAlerts.length === 0}
          >
            清空
          </Button>
        </Space>
      }
    >
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* Latest Critical Alert */}
        {latestAlert && latestAlert.level === 'critical' && (
          <Alert
            message="严重风险告警"
            description={
              <Space direction="vertical" style={{ width: '100%' }}>
                <Text>{latestAlert.message}</Text>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  时间: {new Date(latestAlert.timestamp).toLocaleString()}
                </Text>
              </Space>
            }
            type="error"
            icon={<CloseCircleOutlined />}
            showIcon
            closable
          />
        )}

        {/* Risk Overview Statistics */}
        {riskStats && (
          <div>
            <Title level={5}>风险概览</Title>
            <Row gutter={16}>
              <Col span={12}>
                <div style={{ 
                  textAlign: 'center', 
                  padding: '16px', 
                  backgroundColor: riskStats.dailyPnl >= 0 ? '#f6ffed' : '#fff2f0', 
                  borderRadius: '6px',
                  border: `1px solid ${riskStats.dailyPnl >= 0 ? '#b7eb8f' : '#ffccc7'}`
                }}>
                  <Space direction="vertical" size="small">
                    <Text type="secondary">日盈亏</Text>
                    <div style={{ 
                      fontSize: '20px', 
                      fontWeight: 'bold', 
                      color: riskStats.dailyPnl >= 0 ? '#52c41a' : '#ff4d4f' 
                    }}>
                      {formatCurrency(riskStats.dailyPnl)}
                    </div>
                    <Progress 
                      percent={Math.abs(riskStats.dailyLossRatio) * 100} 
                      size="small"
                      status={riskStats.dailyLossRatio > 0.8 ? 'exception' : 'success'}
                      showInfo={false}
                    />
                  </Space>
                </div>
              </Col>
              <Col span={12}>
                <div style={{ 
                  textAlign: 'center', 
                  padding: '16px', 
                  backgroundColor: '#fff2f0', 
                  borderRadius: '6px',
                  border: '1px solid #ffccc7'
                }}>
                  <Space direction="vertical" size="small">
                    <Text type="secondary">当前回撤</Text>
                    <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#ff4d4f' }}>
                      {formatPercent(Math.abs(riskStats.currentDrawdown))}
                    </div>
                    <Progress 
                      percent={riskStats.drawdownRatio * 100} 
                      size="small"
                      status={riskStats.drawdownRatio > 0.8 ? 'exception' : 'active'}
                      showInfo={false}
                    />
                  </Space>
                </div>
              </Col>
            </Row>
          </div>
        )}

        {/* Risk Level Indicator */}
        {riskStats && (
          <div>
            <Row gutter={16}>
              <Col span={8}>
                <Statistic
                  title="风险等级"
                  value={getRiskLevelText(riskStats.riskLevel)}
                  valueStyle={{ 
                    fontSize: '18px',
                    color: riskStats.riskColor
                  }}
                  prefix={
                    <div style={{ 
                      width: '12px', 
                      height: '12px', 
                      borderRadius: '50%', 
                      backgroundColor: riskStats.riskColor,
                      display: 'inline-block',
                      marginRight: '8px'
                    }} />
                  }
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="风险利用率"
                  value={riskStats.maxRatio * 100}
                  precision={1}
                  suffix="%"
                  valueStyle={{ 
                    fontSize: '18px',
                    color: riskStats.maxRatio > 0.8 ? '#ff4d4f' : '#52c41a'
                  }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="活跃告警"
                  value={allAlerts.length}
                  valueStyle={{ 
                    fontSize: '18px',
                    color: allAlerts.length > 0 ? '#ff4d4f' : '#52c41a'
                  }}
                />
              </Col>
            </Row>
          </div>
        )}

        {/* Alert Statistics */}
        <div>
          <Title level={5}>告警统计</Title>
          <Row gutter={16}>
            <Col span={8}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#ff4d4f' }}>
                  {criticalAlerts}
                </div>
                <Text type="secondary">严重告警</Text>
              </div>
            </Col>
            <Col span={8}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#ff7a45' }}>
                  {errorAlerts}
                </div>
                <Text type="secondary">错误告警</Text>
              </div>
            </Col>
            <Col span={8}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#faad14' }}>
                  {warningAlerts}
                </div>
                <Text type="secondary">警告告警</Text>
              </div>
            </Col>
          </Row>
        </div>

        {/* Risk Limits Status */}
        {riskStats && (
          <div>
            <Title level={5}>
              <Space>
                风险限制状态
                <Tooltip title="显示各项风险指标相对于限制的使用情况">
                  <InfoCircleOutlined style={{ color: '#8c8c8c' }} />
                </Tooltip>
              </Space>
            </Title>
            <Space direction="vertical" size="small" style={{ width: '100%' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <Text>日损失限制</Text>
                  <Text>{formatPercent(riskStats.dailyLossRatio * 100)}</Text>
                </div>
                <Progress 
                  percent={riskStats.dailyLossRatio * 100} 
                  status={riskStats.dailyLossRatio > 0.8 ? 'exception' : 'success'}
                  strokeColor={riskStats.dailyLossRatio > 0.8 ? '#ff4d4f' : '#52c41a'}
                />
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <Text>回撤限制</Text>
                  <Text>{formatPercent(riskStats.drawdownRatio * 100)}</Text>
                </div>
                <Progress 
                  percent={riskStats.drawdownRatio * 100} 
                  status={riskStats.drawdownRatio > 0.8 ? 'exception' : 'success'}
                  strokeColor={riskStats.drawdownRatio > 0.8 ? '#ff4d4f' : '#52c41a'}
                />
              </div>
              {riskStats.positionSizeRatio > 0 && (
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <Text>仓位限制</Text>
                    <Text>{formatPercent(riskStats.positionSizeRatio * 100)}</Text>
                  </div>
                  <Progress 
                    percent={riskStats.positionSizeRatio * 100} 
                    status={riskStats.positionSizeRatio > 0.8 ? 'exception' : 'success'}
                    strokeColor={riskStats.positionSizeRatio > 0.8 ? '#ff4d4f' : '#52c41a'}
                  />
                </div>
              )}
            </Space>
          </div>
        )}

        {/* Alert History List */}
        <div>
          <Title level={5}>告警历史</Title>
          {allAlerts.length === 0 ? (
            <Empty 
              description="暂无风险告警" 
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
          ) : (
            <List
              size="small"
              dataSource={allAlerts}
              renderItem={(alert, index) => (
                <List.Item
                  key={alert.id}
                  actions={[
                    <Button
                      key="dismiss"
                      type="text"
                      size="small"
                      icon={<DeleteOutlined />}
                      onClick={() => dismissAlert(index)}
                    >
                      忽略
                    </Button>
                  ]}
                >
                  <List.Item.Meta
                    avatar={getAlertIcon(alert.level)}
                    title={
                      <Space>
                        <Text strong>{alert.message}</Text>
                        <Tag color={alert.level === 'critical' ? 'red' : alert.level === 'error' ? 'orange' : 'gold'}>
                          {alert.level === 'critical' ? '严重' : alert.level === 'error' ? '错误' : '警告'}
                        </Tag>
                      </Space>
                    }
                    description={
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {new Date(alert.timestamp).toLocaleString()}
                      </Text>
                    }
                  />
                </List.Item>
              )}
            />
          )}
        </div>

        {/* Connection Status */}
        {!isConnected && (
          <Alert
            message="连接警告"
            description="实时风险监控连接已断开，可能无法及时接收最新告警。"
            type="warning"
            showIcon
            closable
          />
        )}
      </Space>
    </Card>
  );
};

export default RiskAlertsCard;