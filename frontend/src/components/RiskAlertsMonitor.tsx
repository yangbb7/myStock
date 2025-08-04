import React from 'react';
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
  Empty
} from 'antd';
import { 
  ExclamationCircleOutlined, 
  WarningOutlined, 
  CloseCircleOutlined,
  DeleteOutlined,
  ClearOutlined
} from '@ant-design/icons';
import { useRiskAlerts } from '../hooks/useRealTime';
import { RiskAlert } from '../hooks/useRealTime';

const { Text, Title } = Typography;

interface RiskAlertsMonitorProps {
  maxDisplayAlerts?: number;
  showMetrics?: boolean;
  autoRefresh?: boolean;
}

const RiskAlertsMonitor: React.FC<RiskAlertsMonitorProps> = ({
  maxDisplayAlerts = 10,
  showMetrics = true,
  autoRefresh = true
}) => {
  const { 
    alerts, 
    latestAlert, 
    isConnected, 
    error, 
    clearAlerts, 
    dismissAlert 
  } = useRiskAlerts({
    autoSubscribe: autoRefresh,
    maxAlertsSize: 50
  });

  const getAlertIcon = (level: RiskAlert['level']) => {
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

  const getAlertColor = (level: RiskAlert['level']): string => {
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

  const getAlertTypeColor = (level: RiskAlert['level']): string => {
    switch (level) {
      case 'critical':
        return '#ff4d4f';
      case 'error':
        return '#ff7a45';
      case 'warning':
        return '#faad14';
      default:
        return '#1890ff';
    }
  };

  const formatRiskMetrics = (metrics: RiskAlert['metrics']) => {
    return (
      <Space direction="vertical" size="small">
        <Row gutter={16}>
          <Col span={8}>
            <Statistic
              title="日盈亏"
              value={metrics.dailyPnl}
              precision={2}
              prefix="¥"
              valueStyle={{ 
                color: metrics.dailyPnl >= 0 ? '#3f8600' : '#cf1322',
                fontSize: '14px'
              }}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title="当前回撤"
              value={metrics.currentDrawdown}
              precision={2}
              suffix="%"
              valueStyle={{ 
                color: '#cf1322',
                fontSize: '14px'
              }}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title="风险利用率"
              value={metrics.riskUtilization.dailyLossRatio * 100}
              precision={1}
              suffix="%"
              valueStyle={{ 
                color: metrics.riskUtilization.dailyLossRatio > 0.8 ? '#cf1322' : '#3f8600',
                fontSize: '14px'
              }}
            />
          </Col>
        </Row>
      </Space>
    );
  };

  const displayAlerts = alerts.slice(0, maxDisplayAlerts);
  const criticalAlerts = alerts.filter(alert => alert.level === 'critical').length;
  const errorAlerts = alerts.filter(alert => alert.level === 'error').length;
  const warningAlerts = alerts.filter(alert => alert.level === 'warning').length;

  if (!isConnected) {
    return (
      <Card title="风险告警监控">
        <Alert
          message="连接断开"
          description="WebSocket连接已断开，无法接收实时风险告警"
          type="warning"
          showIcon
        />
      </Card>
    );
  }

  if (error) {
    return (
      <Card title="风险告警监控">
        <Alert
          message="连接错误"
          description={error.message}
          type="error"
          showIcon
        />
      </Card>
    );
  }

  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      {/* Latest Alert */}
      {latestAlert && (
        <Alert
          message={`最新${latestAlert.level === 'critical' ? '严重' : latestAlert.level === 'error' ? '错误' : '警告'}告警`}
          description={
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>{latestAlert.message}</Text>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                时间: {new Date(latestAlert.timestamp).toLocaleString()}
              </Text>
              {showMetrics && (
                <div style={{ marginTop: '8px' }}>
                  {formatRiskMetrics(latestAlert.metrics)}
                </div>
              )}
            </Space>
          }
          type={getAlertColor(latestAlert.level)}
          icon={getAlertIcon(latestAlert.level)}
          showIcon
          closable
        />
      )}

      {/* Alert Statistics */}
      <Card 
        title="告警统计" 
        extra={
          <Space>
            <Button 
              size="small" 
              icon={<ClearOutlined />} 
              onClick={clearAlerts}
              disabled={alerts.length === 0}
            >
              清空告警
            </Button>
          </Space>
        }
      >
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="总告警数"
              value={alerts.length}
              valueStyle={{ fontSize: '16px' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="严重告警"
              value={criticalAlerts}
              valueStyle={{ 
                color: criticalAlerts > 0 ? '#ff4d4f' : '#666',
                fontSize: '16px'
              }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="错误告警"
              value={errorAlerts}
              valueStyle={{ 
                color: errorAlerts > 0 ? '#ff7a45' : '#666',
                fontSize: '16px'
              }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="警告告警"
              value={warningAlerts}
              valueStyle={{ 
                color: warningAlerts > 0 ? '#faad14' : '#666',
                fontSize: '16px'
              }}
            />
          </Col>
        </Row>
      </Card>

      {/* Alert List */}
      <Card title="告警历史">
        {displayAlerts.length === 0 ? (
          <Empty 
            description="暂无告警信息" 
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        ) : (
          <List
            itemLayout="vertical"
            dataSource={displayAlerts}
            renderItem={(alert, index) => (
              <List.Item
                key={`${alert.timestamp}-${index}`}
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
                extra={
                  <Tag color={getAlertTypeColor(alert.level)}>
                    {alert.level === 'critical' ? '严重' : 
                     alert.level === 'error' ? '错误' : '警告'}
                  </Tag>
                }
              >
                <List.Item.Meta
                  avatar={getAlertIcon(alert.level)}
                  title={
                    <Space>
                      <Text strong>{alert.message}</Text>
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {new Date(alert.timestamp).toLocaleString()}
                      </Text>
                    </Space>
                  }
                  description={
                    showMetrics && (
                      <div style={{ marginTop: '8px' }}>
                        {formatRiskMetrics(alert.metrics)}
                      </div>
                    )
                  }
                />
              </List.Item>
            )}
          />
        )}
        
        {alerts.length > maxDisplayAlerts && (
          <div style={{ textAlign: 'center', marginTop: '16px' }}>
            <Text type="secondary">
              显示最近 {maxDisplayAlerts} 条告警，共 {alerts.length} 条
            </Text>
          </div>
        )}
      </Card>
    </Space>
  );
};

export default RiskAlertsMonitor;