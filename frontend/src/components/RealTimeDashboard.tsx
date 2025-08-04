import React, { useState } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Tabs, 
  Switch, 
  Space, 
  Typography, 
  Badge,
  Button,
  Tooltip
} from 'antd';
import { 
  DashboardOutlined, 
  AlertOutlined, 
  LineChartOutlined,
  SettingOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import SystemStatusMonitor from './SystemStatusMonitor';
import RiskAlertsMonitor from './RiskAlertsMonitor';
import { useRealTime } from '../hooks/useRealTime';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface RealTimeDashboardProps {
  defaultTab?: string;
  showSettings?: boolean;
}

const RealTimeDashboard: React.FC<RealTimeDashboardProps> = ({
  defaultTab = 'system',
  showSettings = true
}) => {
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [showDetails, setShowDetails] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5000);

  const { 
    systemStatus, 
    riskAlerts, 
    isConnected, 
    error 
  } = useRealTime({
    systemStatus: { autoSubscribe: autoRefresh },
    riskAlerts: { autoSubscribe: autoRefresh }
  });

  const getConnectionBadge = () => {
    if (isConnected) {
      return <Badge status="success" text="已连接" />;
    } else {
      return <Badge status="error" text="连接断开" />;
    }
  };

  const handleRefresh = () => {
    // Force reconnection if needed
    if (!isConnected) {
      window.location.reload();
    }
  };

  const settingsPanel = showSettings && (
    <Card size="small" title="设置" style={{ marginBottom: '16px' }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Text>自动刷新</Text>
          <Switch 
            checked={autoRefresh} 
            onChange={setAutoRefresh}
            size="small"
          />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Text>显示详情</Text>
          <Switch 
            checked={showDetails} 
            onChange={setShowDetails}
            size="small"
          />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Text>连接状态</Text>
          {getConnectionBadge()}
        </div>
      </Space>
    </Card>
  );

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Title level={2} style={{ margin: 0 }}>
              <DashboardOutlined style={{ marginRight: '8px' }} />
              实时监控面板
            </Title>
          </Col>
          <Col>
            <Space>
              {getConnectionBadge()}
              <Tooltip title="刷新连接">
                <Button 
                  icon={<ReloadOutlined />} 
                  onClick={handleRefresh}
                  loading={!isConnected}
                >
                  刷新
                </Button>
              </Tooltip>
            </Space>
          </Col>
        </Row>
      </div>

      <Row gutter={[24, 24]}>
        {/* Settings Panel */}
        {showSettings && (
          <Col span={6}>
            {settingsPanel}
          </Col>
        )}

        {/* Main Content */}
        <Col span={showSettings ? 18 : 24}>
          <Tabs 
            defaultActiveKey={defaultTab}
            type="card"
            size="large"
          >
            <TabPane
              tab={
                <Space>
                  <DashboardOutlined />
                  <span>系统状态</span>
                  {systemStatus.data && (
                    <Badge 
                      status={systemStatus.data.systemRunning ? 'success' : 'error'} 
                    />
                  )}
                </Space>
              }
              key="system"
            >
              <SystemStatusMonitor 
                showDetails={showDetails}
                refreshInterval={refreshInterval}
              />
            </TabPane>

            <TabPane
              tab={
                <Space>
                  <AlertOutlined />
                  <span>风险告警</span>
                  {riskAlerts.alerts.length > 0 && (
                    <Badge 
                      count={riskAlerts.alerts.length} 
                      size="small"
                      style={{ backgroundColor: '#ff4d4f' }}
                    />
                  )}
                </Space>
              }
              key="risk"
            >
              <RiskAlertsMonitor 
                maxDisplayAlerts={10}
                showMetrics={showDetails}
                autoRefresh={autoRefresh}
              />
            </TabPane>

            <TabPane
              tab={
                <Space>
                  <LineChartOutlined />
                  <span>市场数据</span>
                </Space>
              }
              key="market"
            >
              <Card>
                <div style={{ textAlign: 'center', padding: '40px' }}>
                  <LineChartOutlined style={{ fontSize: '48px', color: '#ccc' }} />
                  <div style={{ marginTop: '16px' }}>
                    <Text type="secondary">市场数据监控组件将在后续任务中实现</Text>
                  </div>
                </div>
              </Card>
            </TabPane>
          </Tabs>
        </Col>
      </Row>

      {/* Error Display */}
      {error && (
        <div style={{ position: 'fixed', bottom: '24px', right: '24px', zIndex: 1000 }}>
          <Card 
            size="small" 
            style={{ 
              backgroundColor: '#fff2f0', 
              borderColor: '#ffccc7',
              maxWidth: '300px'
            }}
          >
            <Space direction="vertical">
              <Text type="danger" strong>连接错误</Text>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                {error.message}
              </Text>
            </Space>
          </Card>
        </div>
      )}
    </div>
  );
};

export default RealTimeDashboard;