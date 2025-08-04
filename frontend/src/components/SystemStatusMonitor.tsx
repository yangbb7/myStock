import React from 'react';
import { Card, Badge, Statistic, Row, Col, Alert, Typography, Space } from 'antd';
import { 
  CheckCircleOutlined, 
  ExclamationCircleOutlined, 
  CloseCircleOutlined,
  SyncOutlined 
} from '@ant-design/icons';
import { useSystemStatus } from '../hooks/useRealTime';
import { SystemHealth, ModuleHealth } from '../services/types';

const { Text, Title } = Typography;

interface SystemStatusMonitorProps {
  showDetails?: boolean;
  refreshInterval?: number;
}

const SystemStatusMonitor: React.FC<SystemStatusMonitorProps> = ({ 
  showDetails = true,
  refreshInterval = 5000 
}) => {
  const { data, isConnected, error, lastUpdated } = useSystemStatus({
    autoSubscribe: true,
  });

  const getStatusColor = (running: boolean): string => {
    return running ? 'success' : 'error';
  };

  const getStatusIcon = (running: boolean) => {
    return running ? <CheckCircleOutlined /> : <CloseCircleOutlined />;
  };

  const getModuleStatusBadge = (module: ModuleHealth) => {
    if (module.initialized) {
      return <Badge status="success" text="运行中" />;
    } else {
      return <Badge status="error" text="未初始化" />;
    }
  };

  const formatUptime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours}小时${minutes}分钟`;
    } else if (minutes > 0) {
      return `${minutes}分钟${secs}秒`;
    } else {
      return `${secs}秒`;
    }
  };

  const getModuleMetrics = (module: ModuleHealth) => {
    const metrics = module.metrics || {};
    return Object.entries(metrics).map(([key, value]) => (
      <Text key={key} type="secondary" style={{ fontSize: '12px', display: 'block' }}>
        {key}: {typeof value === 'number' ? value.toFixed(2) : String(value)}
      </Text>
    ));
  };

  if (!isConnected) {
    return (
      <Card title="系统状态监控">
        <Alert
          message="连接断开"
          description="WebSocket连接已断开，正在尝试重新连接..."
          type="warning"
          icon={<SyncOutlined spin />}
          showIcon
        />
      </Card>
    );
  }

  if (error) {
    return (
      <Card title="系统状态监控">
        <Alert
          message="连接错误"
          description={error.message}
          type="error"
          showIcon
        />
      </Card>
    );
  }

  if (!data) {
    return (
      <Card title="系统状态监控" loading>
        <div style={{ height: '200px' }} />
      </Card>
    );
  }

  const runningModules = Object.values(data.modules).filter(m => m.initialized).length;
  const totalModules = Object.keys(data.modules).length;

  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      {/* Main System Status */}
      <Card 
        title="系统状态监控" 
        extra={
          <Space>
            <Badge 
              status={data.systemRunning ? 'processing' : 'error'} 
              text={data.systemRunning ? '运行中' : '已停止'} 
            />
            {lastUpdated && (
              <Text type="secondary" style={{ fontSize: '12px' }}>
                更新时间: {lastUpdated.toLocaleTimeString()}
              </Text>
            )}
          </Space>
        }
      >
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="系统状态"
              value={data.systemRunning ? '运行中' : '已停止'}
              valueStyle={{ 
                color: getStatusColor(data.systemRunning),
                fontSize: '16px'
              }}
              prefix={getStatusIcon(data.systemRunning)}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="运行时间"
              value={formatUptime(data.uptimeSeconds)}
              valueStyle={{ fontSize: '16px' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="活跃模块"
              value={runningModules}
              suffix={`/ ${totalModules}`}
              valueStyle={{ 
                color: runningModules === totalModules ? '#3f8600' : '#cf1322',
                fontSize: '16px'
              }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="连接状态"
              value="已连接"
              valueStyle={{ color: '#3f8600', fontSize: '16px' }}
              prefix={<CheckCircleOutlined />}
            />
          </Col>
        </Row>
      </Card>

      {/* Module Details */}
      {showDetails && (
        <Card title="模块状态详情">
          <Row gutter={[16, 16]}>
            {Object.entries(data.modules).map(([moduleName, moduleData]) => (
              <Col span={8} key={moduleName}>
                <Card 
                  size="small" 
                  title={
                    <Space>
                      <Title level={5} style={{ margin: 0 }}>
                        {moduleName.toUpperCase()}
                      </Title>
                      {getModuleStatusBadge(moduleData)}
                    </Space>
                  }
                  style={{ 
                    borderColor: moduleData.initialized ? '#52c41a' : '#ff4d4f',
                    borderWidth: '2px'
                  }}
                >
                  <Space direction="vertical" size="small" style={{ width: '100%' }}>
                    <div>
                      <Text strong>模块: </Text>
                      <Text>{moduleData.module}</Text>
                    </div>
                    <div>
                      <Text strong>状态: </Text>
                      <Text type={moduleData.initialized ? 'success' : 'danger'}>
                        {moduleData.initialized ? '已初始化' : '未初始化'}
                      </Text>
                    </div>
                    <div>
                      <Text strong>更新时间: </Text>
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {new Date(moduleData.timestamp).toLocaleTimeString()}
                      </Text>
                    </div>
                    {Object.keys(moduleData.metrics || {}).length > 0 && (
                      <div>
                        <Text strong>指标:</Text>
                        <div style={{ marginTop: '4px' }}>
                          {getModuleMetrics(moduleData)}
                        </div>
                      </div>
                    )}
                  </Space>
                </Card>
              </Col>
            ))}
          </Row>
        </Card>
      )}

      {/* System Alerts */}
      {!data.systemRunning && (
        <Alert
          message="系统警告"
          description="系统当前处于停止状态，某些功能可能不可用。"
          type="warning"
          icon={<ExclamationCircleOutlined />}
          showIcon
          closable
        />
      )}

      {runningModules < totalModules && (
        <Alert
          message="模块警告"
          description={`有 ${totalModules - runningModules} 个模块未正常运行，请检查系统配置。`}
          type="error"
          icon={<CloseCircleOutlined />}
          showIcon
          closable
        />
      )}
    </Space>
  );
};

export default SystemStatusMonitor;