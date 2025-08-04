import React from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Statistic, 
  Badge, 
  Alert, 
  Typography, 
  Space, 
  Progress,
  Tooltip,
  Spin
} from 'antd';
import { 
  CheckCircleOutlined, 
  ExclamationCircleOutlined, 
  CloseCircleOutlined,
  SyncOutlined,
  ClockCircleOutlined,
  DatabaseOutlined,
  ApiOutlined,
  BarChartOutlined,
  SafetyCertificateOutlined,
  DollarCircleOutlined,
  LineChartOutlined
} from '@ant-design/icons';
import { useSystemHealth, useSystemMetrics } from '../../../hooks/useApi';
import { useSystemStatus } from '../../../hooks/useRealTime';
import { formatUptime, formatNumber } from '../../../utils/format';

const { Text, Title } = Typography;

interface ModuleIconMap {
  [key: string]: React.ReactNode;
}

const moduleIcons: ModuleIconMap = {
  data: <DatabaseOutlined />,
  strategy: <BarChartOutlined />,
  execution: <ApiOutlined />,
  risk: <SafetyCertificateOutlined />,
  portfolio: <DollarCircleOutlined />,
  analytics: <LineChartOutlined />,
};

const SystemStatusCard: React.FC = () => {
  // Use both API and real-time data for comprehensive status
  const { data: healthData, isLoading: healthLoading, error: healthError } = useSystemHealth();
  const { data: metricsData, isLoading: metricsLoading } = useSystemMetrics();
  const { data: realtimeData, isConnected, error: realtimeError, lastUpdated } = useSystemStatus();

  // Use real-time data if available, fallback to API data
  const systemData = realtimeData || healthData;
  const isLoading = healthLoading || metricsLoading;
  const hasError = healthError || realtimeError;

  const getStatusColor = (running: boolean): string => {
    return running ? '#52c41a' : '#ff4d4f';
  };

  const getStatusIcon = (running: boolean) => {
    return running ? <CheckCircleOutlined /> : <CloseCircleOutlined />;
  };

  const getModuleStatusBadge = (initialized: boolean) => {
    return (
      <Badge 
        status={initialized ? 'success' : 'error'} 
        text={initialized ? '运行中' : '未初始化'} 
      />
    );
  };

  const getModuleHealthScore = (moduleData: any): number => {
    if (!moduleData?.initialized) return 0;
    
    // Calculate health score based on metrics
    const metrics = moduleData.metrics || {};
    let score = 100;
    
    // Reduce score based on error rates, response times, etc.
    if (metrics.errorRate && metrics.errorRate > 0.01) {
      score -= Math.min(50, metrics.errorRate * 1000);
    }
    
    if (metrics.responseTime && metrics.responseTime > 100) {
      score -= Math.min(30, (metrics.responseTime - 100) / 10);
    }
    
    return Math.max(0, Math.round(score));
  };

  const renderModuleCard = (moduleName: string, moduleData: any) => {
    const healthScore = getModuleHealthScore(moduleData);
    const isHealthy = healthScore >= 80;
    
    return (
      <Col span={8} key={moduleName}>
        <Card 
          size="small"
          style={{ 
            borderColor: moduleData.initialized ? '#52c41a' : '#ff4d4f',
            borderWidth: '1px',
            height: '100%'
          }}
        >
          <Space direction="vertical" size="small" style={{ width: '100%' }}>
            {/* Module Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Space>
                {moduleIcons[moduleName] || <ApiOutlined />}
                <Title level={5} style={{ margin: 0 }}>
                  {moduleName.toUpperCase()}
                </Title>
              </Space>
              {getModuleStatusBadge(moduleData.initialized)}
            </div>
            
            {/* Health Score */}
            <div>
              <Text type="secondary" style={{ fontSize: '12px' }}>健康度</Text>
              <Progress 
                percent={healthScore} 
                size="small" 
                status={isHealthy ? 'success' : healthScore > 50 ? 'active' : 'exception'}
                showInfo={false}
              />
              <Text style={{ fontSize: '12px', color: isHealthy ? '#52c41a' : '#ff4d4f' }}>
                {healthScore}%
              </Text>
            </div>
            
            {/* Module Metrics */}
            {Object.keys(moduleData.metrics || {}).length > 0 && (
              <div>
                <Text type="secondary" style={{ fontSize: '12px' }}>关键指标:</Text>
                <div style={{ marginTop: '4px' }}>
                  {Object.entries(moduleData.metrics || {}).slice(0, 3).map(([key, value]) => (
                    <div key={key} style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Text style={{ fontSize: '11px' }}>{key}:</Text>
                      <Text style={{ fontSize: '11px', fontWeight: 'bold' }}>
                        {typeof value === 'number' ? formatNumber(value) : String(value)}
                      </Text>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {/* Last Update */}
            <div>
              <Text type="secondary" style={{ fontSize: '10px' }}>
                更新: {new Date(moduleData.timestamp).toLocaleTimeString()}
              </Text>
            </div>
          </Space>
        </Card>
      </Col>
    );
  };

  if (isLoading) {
    return (
      <Card title="系统状态监控" loading>
        <div style={{ height: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          <Spin size="large" />
        </div>
      </Card>
    );
  }

  if (hasError && !systemData) {
    return (
      <Card title="系统状态监控">
        <Alert
          message="连接错误"
          description={hasError?.message || '无法获取系统状态信息'}
          type="error"
          showIcon
        />
      </Card>
    );
  }

  if (!systemData) {
    return (
      <Card title="系统状态监控">
        <Alert
          message="暂无数据"
          description="系统状态数据暂不可用"
          type="warning"
          showIcon
        />
      </Card>
    );
  }

  const runningModules = systemData?.modules ? Object.values(systemData.modules).filter((m: any) => m.initialized).length : 0;
  const totalModules = systemData?.modules ? Object.keys(systemData.modules).length : 0;
  const systemHealthScore = Math.round((runningModules / totalModules) * 100);

  return (
    <Card 
      title={
        <Space>
          <Title level={4} style={{ margin: 0 }}>系统状态监控</Title>
          <Badge 
            status={systemData?.systemRunning ? 'processing' : 'error'} 
            text={systemData?.systemRunning ? '运行中' : '已停止'} 
          />
        </Space>
      }
      extra={
        <Space>
          {/* Connection Status */}
          <Tooltip title={isConnected ? '实时连接正常' : '实时连接断开'}>
            <Badge 
              status={isConnected ? 'success' : 'error'} 
              text={isConnected ? '实时' : '离线'} 
            />
          </Tooltip>
          
          {/* Last Update Time */}
          {lastUpdated && (
            <Text type="secondary" style={{ fontSize: '12px' }}>
              <ClockCircleOutlined /> {lastUpdated.toLocaleTimeString()}
            </Text>
          )}
        </Space>
      }
    >
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* System Overview Statistics */}
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="系统状态"
              value={systemData?.systemRunning ? '运行中' : '已停止'}
              valueStyle={{ 
                color: getStatusColor(systemData?.systemRunning || false),
                fontSize: '18px'
              }}
              prefix={getStatusIcon(systemData?.systemRunning || false)}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="运行时间"
              value={formatUptime(systemData?.uptimeSeconds || 0)}
              valueStyle={{ fontSize: '18px' }}
              prefix={<ClockCircleOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="活跃模块"
              value={runningModules}
              suffix={`/ ${totalModules}`}
              valueStyle={{ 
                color: runningModules === totalModules ? '#52c41a' : '#ff4d4f',
                fontSize: '18px'
              }}
              prefix={<ApiOutlined />}
            />
          </Col>
          <Col span={6}>
            <div>
              <Text type="secondary">系统健康度</Text>
              <div style={{ marginTop: '8px' }}>
                <Progress 
                  type="circle" 
                  percent={systemHealthScore} 
                  size={60}
                  status={systemHealthScore >= 80 ? 'success' : systemHealthScore >= 50 ? 'active' : 'exception'}
                />
              </div>
            </div>
          </Col>
        </Row>

        {/* System Performance Metrics */}
        {metricsData && (
          <Row gutter={16}>
            <Col span={6}>
              <Statistic
                title="API响应时间"
                value={0}
                suffix="ms"
                precision={1}
                valueStyle={{ fontSize: '16px' }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="内存使用率"
                value={0}
                suffix="%"
                precision={1}
                valueStyle={{ 
                  fontSize: '16px',
                  color: '#52c41a'
                }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="缓存命中率"
                value={0}
                suffix="%"
                precision={1}
                valueStyle={{ fontSize: '16px' }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="处理队列"
                value={0}
                valueStyle={{ fontSize: '16px' }}
              />
            </Col>
          </Row>
        )}

        {/* Module Status Grid */}
        <div>
          <Title level={5}>模块状态详情</Title>
          <Row gutter={[16, 16]}>
            {systemData?.modules ? Object.entries(systemData.modules).map(([moduleName, moduleData]) =>
              renderModuleCard(moduleName, moduleData)
            ) : (
              <Col span={24}>
                <Text type="secondary">模块信息加载中...</Text>
              </Col>
            )}
          </Row>
        </div>

        {/* System Alerts */}
        {!systemData?.systemRunning && (
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

        {!isConnected && (
          <Alert
            message="连接警告"
            description="实时数据连接已断开，显示的可能不是最新状态。"
            type="warning"
            icon={<SyncOutlined spin />}
            showIcon
            closable
          />
        )}
      </Space>
    </Card>
  );
};

export default SystemStatusCard;