import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Badge,
  Table,
  Alert,
  Button,
  Space,
  Tooltip,
  Typography,
  Divider,
  Tag,
} from 'antd';
import {
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  ReloadOutlined,
  InfoCircleOutlined,
  WarningOutlined,
} from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import { api } from '../../services/api';
import { SystemHealth, SystemMetrics, ModuleHealth } from '../../services/types';
import { formatUptime, formatBytes, formatNumber } from '../../utils/format';

const { Title, Text } = Typography;

interface SystemResourceMetrics {
  cpuUsage: number;
  memoryUsage: number;
  memoryTotal: number;
  diskUsage: number;
  diskTotal: number;
  networkIn: number;
  networkOut: number;
}

interface ModuleStatusProps {
  name: string;
  health: ModuleHealth;
}

const ModuleStatusCard: React.FC<ModuleStatusProps> = ({ name, health }) => {
  const getStatusColor = (initialized: boolean) => {
    return initialized ? 'success' : 'error';
  };

  const getStatusIcon = (initialized: boolean) => {
    return initialized ? <CheckCircleOutlined /> : <CloseCircleOutlined />;
  };

  return (
    <Card size="small" className="module-status-card">
      <Space direction="vertical" size="small" style={{ width: '100%' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Text strong>{name}</Text>
          <Badge
            status={getStatusColor(health.initialized)}
            text={health.initialized ? '运行中' : '已停止'}
          />
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {getStatusIcon(health.initialized)}
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {new Date(health.timestamp).toLocaleTimeString()}
          </Text>
        </div>

        {health.metrics && Object.keys(health.metrics).length > 0 && (
          <div>
            <Divider style={{ margin: '8px 0' }} />
            <Space wrap size="small">
              {Object.entries(health.metrics).map(([key, value]) => (
                <Tag key={key} color="blue" style={{ fontSize: '10px' }}>
                  {key}: {typeof value === 'number' ? formatNumber(value) : String(value)}
                </Tag>
              ))}
            </Space>
          </div>
        )}
      </Space>
    </Card>
  );
};

const SystemResourcesCard: React.FC<{ metrics?: SystemResourceMetrics }> = ({ metrics }) => {
  if (!metrics) {
    return (
      <Card title="系统资源" loading>
        <div style={{ height: 200 }} />
      </Card>
    );
  }

  return (
    <Card title="系统资源使用情况">
      <Row gutter={[16, 16]}>
        <Col span={12}>
          <div>
            <Text strong>CPU 使用率</Text>
            <Progress
              percent={metrics.cpuUsage}
              status={metrics.cpuUsage > 80 ? 'exception' : 'normal'}
              strokeColor={metrics.cpuUsage > 80 ? '#ff4d4f' : '#52c41a'}
            />
          </div>
        </Col>
        <Col span={12}>
          <div>
            <Text strong>内存使用率</Text>
            <Progress
              percent={(metrics.memoryUsage / metrics.memoryTotal) * 100}
              status={(metrics.memoryUsage / metrics.memoryTotal) > 0.8 ? 'exception' : 'normal'}
              strokeColor={(metrics.memoryUsage / metrics.memoryTotal) > 0.8 ? '#ff4d4f' : '#52c41a'}
              format={() => `${formatBytes(metrics.memoryUsage)} / ${formatBytes(metrics.memoryTotal)}`}
            />
          </div>
        </Col>
        <Col span={12}>
          <div>
            <Text strong>磁盘使用率</Text>
            <Progress
              percent={(metrics.diskUsage / metrics.diskTotal) * 100}
              status={(metrics.diskUsage / metrics.diskTotal) > 0.9 ? 'exception' : 'normal'}
              strokeColor={(metrics.diskUsage / metrics.diskTotal) > 0.9 ? '#ff4d4f' : '#52c41a'}
              format={() => `${formatBytes(metrics.diskUsage)} / ${formatBytes(metrics.diskTotal)}`}
            />
          </div>
        </Col>
        <Col span={12}>
          <div>
            <Text strong>网络流量</Text>
            <div style={{ marginTop: 8 }}>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                ↑ {formatBytes(metrics.networkOut)}/s ↓ {formatBytes(metrics.networkIn)}/s
              </Text>
            </div>
          </div>
        </Col>
      </Row>
    </Card>
  );
};

const SystemHealthDiagnostics: React.FC<{ health: SystemHealth }> = ({ health }) => {
  const diagnostics = [];

  // Check system running status
  if (!health.systemRunning) {
    diagnostics.push({
      level: 'error',
      message: '系统未运行',
      suggestion: '请启动系统以开始交易',
    });
  }

  // Check module status
  const failedModules = Object.entries(health.modules).filter(
    ([_, module]) => !module.initialized
  );

  if (failedModules.length > 0) {
    diagnostics.push({
      level: 'warning',
      message: `${failedModules.length} 个模块未初始化`,
      suggestion: `检查模块: ${failedModules.map(([name]) => name).join(', ')}`,
    });
  }

  // Check uptime
  if (health.uptimeSeconds < 300) { // Less than 5 minutes
    diagnostics.push({
      level: 'info',
      message: '系统刚刚启动',
      suggestion: '等待系统完全初始化',
    });
  }

  const getAlertType = (level: string) => {
    switch (level) {
      case 'error': return 'error';
      case 'warning': return 'warning';
      case 'info': return 'info';
      default: return 'info';
    }
  };

  const getAlertIcon = (level: string) => {
    switch (level) {
      case 'error': return <CloseCircleOutlined />;
      case 'warning': return <WarningOutlined />;
      case 'info': return <InfoCircleOutlined />;
      default: return <InfoCircleOutlined />;
    }
  };

  if (diagnostics.length === 0) {
    return (
      <Card title="系统诊断">
        <Alert
          message="系统运行正常"
          description="所有模块运行正常，无异常情况"
          type="success"
          icon={<CheckCircleOutlined />}
          showIcon
        />
      </Card>
    );
  }

  return (
    <Card title="系统诊断">
      <Space direction="vertical" size="middle" style={{ width: '100%' }}>
        {diagnostics.map((diagnostic, index) => (
          <Alert
            key={index}
            message={diagnostic.message}
            description={diagnostic.suggestion}
            type={getAlertType(diagnostic.level)}
            icon={getAlertIcon(diagnostic.level)}
            showIcon
          />
        ))}
      </Space>
    </Card>
  );
};

const SystemStatusMonitor: React.FC = () => {
  const [refreshKey, setRefreshKey] = useState(0);

  // Query system health
  const {
    data: systemHealth,
    isLoading: healthLoading,
    error: healthError,
    refetch: refetchHealth,
  } = useQuery({
    queryKey: ['systemHealth', refreshKey],
    queryFn: () => api.system.getHealth(),
    refetchInterval: 5000, // Refresh every 5 seconds
    retry: 3,
  });

  // Query system metrics
  const {
    data: systemMetrics,
    isLoading: metricsLoading,
    error: metricsError,
    refetch: refetchMetrics,
  } = useQuery({
    queryKey: ['systemMetrics', refreshKey],
    queryFn: () => api.system.getMetrics(),
    refetchInterval: 10000, // Refresh every 10 seconds
    retry: 3,
  });

  const handleRefresh = () => {
    setRefreshKey(prev => prev + 1);
    refetchHealth();
    refetchMetrics();
  };

  const moduleColumns = [
    {
      title: '模块名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string) => <Text strong>{name}</Text>,
    },
    {
      title: '状态',
      dataIndex: 'initialized',
      key: 'status',
      render: (initialized: boolean) => (
        <Badge
          status={initialized ? 'success' : 'error'}
          text={initialized ? '运行中' : '已停止'}
        />
      ),
    },
    {
      title: '最后更新',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: string) => (
        <Text type="secondary">
          {new Date(timestamp).toLocaleString()}
        </Text>
      ),
    },
    {
      title: '指标数量',
      dataIndex: 'metrics',
      key: 'metricsCount',
      render: (metrics: Record<string, any>) => (
        <Text>{Object.keys(metrics || {}).length}</Text>
      ),
    },
  ];

  const moduleData = systemHealth?.modules
    ? Object.entries(systemHealth.modules).map(([name, health]) => ({
        key: name,
        name,
        ...health,
      }))
    : [];

  // Mock system resources data (in real implementation, this would come from the API)
  const systemResources: SystemResourceMetrics = {
    cpuUsage: 45,
    memoryUsage: 2.1 * 1024 * 1024 * 1024, // 2.1 GB
    memoryTotal: 8 * 1024 * 1024 * 1024, // 8 GB
    diskUsage: 120 * 1024 * 1024 * 1024, // 120 GB
    diskTotal: 500 * 1024 * 1024 * 1024, // 500 GB
    networkIn: 1024 * 1024, // 1 MB/s
    networkOut: 512 * 1024, // 512 KB/s
  };

  if (healthError || metricsError) {
    return (
      <div style={{ padding: 24 }}>
        <Alert
          message="系统状态获取失败"
          description="无法获取系统状态信息，请检查网络连接或联系管理员"
          type="error"
          showIcon
          action={
            <Button size="small" onClick={handleRefresh}>
              重试
            </Button>
          }
        />
      </div>
    );
  }

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>系统状态监控</Title>
        <Button
          icon={<ReloadOutlined />}
          onClick={handleRefresh}
          loading={healthLoading || metricsLoading}
        >
          刷新
        </Button>
      </div>

      {/* System Overview */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="系统状态"
              value={systemHealth?.systemRunning ? '运行中' : '已停止'}
              valueStyle={{
                color: systemHealth?.systemRunning ? '#3f8600' : '#cf1322',
              }}
              prefix={
                systemHealth?.systemRunning ? (
                  <CheckCircleOutlined />
                ) : (
                  <CloseCircleOutlined />
                )
              }
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="运行时间"
              value={systemHealth?.uptimeSeconds ? formatUptime(systemHealth.uptimeSeconds) : '0s'}
              prefix={<InfoCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃模块"
              value={
                systemHealth?.modules
                  ? Object.values(systemHealth.modules).filter(m => m.initialized).length
                  : 0
              }
              suffix={`/ ${systemHealth?.modules ? Object.keys(systemHealth.modules).length : 0}`}
              prefix={<InfoCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="最后更新"
              value={systemHealth?.timestamp ? new Date(systemHealth.timestamp).toLocaleTimeString() : '--'}
              prefix={<InfoCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Module Status Grid */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        {systemHealth?.modules &&
          Object.entries(systemHealth.modules).map(([name, health]) => (
            <Col key={name} span={8}>
              <ModuleStatusCard name={name} health={health} />
            </Col>
          ))}
      </Row>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        {/* System Resources */}
        <Col span={12}>
          <SystemResourcesCard metrics={systemResources} />
        </Col>

        {/* System Diagnostics */}
        <Col span={12}>
          {systemHealth && <SystemHealthDiagnostics health={systemHealth} />}
        </Col>
      </Row>

      {/* Detailed Module Table */}
      <Card title="模块详细状态" style={{ marginBottom: 24 }}>
        <Table
          columns={moduleColumns}
          dataSource={moduleData}
          loading={healthLoading}
          pagination={false}
          size="small"
        />
      </Card>
    </div>
  );
};

export default SystemStatusMonitor;