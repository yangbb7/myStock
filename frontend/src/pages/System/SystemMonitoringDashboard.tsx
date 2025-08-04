import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Table,
  Alert,
  Button,
  Space,
  Typography,
  Tag,
  Badge,
  Tooltip,
  Select,
  DatePicker,
  Switch,
  notification,
} from 'antd';
import {
  DashboardOutlined,
  ApiOutlined,
  DatabaseOutlined,
  CloudServerOutlined,
  BellOutlined,
  ReloadOutlined,
  SettingOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
} from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import ReactECharts from 'echarts-for-react';
import { api } from '../../services/api';
import { SystemMetrics } from '../../services/types';
import { formatDateTime, formatBytes, formatNumber } from '../../utils/format';
import dayjs from 'dayjs';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;

interface PerformanceMetrics {
  timestamp: string;
  cpuUsage: number;
  memoryUsage: number;
  diskUsage: number;
  networkIn: number;
  networkOut: number;
  apiResponseTime: number;
  apiSuccessRate: number;
  cacheHitRate: number;
  activeConnections: number;
}

interface ApiEndpointMetrics {
  endpoint: string;
  method: string;
  totalRequests: number;
  successRequests: number;
  failedRequests: number;
  avgResponseTime: number;
  maxResponseTime: number;
  minResponseTime: number;
  successRate: number;
  lastAccessed: string;
}

interface SystemAlert {
  id: string;
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'critical';
  title: string;
  message: string;
  source: string;
  acknowledged: boolean;
  resolvedAt?: string;
}

const PerformanceChart: React.FC<{
  data: PerformanceMetrics[];
  metric: keyof PerformanceMetrics;
  title: string;
  unit?: string;
  color?: string;
}> = ({ data, metric, title, unit = '', color = '#1890ff' }) => {
  const chartData = data.map(item => ({
    time: dayjs(item.timestamp).format('HH:mm:ss'),
    value: item[metric] as number,
  }));

  const option = {
    title: {
      text: title,
      textStyle: {
        fontSize: 14,
        fontWeight: 'normal',
      },
    },
    tooltip: {
      trigger: 'axis',
      formatter: (params: any) => {
        const point = params[0];
        return `${point.name}<br/>${title}: ${point.value}${unit}`;
      },
    },
    xAxis: {
      type: 'category',
      data: chartData.map(item => item.time),
      axisLabel: {
        fontSize: 10,
      },
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        formatter: `{value}${unit}`,
        fontSize: 10,
      },
    },
    series: [
      {
        data: chartData.map(item => item.value),
        type: 'line',
        smooth: true,
        lineStyle: {
          color: color,
        },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: color + '40' },
              { offset: 1, color: color + '10' },
            ],
          },
        },
      },
    ],
    grid: {
      left: '10%',
      right: '5%',
      top: '15%',
      bottom: '15%',
    },
  };

  return (
    <ReactECharts
      option={option}
      style={{ height: '200px' }}
      opts={{ renderer: 'canvas' }}
    />
  );
};

const ApiMetricsTable: React.FC<{ data: ApiEndpointMetrics[] }> = ({ data }) => {
  const columns = [
    {
      title: 'API端点',
      dataIndex: 'endpoint',
      key: 'endpoint',
      render: (endpoint: string, record: ApiEndpointMetrics) => (
        <div>
          <Text code>{record.method}</Text> <Text>{endpoint}</Text>
        </div>
      ),
    },
    {
      title: '总请求数',
      dataIndex: 'totalRequests',
      key: 'totalRequests',
      render: (value: number) => formatNumber(value),
      sorter: (a: ApiEndpointMetrics, b: ApiEndpointMetrics) => a.totalRequests - b.totalRequests,
    },
    {
      title: '成功率',
      dataIndex: 'successRate',
      key: 'successRate',
      render: (rate: number) => (
        <div>
          <Progress
            percent={rate * 100}
            size="small"
            status={rate >= 0.95 ? 'success' : rate >= 0.9 ? 'normal' : 'exception'}
            format={() => `${(rate * 100).toFixed(1)}%`}
          />
        </div>
      ),
      sorter: (a: ApiEndpointMetrics, b: ApiEndpointMetrics) => a.successRate - b.successRate,
    },
    {
      title: '平均响应时间',
      dataIndex: 'avgResponseTime',
      key: 'avgResponseTime',
      render: (time: number) => (
        <Text style={{ color: time > 1000 ? '#ff4d4f' : time > 500 ? '#faad14' : '#52c41a' }}>
          {time}ms
        </Text>
      ),
      sorter: (a: ApiEndpointMetrics, b: ApiEndpointMetrics) => a.avgResponseTime - b.avgResponseTime,
    },
    {
      title: '最后访问',
      dataIndex: 'lastAccessed',
      key: 'lastAccessed',
      render: (timestamp: string) => (
        <Text type="secondary">{formatDateTime(timestamp, 'MM-DD HH:mm')}</Text>
      ),
    },
  ];

  return (
    <Table
      columns={columns}
      dataSource={data}
      rowKey="endpoint"
      size="small"
      pagination={{ pageSize: 10 }}
    />
  );
};

const SystemAlertsPanel: React.FC<{
  alerts: SystemAlert[];
  onAcknowledge: (alertId: string) => void;
  onResolve: (alertId: string) => void;
}> = ({ alerts, onAcknowledge, onResolve }) => {
  const getAlertIcon = (level: string) => {
    switch (level) {
      case 'critical':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'error':
        return <CloseCircleOutlined style={{ color: '#ff7875' }} />;
      case 'warning':
        return <WarningOutlined style={{ color: '#faad14' }} />;
      default:
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
    }
  };

  const getAlertColor = (level: string) => {
    switch (level) {
      case 'critical':
        return 'red';
      case 'error':
        return 'volcano';
      case 'warning':
        return 'orange';
      default:
        return 'blue';
    }
  };

  const unacknowledgedAlerts = alerts.filter(alert => !alert.acknowledged && !alert.resolvedAt);
  const acknowledgedAlerts = alerts.filter(alert => alert.acknowledged && !alert.resolvedAt);
  const resolvedAlerts = alerts.filter(alert => alert.resolvedAt);

  return (
    <div>
      {unacknowledgedAlerts.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <Title level={5}>未确认告警 ({unacknowledgedAlerts.length})</Title>
          <Space direction="vertical" style={{ width: '100%' }}>
            {unacknowledgedAlerts.map(alert => (
              <Alert
                key={alert.id}
                message={
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <Space>
                        {getAlertIcon(alert.level)}
                        <Text strong>{alert.title}</Text>
                        <Tag color={getAlertColor(alert.level)}>{alert.level.toUpperCase()}</Tag>
                        <Text type="secondary">{alert.source}</Text>
                      </Space>
                    </div>
                    <Space>
                      <Button size="small" onClick={() => onAcknowledge(alert.id)}>
                        确认
                      </Button>
                      <Button size="small" type="primary" onClick={() => onResolve(alert.id)}>
                        解决
                      </Button>
                    </Space>
                  </div>
                }
                description={
                  <div>
                    <Text>{alert.message}</Text>
                    <br />
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      {formatDateTime(alert.timestamp)}
                    </Text>
                  </div>
                }
                type={alert.level === 'critical' || alert.level === 'error' ? 'error' : 'warning'}
                showIcon={false}
                style={{ marginBottom: 8 }}
              />
            ))}
          </Space>
        </div>
      )}

      {acknowledgedAlerts.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <Title level={5}>已确认告警 ({acknowledgedAlerts.length})</Title>
          <Space direction="vertical" style={{ width: '100%' }}>
            {acknowledgedAlerts.slice(0, 3).map(alert => (
              <Alert
                key={alert.id}
                message={
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <Space>
                        {getAlertIcon(alert.level)}
                        <Text>{alert.title}</Text>
                        <Tag color={getAlertColor(alert.level)}>{alert.level.toUpperCase()}</Tag>
                        <Text type="secondary">{alert.source}</Text>
                      </Space>
                    </div>
                    <Button size="small" type="primary" onClick={() => onResolve(alert.id)}>
                      解决
                    </Button>
                  </div>
                }
                type="info"
                showIcon={false}
                style={{ marginBottom: 8 }}
              />
            ))}
          </Space>
        </div>
      )}
    </div>
  );
};

const SystemMonitoringDashboard: React.FC = () => {
  const [timeRange, setTimeRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    dayjs().subtract(1, 'hour'),
    dayjs(),
  ]);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30); // seconds

  // Query system metrics
  const {
    data: systemMetrics,
    isLoading: metricsLoading,
    error: metricsError,
    refetch: refetchMetrics,
  } = useQuery({
    queryKey: ['systemMetrics'],
    queryFn: () => api.system.getMetrics(),
    refetchInterval: autoRefresh ? refreshInterval * 1000 : false,
  });

  // Mock performance data - in real implementation, this would come from API
  const performanceData: PerformanceMetrics[] = Array.from({ length: 60 }, (_, i) => ({
    timestamp: dayjs().subtract(60 - i, 'minute').toISOString(),
    cpuUsage: 30 + Math.random() * 40,
    memoryUsage: 60 + Math.random() * 20,
    diskUsage: 45 + Math.random() * 10,
    networkIn: Math.random() * 100,
    networkOut: Math.random() * 80,
    apiResponseTime: 100 + Math.random() * 200,
    apiSuccessRate: 0.95 + Math.random() * 0.05,
    cacheHitRate: 0.8 + Math.random() * 0.2,
    activeConnections: 50 + Math.random() * 50,
  }));

  // Mock API metrics data
  const apiMetricsData: ApiEndpointMetrics[] = [
    {
      endpoint: '/health',
      method: 'GET',
      totalRequests: 15420,
      successRequests: 15418,
      failedRequests: 2,
      avgResponseTime: 45,
      maxResponseTime: 120,
      minResponseTime: 20,
      successRate: 0.9999,
      lastAccessed: dayjs().subtract(1, 'minute').toISOString(),
    },
    {
      endpoint: '/metrics',
      method: 'GET',
      totalRequests: 8760,
      successRequests: 8755,
      failedRequests: 5,
      avgResponseTime: 180,
      maxResponseTime: 500,
      minResponseTime: 80,
      successRate: 0.9994,
      lastAccessed: dayjs().subtract(2, 'minutes').toISOString(),
    },
    {
      endpoint: '/data/market/{symbol}',
      method: 'GET',
      totalRequests: 25680,
      successRequests: 25450,
      failedRequests: 230,
      avgResponseTime: 320,
      maxResponseTime: 2000,
      minResponseTime: 150,
      successRate: 0.991,
      lastAccessed: dayjs().subtract(30, 'seconds').toISOString(),
    },
    {
      endpoint: '/strategy/performance',
      method: 'GET',
      totalRequests: 4320,
      successRequests: 4315,
      failedRequests: 5,
      avgResponseTime: 250,
      maxResponseTime: 800,
      minResponseTime: 100,
      successRate: 0.9988,
      lastAccessed: dayjs().subtract(5, 'minutes').toISOString(),
    },
  ];

  // Mock system alerts
  const [systemAlerts, setSystemAlerts] = useState<SystemAlert[]>([
    {
      id: '1',
      timestamp: dayjs().subtract(10, 'minutes').toISOString(),
      level: 'warning',
      title: 'API响应时间过长',
      message: '/data/market API平均响应时间超过500ms',
      source: 'API监控',
      acknowledged: false,
    },
    {
      id: '2',
      timestamp: dayjs().subtract(30, 'minutes').toISOString(),
      level: 'error',
      title: '数据库连接异常',
      message: '数据库连接池使用率达到90%',
      source: '数据库监控',
      acknowledged: true,
    },
    {
      id: '3',
      timestamp: dayjs().subtract(1, 'hour').toISOString(),
      level: 'info',
      title: '系统重启完成',
      message: '系统维护重启已完成，所有模块正常运行',
      source: '系统管理',
      acknowledged: true,
      resolvedAt: dayjs().subtract(50, 'minutes').toISOString(),
    },
  ]);

  const handleAcknowledgeAlert = (alertId: string) => {
    setSystemAlerts(prev =>
      prev.map(alert =>
        alert.id === alertId ? { ...alert, acknowledged: true } : alert
      )
    );
    notification.success({
      message: '告警已确认',
      description: '告警已标记为已确认状态',
    });
  };

  const handleResolveAlert = (alertId: string) => {
    setSystemAlerts(prev =>
      prev.map(alert =>
        alert.id === alertId
          ? { ...alert, acknowledged: true, resolvedAt: dayjs().toISOString() }
          : alert
      )
    );
    notification.success({
      message: '告警已解决',
      description: '告警已标记为已解决状态',
    });
  };

  const latestMetrics = performanceData[performanceData.length - 1];
  const unacknowledgedAlertsCount = systemAlerts.filter(
    alert => !alert.acknowledged && !alert.resolvedAt
  ).length;

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>系统监控仪表板</Title>
        <Space>
          <Text>自动刷新:</Text>
          <Switch
            checked={autoRefresh}
            onChange={setAutoRefresh}
            size="small"
          />
          <Select
            value={refreshInterval}
            onChange={setRefreshInterval}
            size="small"
            style={{ width: 80 }}
            disabled={!autoRefresh}
          >
            <Select.Option value={10}>10s</Select.Option>
            <Select.Option value={30}>30s</Select.Option>
            <Select.Option value={60}>1m</Select.Option>
          </Select>
          <Button icon={<ReloadOutlined />} onClick={() => refetchMetrics()}>
            刷新
          </Button>
        </Space>
      </div>

      {/* Key Metrics Overview */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="CPU使用率"
              value={latestMetrics.cpuUsage}
              precision={1}
              suffix="%"
              prefix={<DashboardOutlined />}
              valueStyle={{
                color: latestMetrics.cpuUsage > 80 ? '#ff4d4f' : '#3f8600',
              }}
            />
            <Progress
              percent={latestMetrics.cpuUsage}
              size="small"
              status={latestMetrics.cpuUsage > 80 ? 'exception' : 'normal'}
              showInfo={false}
              style={{ marginTop: 8 }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="内存使用率"
              value={latestMetrics.memoryUsage}
              precision={1}
              suffix="%"
              prefix={<DatabaseOutlined />}
              valueStyle={{
                color: latestMetrics.memoryUsage > 85 ? '#ff4d4f' : '#3f8600',
              }}
            />
            <Progress
              percent={latestMetrics.memoryUsage}
              size="small"
              status={latestMetrics.memoryUsage > 85 ? 'exception' : 'normal'}
              showInfo={false}
              style={{ marginTop: 8 }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="API响应时间"
              value={latestMetrics.apiResponseTime}
              precision={0}
              suffix="ms"
              prefix={<ApiOutlined />}
              valueStyle={{
                color: latestMetrics.apiResponseTime > 500 ? '#ff4d4f' : '#3f8600',
              }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃连接数"
              value={latestMetrics.activeConnections}
              precision={0}
              prefix={<CloudServerOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Performance Charts */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={12}>
          <Card title="CPU & 内存使用率">
            <div style={{ height: 200 }}>
              <PerformanceChart
                data={performanceData}
                metric="cpuUsage"
                title="CPU使用率"
                unit="%"
                color="#1890ff"
              />
            </div>
          </Card>
        </Col>
        <Col span={12}>
          <Card title="API性能监控">
            <div style={{ height: 200 }}>
              <PerformanceChart
                data={performanceData}
                metric="apiResponseTime"
                title="API响应时间"
                unit="ms"
                color="#52c41a"
              />
            </div>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={12}>
          <Card title="网络流量">
            <div style={{ height: 200 }}>
              <PerformanceChart
                data={performanceData}
                metric="networkIn"
                title="网络入流量"
                unit="MB/s"
                color="#faad14"
              />
            </div>
          </Card>
        </Col>
        <Col span={12}>
          <Card title="缓存命中率">
            <div style={{ height: 200 }}>
              <PerformanceChart
                data={performanceData}
                metric="cacheHitRate"
                title="缓存命中率"
                unit="%"
                color="#722ed1"
              />
            </div>
          </Card>
        </Col>
      </Row>

      {/* API Metrics Table */}
      <Card title="API端点性能统计" style={{ marginBottom: 24 }}>
        <ApiMetricsTable data={apiMetricsData} />
      </Card>

      {/* System Alerts */}
      <Card
        title={
          <Space>
            <BellOutlined />
            <span>系统告警</span>
            {unacknowledgedAlertsCount > 0 && (
              <Badge count={unacknowledgedAlertsCount} />
            )}
          </Space>
        }
      >
        <SystemAlertsPanel
          alerts={systemAlerts}
          onAcknowledge={handleAcknowledgeAlert}
          onResolve={handleResolveAlert}
        />
      </Card>
    </div>
  );
};

export default SystemMonitoringDashboard;