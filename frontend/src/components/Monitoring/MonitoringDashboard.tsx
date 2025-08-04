/**
 * 监控仪表板组件
 * 提供实时监控数据展示和分析
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Table,
  Alert,
  Tabs,
  Button,
  Space,
  Tag,
  Timeline,
  Tooltip,
  Switch,
  Select,
  DatePicker,
  message,
} from 'antd';
import {
  BugOutlined,
  DashboardOutlined,
  UserOutlined,
  BarChartOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  DownloadOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import { useMonitoring } from '@/hooks/useMonitoring';
import { monitoring, logger } from '@/services/monitoring';
import { performanceMonitor } from '@/utils/performanceMonitor';
import type { LogLevel } from '@/services/monitoring/logger';
import dayjs from 'dayjs';

const { TabPane } = Tabs;
const { RangePicker } = DatePicker;

interface MonitoringStats {
  errors: {
    total: number;
    byLevel: Record<string, number>;
    recent: Array<{
      message: string;
      level: string;
      timestamp: number;
      count: number;
    }>;
  };
  performance: {
    averageLoadTime: number;
    memoryUsage: number;
    apiResponseTime: number;
    cacheHitRate: number;
  };
  userActivity: {
    activeUsers: number;
    pageViews: number;
    interactions: number;
    sessionDuration: number;
  };
  system: {
    uptime: number;
    errorRate: number;
    performanceScore: number;
    healthStatus: 'healthy' | 'warning' | 'critical';
  };
}

export const MonitoringDashboard: React.FC = () => {
  const [stats, setStats] = useState<MonitoringStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30); // seconds
  const [logLevel, setLogLevel] = useState<LogLevel>('info');
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    dayjs().subtract(1, 'hour'),
    dayjs(),
  ]);

  const monitoringHook = useMonitoring({
    trackPageViews: false,
    trackUserInteractions: false,
  });

  // 获取监控统计数据
  const fetchStats = async () => {
    try {
      setLoading(true);
      
      // 获取各种监控数据
      const [
        monitoringStats,
        performanceReport,
        loggerStats,
      ] = await Promise.all([
        monitoring.getStats(),
        performanceMonitor.getPerformanceReport(),
        logger.getStats(),
      ]);

      // 模拟获取更多统计数据（实际应用中应该从API获取）
      const mockStats: MonitoringStats = {
        errors: {
          total: loggerStats?.levelCounts?.error || 0,
          byLevel: loggerStats?.levelCounts || {},
          recent: [
            {
              message: 'API request failed',
              level: 'error',
              timestamp: Date.now() - 300000,
              count: 3,
            },
            {
              message: 'Component render warning',
              level: 'warn',
              timestamp: Date.now() - 600000,
              count: 1,
            },
          ],
        },
        performance: {
          averageLoadTime: performanceReport?.averages?.renderTime || 0,
          memoryUsage: performanceReport?.current?.memoryUsage?.usagePercentage || 0,
          apiResponseTime: performanceReport?.averages?.networkRequests || 0,
          cacheHitRate: performanceReport?.averages?.cacheHitRate || 0,
        },
        userActivity: {
          activeUsers: 1,
          pageViews: 15,
          interactions: 45,
          sessionDuration: 1800, // 30 minutes
        },
        system: {
          uptime: Date.now() - (Date.now() - 3600000), // 1 hour
          errorRate: 0.02, // 2%
          performanceScore: 85,
          healthStatus: 'healthy',
        },
      };

      setStats(mockStats);
    } catch (error) {
      console.error('Failed to fetch monitoring stats:', error);
      message.error('获取监控数据失败');
    } finally {
      setLoading(false);
    }
  };

  // 自动刷新
  useEffect(() => {
    fetchStats();

    if (autoRefresh) {
      const interval = setInterval(fetchStats, refreshInterval * 1000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval]);

  // 导出监控报告
  const exportReport = async () => {
    try {
      const report = {
        timestamp: new Date().toISOString(),
        stats,
        performanceReport: performanceMonitor.getPerformanceReport(),
        loggerStats: logger.getStats(),
      };

      const blob = new Blob([JSON.stringify(report, null, 2)], {
        type: 'application/json',
      });
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `monitoring-report-${dayjs().format('YYYY-MM-DD-HH-mm-ss')}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      message.success('监控报告已导出');
    } catch (error) {
      console.error('Failed to export report:', error);
      message.error('导出报告失败');
    }
  };

  // 清理日志
  const clearLogs = () => {
    logger.clear();
    message.success('日志已清理');
    fetchStats();
  };

  // 手动上报数据
  const reportData = async () => {
    try {
      await monitoring.reportAll();
      message.success('数据上报成功');
    } catch (error) {
      console.error('Failed to report data:', error);
      message.error('数据上报失败');
    }
  };

  if (loading && !stats) {
    return <Card loading />;
  }

  const getHealthStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'warning': return 'warning';
      case 'critical': return 'error';
      default: return 'default';
    }
  };

  const errorColumns = [
    {
      title: '错误信息',
      dataIndex: 'message',
      key: 'message',
    },
    {
      title: '级别',
      dataIndex: 'level',
      key: 'level',
      render: (level: string) => (
        <Tag color={level === 'error' ? 'red' : 'orange'}>{level.toUpperCase()}</Tag>
      ),
    },
    {
      title: '次数',
      dataIndex: 'count',
      key: 'count',
    },
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: number) => dayjs(timestamp).format('HH:mm:ss'),
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      {/* 控制面板 */}
      <Card style={{ marginBottom: '16px' }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Space>
              <Switch
                checked={autoRefresh}
                onChange={setAutoRefresh}
                checkedChildren="自动刷新"
                unCheckedChildren="手动刷新"
              />
              <Select
                value={refreshInterval}
                onChange={setRefreshInterval}
                disabled={!autoRefresh}
                style={{ width: 120 }}
              >
                <Select.Option value={10}>10秒</Select.Option>
                <Select.Option value={30}>30秒</Select.Option>
                <Select.Option value={60}>1分钟</Select.Option>
                <Select.Option value={300}>5分钟</Select.Option>
              </Select>
              <RangePicker
                value={dateRange}
                onChange={(dates) => dates && setDateRange(dates)}
                showTime
                format="YYYY-MM-DD HH:mm"
              />
            </Space>
          </Col>
          <Col>
            <Space>
              <Button icon={<ReloadOutlined />} onClick={fetchStats}>
                刷新
              </Button>
              <Button icon={<DownloadOutlined />} onClick={exportReport}>
                导出报告
              </Button>
              <Button onClick={reportData}>
                上报数据
              </Button>
              <Button danger onClick={clearLogs}>
                清理日志
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* 系统健康状态 */}
      {stats?.system.healthStatus !== 'healthy' && (
        <Alert
          message={`系统状态: ${stats?.system.healthStatus}`}
          description="检测到系统异常，请及时处理"
          type={stats?.system.healthStatus === 'warning' ? 'warning' : 'error'}
          showIcon
          style={{ marginBottom: '16px' }}
        />
      )}

      {/* 概览统计 */}
      <Row gutter={16} style={{ marginBottom: '16px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="系统健康度"
              value={stats?.system.performanceScore || 0}
              suffix="%"
              valueStyle={{ 
                color: (stats?.system.performanceScore || 0) > 80 ? '#3f8600' : '#cf1322' 
              }}
              prefix={<DashboardOutlined />}
            />
            <Progress
              percent={stats?.system.performanceScore || 0}
              size="small"
              status={(stats?.system.performanceScore || 0) > 80 ? 'success' : 'exception'}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="错误率"
              value={(stats?.system.errorRate || 0) * 100}
              suffix="%"
              precision={2}
              valueStyle={{ 
                color: (stats?.system.errorRate || 0) < 0.05 ? '#3f8600' : '#cf1322' 
              }}
              prefix={<BugOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="内存使用率"
              value={stats?.performance.memoryUsage || 0}
              suffix="%"
              precision={1}
              valueStyle={{ 
                color: (stats?.performance.memoryUsage || 0) < 80 ? '#3f8600' : '#cf1322' 
              }}
              prefix={<BarChartOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃用户"
              value={stats?.userActivity.activeUsers || 0}
              prefix={<UserOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* 详细监控数据 */}
      <Tabs defaultActiveKey="errors">
        <TabPane tab="错误监控" key="errors" icon={<BugOutlined />}>
          <Row gutter={16}>
            <Col span={8}>
              <Card title="错误统计" size="small">
                <Statistic
                  title="总错误数"
                  value={stats?.errors.total || 0}
                  valueStyle={{ color: '#cf1322' }}
                />
                <div style={{ marginTop: '16px' }}>
                  {Object.entries(stats?.errors.byLevel || {}).map(([level, count]) => (
                    <div key={level} style={{ marginBottom: '8px' }}>
                      <Tag color={level === 'error' ? 'red' : 'orange'}>
                        {level.toUpperCase()}: {count}
                      </Tag>
                    </div>
                  ))}
                </div>
              </Card>
            </Col>
            <Col span={16}>
              <Card title="最近错误" size="small">
                <Table
                  columns={errorColumns}
                  dataSource={stats?.errors.recent || []}
                  size="small"
                  pagination={false}
                />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="性能监控" key="performance" icon={<DashboardOutlined />}>
          <Row gutter={16}>
            <Col span={12}>
              <Card title="性能指标" size="small">
                <Row gutter={16}>
                  <Col span={12}>
                    <Statistic
                      title="平均加载时间"
                      value={stats?.performance.averageLoadTime || 0}
                      suffix="ms"
                      precision={2}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="API响应时间"
                      value={stats?.performance.apiResponseTime || 0}
                      suffix="ms"
                      precision={2}
                    />
                  </Col>
                </Row>
                <Row gutter={16} style={{ marginTop: '16px' }}>
                  <Col span={12}>
                    <Statistic
                      title="缓存命中率"
                      value={(stats?.performance.cacheHitRate || 0) * 100}
                      suffix="%"
                      precision={1}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="内存使用"
                      value={stats?.performance.memoryUsage || 0}
                      suffix="%"
                      precision={1}
                    />
                  </Col>
                </Row>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="优化建议" size="small">
                <Timeline size="small">
                  {performanceMonitor.getPerformanceReport().recommendations?.map((rec, index) => (
                    <Timeline.Item
                      key={index}
                      dot={<WarningOutlined style={{ color: '#faad14' }} />}
                    >
                      {rec}
                    </Timeline.Item>
                  )) || [
                    <Timeline.Item
                      key="no-issues"
                      dot={<CheckCircleOutlined style={{ color: '#52c41a' }} />}
                    >
                      系统运行良好，暂无优化建议
                    </Timeline.Item>
                  ]}
                </Timeline>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="用户行为" key="user-behavior" icon={<UserOutlined />}>
          <Row gutter={16}>
            <Col span={8}>
              <Card title="用户活动" size="small">
                <Statistic
                  title="页面浏览量"
                  value={stats?.userActivity.pageViews || 0}
                  style={{ marginBottom: '16px' }}
                />
                <Statistic
                  title="用户交互次数"
                  value={stats?.userActivity.interactions || 0}
                  style={{ marginBottom: '16px' }}
                />
                <Statistic
                  title="平均会话时长"
                  value={Math.round((stats?.userActivity.sessionDuration || 0) / 60)}
                  suffix="分钟"
                />
              </Card>
            </Col>
            <Col span={16}>
              <Card title="行为分析" size="small">
                <Timeline>
                  <Timeline.Item dot={<ClockCircleOutlined />}>
                    用户进入系统仪表板页面
                  </Timeline.Item>
                  <Timeline.Item dot={<ClockCircleOutlined />}>
                    查看投资组合数据
                  </Timeline.Item>
                  <Timeline.Item dot={<ClockCircleOutlined />}>
                    执行策略配置操作
                  </Timeline.Item>
                  <Timeline.Item dot={<ClockCircleOutlined />}>
                    查看实时数据监控
                  </Timeline.Item>
                </Timeline>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="日志分析" key="logs" icon={<BarChartOutlined />}>
          <Row gutter={16}>
            <Col span={8}>
              <Card title="日志配置" size="small">
                <div style={{ marginBottom: '16px' }}>
                  <label>日志级别：</label>
                  <Select
                    value={logLevel}
                    onChange={(value) => {
                      setLogLevel(value);
                      logger.setLevel(value);
                    }}
                    style={{ width: '100%', marginTop: '8px' }}
                  >
                    <Select.Option value="debug">DEBUG</Select.Option>
                    <Select.Option value="info">INFO</Select.Option>
                    <Select.Option value="warn">WARN</Select.Option>
                    <Select.Option value="error">ERROR</Select.Option>
                  </Select>
                </div>
                <div>
                  <Statistic
                    title="队列大小"
                    value={logger.getStats()?.queueSize || 0}
                    suffix={`/ ${logger.getStats()?.maxQueueSize || 0}`}
                  />
                </div>
              </Card>
            </Col>
            <Col span={16}>
              <Card title="日志统计" size="small">
                <Row gutter={16}>
                  {Object.entries(logger.getStats()?.levelCounts || {}).map(([level, count]) => (
                    <Col span={6} key={level}>
                      <Statistic
                        title={level.toUpperCase()}
                        value={count}
                        valueStyle={{
                          color: level === 'error' ? '#cf1322' : 
                                level === 'warn' ? '#faad14' : '#3f8600'
                        }}
                      />
                    </Col>
                  ))}
                </Row>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default MonitoringDashboard;