import React, { useState, useEffect } from 'react';
import {
  Card,
  Button,
  Space,
  Typography,
  Divider,
  Tag,
  Alert,
  Collapse,
  Spin,
  Badge,
  Tooltip,
  message,
  Row,
  Col,
  Statistic,
  Progress,
  Timeline,
} from 'antd';
import {
  ApiOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  BugOutlined,
  RocketOutlined,
  ThunderboltOutlined,
  HeartOutlined,
  DatabaseOutlined,
  MonitorOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons';
import { api } from '../services/api';
import { SystemHealth, SystemControlResponse } from '../services/types';

const { Title, Text } = Typography;
const { Panel } = Collapse;

interface TestResult {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'success' | 'error';
  message?: string;
  details?: any;
  duration?: number;
  timestamp: string;
}

interface ApiEndpoint {
  id: string;
  name: string;
  description: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  endpoint: string;
  category: 'system' | 'data' | 'strategy' | 'order' | 'portfolio' | 'risk';
  testFunction: () => Promise<any>;
}

const ApiTestingPanel: React.FC = () => {
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [isSystemRunning, setIsSystemRunning] = useState(false);
  const [globalLoading, setGlobalLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  // Define API endpoints for testing
  const apiEndpoints: ApiEndpoint[] = [
    {
      id: 'health-check',
      name: '健康检查',
      description: '检查系统整体健康状态',
      method: 'GET',
      endpoint: '/health',
      category: 'system',
      testFunction: api.system.getHealth,
    },
    {
      id: 'system-metrics',
      name: '系统指标',
      description: '获取系统运行指标',
      method: 'GET',
      endpoint: '/metrics',
      category: 'system',
      testFunction: api.system.getMetrics,
    },
    {
      id: 'system-start',
      name: '启动系统',
      description: '启动交易系统',
      method: 'POST',
      endpoint: '/system/start',
      category: 'system',
      testFunction: api.system.startSystem,
    },
    {
      id: 'system-stop',
      name: '停止系统',
      description: '停止交易系统',
      method: 'POST',
      endpoint: '/system/stop',
      category: 'system',
      testFunction: api.system.stopSystem,
    },
    {
      id: 'portfolio-summary',
      name: '投资组合摘要',
      description: '获取投资组合概览',
      method: 'GET',
      endpoint: '/portfolio/summary',
      category: 'portfolio',
      testFunction: api.portfolio.getSummary,
    },
    {
      id: 'risk-metrics',
      name: '风险指标',
      description: '获取风险管理指标',
      method: 'GET',
      endpoint: '/risk/metrics',
      category: 'risk',
      testFunction: api.risk.getMetrics,
    },
  ];

  // Fetch system health
  const fetchSystemHealth = async () => {
    try {
      const health = await api.system.getHealth();
      setSystemHealth(health);
      setIsSystemRunning(health.systemRunning);
      setLastRefresh(new Date());
    } catch (error) {
      console.error('Failed to fetch system health:', error);
      setSystemHealth(null);
      setIsSystemRunning(false);
    }
  };

  // Auto-refresh system health
  useEffect(() => {
    fetchSystemHealth();

    if (autoRefresh) {
      const interval = setInterval(fetchSystemHealth, 5000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  // Execute API test
  const executeTest = async (endpoint: ApiEndpoint) => {
    const testId = `${endpoint.id}-${Date.now()}`;
    const startTime = Date.now();

    // Create initial test result
    const initialResult: TestResult = {
      id: testId,
      name: endpoint.name,
      status: 'running',
      timestamp: new Date().toISOString(),
    };

    setTestResults(prev => [initialResult, ...prev.slice(0, 19)]); // Keep only last 20 results

    try {
      const result = await endpoint.testFunction();
      const duration = Date.now() - startTime;

      const successResult: TestResult = {
        ...initialResult,
        status: 'success',
        message: '测试成功',
        details: result,
        duration,
      };

      setTestResults(prev => prev.map(r => r.id === testId ? successResult : r));
      message.success(`${endpoint.name} 测试成功`);

    } catch (error: any) {
      const duration = Date.now() - startTime;
      const errorMessage = error?.message || error?.detail || error?.toString() || '未知错误';

      const errorResult: TestResult = {
        ...initialResult,
        status: 'error',
        message: errorMessage,
        details: error,
        duration,
      };

      setTestResults(prev => prev.map(r => r.id === testId ? errorResult : r));
      message.error(`${endpoint.name} 测试失败: ${errorMessage}`);
    }
  };

  // Execute all tests in sequence
  const executeAllTests = async () => {
    setGlobalLoading(true);
    
    const filteredEndpoints = selectedCategory === 'all' 
      ? apiEndpoints 
      : apiEndpoints.filter(ep => ep.category === selectedCategory);

    for (const endpoint of filteredEndpoints) {
      await executeTest(endpoint);
      // Small delay between tests to avoid overwhelming the server
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    setGlobalLoading(false);
  };

  // Clear test results
  const clearResults = () => {
    setTestResults([]);
    message.info('测试结果已清空');
  };

  // Quick health test (only system endpoints)
  const executeQuickTest = async () => {
    setGlobalLoading(true);
    
    const systemEndpoints = apiEndpoints.filter(ep => ep.category === 'system');
    
    for (const endpoint of systemEndpoints) {
      await executeTest(endpoint);
      await new Promise(resolve => setTimeout(resolve, 300));
    }
    
    setGlobalLoading(false);
    message.info('快速健康检查完成');
  };

  // Get category badge color
  const getCategoryColor = (category: string) => {
    const colors = {
      system: 'blue',
      data: 'green',
      strategy: 'purple',
      order: 'orange',
      portfolio: 'cyan',
      risk: 'red',
    };
    return colors[category as keyof typeof colors] || 'default';
  };

  // Get status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'error':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'running':
        return <Spin size="small" />;
      default:
        return <ClockCircleOutlined style={{ color: '#d9d9d9' }} />;
    }
  };

  // Calculate success rate
  const successRate = testResults.length > 0 
    ? (testResults.filter(r => r.status === 'success').length / testResults.length) * 100 
    : 0;

  // Get running tests count
  const runningTests = testResults.filter(r => r.status === 'running').length;

  // Get error tests count
  const errorTests = testResults.filter(r => r.status === 'error').length;

  // Get success tests count
  const successTests = testResults.filter(r => r.status === 'success').length;

  // Calculate average response time
  const avgResponseTime = testResults.length > 0
    ? testResults
        .filter(r => r.duration)
        .reduce((sum, r) => sum + (r.duration || 0), 0) / 
      testResults.filter(r => r.duration).length
    : 0;

  return (
    <Card
      title={
        <Space>
          <ApiOutlined style={{ color: '#1890ff' }} />
          <Title level={4} style={{ margin: 0 }}>
            API 测试中心
          </Title>
          <Tag color="blue">生产级测试工具</Tag>
        </Space>
      }
      extra={
        <Space>
          <Tooltip title="手动刷新系统状态">
            <Button
              size="small"
              icon={<ReloadOutlined />}
              onClick={fetchSystemHealth}
            >
              刷新
            </Button>
          </Tooltip>
          <Tooltip title={autoRefresh ? '关闭自动刷新' : '开启自动刷新'}>
            <Button
              size="small"
              type={autoRefresh ? 'primary' : 'default'}
              icon={<HeartOutlined />}
              onClick={() => setAutoRefresh(!autoRefresh)}
            >
              {autoRefresh ? '自动' : '手动'}
            </Button>
          </Tooltip>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            更新: {lastRefresh.toLocaleTimeString()}
          </Text>
        </Space>
      }
      bodyStyle={{ padding: '16px' }}
    >
      {/* System Status Overview */}
      <Alert
        message={
          <Space>
            <Text strong>系统状态</Text>
            <Badge 
              status={isSystemRunning ? 'success' : 'error'} 
              text={isSystemRunning ? '运行中' : '已停止'} 
            />
          </Space>
        }
        description={
          systemHealth ? (
            <Row gutter={16}>
              <Col span={6}>
                <Statistic
                  title="运行时间"
                  value={Math.floor(systemHealth.uptimeSeconds / 3600)}
                  suffix="小时"
                  prefix={<ClockCircleOutlined />}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="模块数量"
                  value={Object.keys(systemHealth.modules).length}
                  prefix={<DatabaseOutlined />}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="测试成功率"
                  value={successRate}
                  precision={1}
                  suffix="%"
                  prefix={<CheckCircleOutlined />}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="平均响应时间"
                  value={avgResponseTime}
                  precision={0}
                  suffix="ms"
                  prefix={<ThunderboltOutlined />}
                />
              </Col>
            </Row>
          ) : (
            <Text type="secondary">正在获取系统状态...</Text>
          )
        }
        type={isSystemRunning ? 'success' : 'warning'}
        showIcon
        style={{ marginBottom: '16px' }}
      />

      <Divider />

      {/* Test Controls */}
      <Row gutter={[16, 16]} style={{ marginBottom: '16px' }}>
        <Col span={12}>
          <Space wrap>
            <Button
              type="primary"
              icon={<RocketOutlined />}
              loading={globalLoading}
              onClick={executeAllTests}
              size="large"
            >
              执行全部测试
            </Button>
            <Button
              icon={<ThunderboltOutlined />}
              loading={globalLoading}
              onClick={executeQuickTest}
            >
              快速检查
            </Button>
            <Button
              icon={<BugOutlined />}
              onClick={clearResults}
              disabled={testResults.length === 0}
            >
              清空结果
            </Button>
          </Space>
        </Col>
        <Col span={12} style={{ textAlign: 'right' }}>
          <Space>
            <Text>过滤类别:</Text>
            <Button.Group>
              <Button
                size="small"
                type={selectedCategory === 'all' ? 'primary' : 'default'}
                onClick={() => setSelectedCategory('all')}
              >
                全部
              </Button>
              <Button
                size="small"
                type={selectedCategory === 'system' ? 'primary' : 'default'}
                onClick={() => setSelectedCategory('system')}
              >
                系统
              </Button>
              <Button
                size="small"
                type={selectedCategory === 'portfolio' ? 'primary' : 'default'}
                onClick={() => setSelectedCategory('portfolio')}
              >
                投资组合
              </Button>
              <Button
                size="small"
                type={selectedCategory === 'risk' ? 'primary' : 'default'}
                onClick={() => setSelectedCategory('risk')}
              >
                风险
              </Button>
            </Button.Group>
          </Space>
        </Col>
      </Row>

      {/* API Endpoints Grid */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        {apiEndpoints
          .filter(ep => selectedCategory === 'all' || ep.category === selectedCategory)
          .map(endpoint => (
            <Col span={8} key={endpoint.id}>
              <Card
                size="small"
                hoverable
                bodyStyle={{ padding: '12px' }}
                style={{
                  transition: 'all 0.3s ease',
                  borderColor: testResults.some(r => 
                    r.name === endpoint.name && r.status === 'success'
                  ) ? '#52c41a' : testResults.some(r => 
                    r.name === endpoint.name && r.status === 'error'
                  ) ? '#ff4d4f' : '#d9d9d9'
                }}
                actions={[
                  <Tooltip title="执行测试">
                    <Button
                      type="text"
                      icon={<PlayCircleOutlined />}
                      onClick={() => executeTest(endpoint)}
                      loading={testResults.some(r => 
                        r.name === endpoint.name && r.status === 'running'
                      )}
                    >
                      测试
                    </Button>
                  </Tooltip>
                ]}
              >
                <Space direction="vertical" size="small" style={{ width: '100%' }}>
                  <Space>
                    <Tag color={getCategoryColor(endpoint.category)}>
                      {endpoint.method}
                    </Tag>
                    <Text strong style={{ fontSize: '14px' }}>
                      {endpoint.name}
                    </Text>
                  </Space>
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    {endpoint.description}
                  </Text>
                  <Text code style={{ fontSize: '11px' }}>
                    {endpoint.endpoint}
                  </Text>
                </Space>
              </Card>
            </Col>
          ))}
      </Row>

      <Divider />

      {/* Test Results */}
      <div>
        <Title level={5} style={{ marginBottom: '16px' }}>
          <Space>
            <MonitorOutlined />
            测试结果
            {testResults.length > 0 && (
              <Badge count={testResults.length} showZero color="#52c41a" />
            )}
          </Space>
        </Title>

        {testResults.length === 0 ? (
          <Alert
            message="暂无测试结果"
            description="点击上方的测试按钮开始执行 API 测试"
            type="info"
            showIcon
            style={{ textAlign: 'center' }}
          />
        ) : (
          <Timeline
            mode="left"
            style={{ maxHeight: '400px', overflowY: 'auto' }}
          >
            {testResults.map(result => (
              <Timeline.Item
                key={result.id}
                dot={getStatusIcon(result.status)}
                color={
                  result.status === 'success' ? 'green' :
                  result.status === 'error' ? 'red' : 'blue'
                }
              >
                <Card size="small" style={{ marginBottom: '8px' }}>
                  <Space direction="vertical" size="small" style={{ width: '100%' }}>
                    <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                      <Text strong>{result.name}</Text>
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {new Date(result.timestamp).toLocaleTimeString()}
                        {result.duration && ` (${result.duration}ms)`}
                      </Text>
                    </Space>
                    
                    {result.message && (
                      <Text 
                        type={result.status === 'error' ? 'danger' : 'success'}
                        style={{ fontSize: '13px' }}
                      >
                        {result.message}
                      </Text>
                    )}
                    
                    {result.details && (
                      <Collapse ghost size="small">
                        <Panel 
                          header="详细信息" 
                          key="details"
                          style={{ fontSize: '12px' }}
                        >
                          <pre style={{ 
                            fontSize: '11px', 
                            maxHeight: '200px', 
                            overflow: 'auto',
                            background: '#fafafa',
                            padding: '8px',
                            borderRadius: '4px'
                          }}>
                            {JSON.stringify(result.details, null, 2)}
                          </pre>
                        </Panel>
                      </Collapse>
                    )}
                  </Space>
                </Card>
              </Timeline.Item>
            ))}
          </Timeline>
        )}
      </div>

      {testResults.length > 0 && (
        <div style={{ marginTop: '16px', textAlign: 'center' }}>
          <Progress
            percent={successRate}
            status={successRate === 100 ? 'success' : successRate > 70 ? 'active' : 'exception'}
            format={percent => `成功率 ${percent?.toFixed(1)}%`}
          />
        </div>
      )}
    </Card>
  );
};

export default ApiTestingPanel;