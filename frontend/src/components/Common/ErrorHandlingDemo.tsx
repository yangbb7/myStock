import React, { useState } from 'react';
import { 
  Card, 
  Button, 
  Space, 
  Divider, 
  Typography, 
  Row, 
  Col,
  Alert,
  Statistic,
  Tag
} from 'antd';
import {
  BugOutlined,
  ReloadOutlined,
  WifiOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined
} from '@ant-design/icons';

import { 
  ErrorBoundary,
  LoadingState,
  DataFallback,
  NetworkStatusIndicator,
  DataSyncStatus,
  ProgressIndicator,
  useProgressSteps
} from './index';

import { 
  useErrorHandling,
  useApiErrorHandling,
  useAsyncErrorHandling,
  useValidationErrorHandling
} from '../../hooks/useErrorHandling';

import { useNetworkStatus, useDataCache } from './DataFallback';
import { errorReporting } from '../../services/errorReporting';

const { Title, Text, Paragraph } = Typography;

// Demo component for testing error handling
const ErrorHandlingDemo: React.FC = () => {
  const [demoState, setDemoState] = useState({
    loading: false,
    error: null as Error | null,
    data: null as any,
    showOffline: false,
  });

  // Error handling hooks
  const { handleError, hasError, error, clearError, retry } = useErrorHandling({
    context: 'demo',
    component: 'ErrorHandlingDemo',
    onRetry: () => {
      console.log('Retrying operation...');
      setDemoState(prev => ({ ...prev, error: null }));
    },
  });

  const { handleError: handleApiError } = useApiErrorHandling('demo-api');
  
  const { 
    validationErrors, 
    handleValidationError, 
    clearValidationErrors 
  } = useValidationErrorHandling();

  const { 
    loading: asyncLoading, 
    execute: executeAsync 
  } = useAsyncErrorHandling(
    async () => {
      // Simulate async operation
      await new Promise(resolve => setTimeout(resolve, 2000));
      if (Math.random() > 0.5) {
        throw new Error('Random async error');
      }
      return { message: 'Success!' };
    },
    {
      context: 'async-demo',
      onSuccess: (data) => {
        console.log('Async operation succeeded:', data);
      },
    }
  );

  // Network status
  const { isOnline } = useNetworkStatus();
  
  // Data cache
  const { setCache, getCache, hasCache } = useDataCache('demo-data');

  // Progress steps
  const { steps, currentStepId, startStep, completeStep, errorStep } = useProgressSteps([
    { id: 'step1', title: '初始化', description: '准备数据和环境' },
    { id: 'step2', title: '数据处理', description: '处理业务逻辑' },
    { id: 'step3', title: '保存结果', description: '保存处理结果' },
  ]);

  // Demo functions
  const triggerJavaScriptError = () => {
    try {
      // @ts-ignore
      nonExistentFunction();
    } catch (error) {
      handleError(error as Error, { context: 'javascript-error' });
    }
  };

  const triggerApiError = () => {
    const apiError = {
      name: 'ApiError',
      message: 'API request failed',
      code: 'HTTP_500',
      details: { endpoint: '/api/demo' },
      timestamp: new Date().toISOString(),
    };
    handleApiError(apiError as any);
  };

  const triggerNetworkError = () => {
    const networkError = new Error('Network connection failed');
    networkError.name = 'NetworkError';
    handleError(networkError, { context: 'network-error' });
  };

  const triggerValidationError = () => {
    handleValidationError({
      username: ['用户名不能为空', '用户名长度至少3个字符'],
      email: ['邮箱格式不正确'],
    });
  };

  const simulateProgressOperation = async () => {
    startStep('step1');
    await new Promise(resolve => setTimeout(resolve, 1000));
    completeStep('step1');

    startStep('step2');
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    if (Math.random() > 0.7) {
      errorStep('step2', '数据处理失败');
      return;
    }
    
    completeStep('step2');

    startStep('step3');
    await new Promise(resolve => setTimeout(resolve, 800));
    completeStep('step3');
  };

  const toggleOfflineMode = () => {
    setDemoState(prev => ({ ...prev, showOffline: !prev.showOffline }));
  };

  // Get error statistics
  const errorStats = errorReporting.getErrorStats();

  return (
    <div style={{ padding: 24 }}>
      <Title level={2}>错误处理系统演示</Title>
      <Paragraph>
        这个演示页面展示了前端错误处理系统的各种功能，包括错误捕获、用户通知、离线支持、进度指示等。
      </Paragraph>

      {/* Network Status */}
      <NetworkStatusIndicator showDetails position="topRight" />

      <Row gutter={[16, 16]}>
        {/* Error Triggers */}
        <Col span={12}>
          <Card title="错误触发测试" extra={<BugOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button onClick={triggerJavaScriptError} danger>
                触发 JavaScript 错误
              </Button>
              <Button onClick={triggerApiError} danger>
                触发 API 错误
              </Button>
              <Button onClick={triggerNetworkError} danger>
                触发网络错误
              </Button>
              <Button onClick={triggerValidationError} danger>
                触发验证错误
              </Button>
              <Button onClick={executeAsync} loading={asyncLoading}>
                异步操作测试
              </Button>
            </Space>
          </Card>
        </Col>

        {/* Error Status */}
        <Col span={12}>
          <Card title="错误状态" extra={hasError ? <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} /> : <CheckCircleOutlined style={{ color: '#52c41a' }} />}>
            {hasError ? (
              <Alert
                message="检测到错误"
                description={error?.message}
                type="error"
                action={
                  <Button size="small" onClick={clearError}>
                    清除
                  </Button>
                }
              />
            ) : (
              <Alert
                message="系统正常"
                description="没有检测到错误"
                type="success"
              />
            )}

            {Object.keys(validationErrors).length > 0 && (
              <Alert
                style={{ marginTop: 8 }}
                message="验证错误"
                description={
                  <ul>
                    {Object.entries(validationErrors).map(([field, errors]) => (
                      <li key={field}>
                        <strong>{field}:</strong> {errors.join(', ')}
                      </li>
                    ))}
                  </ul>
                }
                type="warning"
                action={
                  <Button size="small" onClick={clearValidationErrors}>
                    清除
                  </Button>
                }
              />
            )}
          </Card>
        </Col>

        {/* Error Statistics */}
        <Col span={12}>
          <Card title="错误统计">
            <Row gutter={16}>
              <Col span={8}>
                <Statistic title="总错误数" value={errorStats.total} />
              </Col>
              <Col span={8}>
                <Statistic 
                  title="严重错误" 
                  value={errorStats.bySeverity.critical || 0}
                  valueStyle={{ color: '#ff4d4f' }}
                />
              </Col>
              <Col span={8}>
                <Statistic 
                  title="API错误" 
                  value={errorStats.byCategory.api || 0}
                  valueStyle={{ color: '#faad14' }}
                />
              </Col>
            </Row>
            
            <Divider />
            
            <div>
              <Text strong>最近错误:</Text>
              <div style={{ marginTop: 8 }}>
                {errorStats.recent.length > 0 ? (
                  errorStats.recent.slice(-3).map((report, index) => (
                    <Tag key={index} color="red" style={{ marginBottom: 4 }}>
                      {report.error.name}: {report.error.message.substring(0, 30)}...
                    </Tag>
                  ))
                ) : (
                  <Text type="secondary">暂无错误记录</Text>
                )}
              </div>
            </div>
          </Card>
        </Col>

        {/* Network & Offline Demo */}
        <Col span={12}>
          <Card title="网络状态与离线支持">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text>网络状态:</Text>
                <Tag color={isOnline ? 'green' : 'red'} icon={<WifiOutlined />}>
                  {isOnline ? '在线' : '离线'}
                </Tag>
              </div>
              
              <DataSyncStatus
                lastSync={new Date()}
                syncInProgress={false}
                onSync={() => console.log('Syncing...')}
              />
              
              <Button onClick={toggleOfflineMode}>
                {demoState.showOffline ? '显示在线模式' : '模拟离线模式'}
              </Button>
            </Space>
          </Card>
        </Col>

        {/* Progress Demo */}
        <Col span={24}>
          <Card title="进度指示器演示">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button onClick={simulateProgressOperation} icon={<ReloadOutlined />}>
                开始进度演示
              </Button>
              
              <ProgressIndicator
                steps={steps}
                currentStep={currentStepId}
                title="操作进度演示"
                showOverallProgress
                showStepDetails
                onRetry={(stepId) => {
                  console.log('Retrying step:', stepId);
                  startStep(stepId);
                  setTimeout(() => completeStep(stepId), 1000);
                }}
              />
            </Space>
          </Card>
        </Col>

        {/* Data Fallback Demo */}
        <Col span={24}>
          <Card title="数据降级演示">
            <DataFallback
              error={demoState.showOffline ? new Error('模拟离线错误') : null}
              loading={false}
              data={demoState.showOffline ? null : { message: '这是正常数据' }}
              cacheKey="demo-fallback"
              onRetry={() => {
                console.log('Retrying data fetch...');
                setDemoState(prev => ({ ...prev, showOffline: false }));
              }}
              allowStaleData
              showCacheInfo
            >
              <div>
                <Alert
                  message="数据加载成功"
                  description="这里显示的是正常的数据内容"
                  type="success"
                />
              </div>
            </DataFallback>
          </Card>
        </Col>

        {/* Loading States Demo */}
        <Col span={24}>
          <Card title="加载状态演示">
            <Row gutter={16}>
              <Col span={8}>
                <LoadingState loading={true} skeleton>
                  <div>内容区域</div>
                </LoadingState>
              </Col>
              <Col span={8}>
                <LoadingState 
                  loading={true} 
                  showProgress 
                  progress={65}
                  progressTitle="数据处理中..."
                  progressDescription="正在处理您的请求，请稍候"
                >
                  <div>内容区域</div>
                </LoadingState>
              </Col>
              <Col span={8}>
                <LoadingState 
                  error={new Error('加载失败示例')}
                  onRetry={() => console.log('重试加载')}
                >
                  <div>内容区域</div>
                </LoadingState>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

// Wrap with error boundary
const ErrorHandlingDemoWithBoundary: React.FC = () => (
  <ErrorBoundary
    onError={(error, errorInfo) => {
      console.error('Error boundary caught error:', error, errorInfo);
      errorReporting.reportError(error, {
        component: 'ErrorHandlingDemo',
        action: 'error_boundary',
        additionalData: errorInfo,
      });
    }}
  >
    <ErrorHandlingDemo />
  </ErrorBoundary>
);

export default ErrorHandlingDemoWithBoundary;