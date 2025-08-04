import React, { useState, useEffect } from 'react';
import {
  Card,
  Statistic,
  Progress,
  Tag,
  Space,
  Row,
  Col,
  Alert,
  Typography,
  Badge,
  Tooltip,
  Button,
} from 'antd';
import {
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined,
  ReloadOutlined,
  WarningOutlined,
} from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import { api } from '../../../services/api';
import { formatNumber, formatTime } from '../../../utils/format';

const { Text, Title } = Typography;

interface DataProcessingMetrics {
  ticksProcessed: number;
  ticksPerSecond: number;
  processingLatency: number;
  errorRate: number;
  queueSize: number;
  lastProcessedTime: string;
  dataQuality: {
    completeness: number;
    accuracy: number;
    timeliness: number;
  };
  marketStatus: {
    isOpen: boolean;
    nextOpenTime?: string;
    nextCloseTime?: string;
    tradingSession: string;
  };
}

const DataProcessingStatus: React.FC = () => {
  const [alerts, setAlerts] = useState<Array<{
    type: 'warning' | 'error' | 'info';
    message: string;
    timestamp: string;
  }>>([]);

  // Fetch data processing status
  const { data: dataStatus, isLoading, error, refetch } = useQuery({
    queryKey: ['dataProcessingStatus'],
    queryFn: async () => {
      const [systemMetrics, dataStatusResponse] = await Promise.all([
        api.system.getMetrics(),
        api.data.getDataStatus(),
      ]);

      // Mock data processing metrics (in real implementation, this would come from the API)
      const mockMetrics: DataProcessingMetrics = {
        ticksProcessed: Math.floor(Math.random() * 100000) + 50000,
        ticksPerSecond: Math.floor(Math.random() * 1000) + 500,
        processingLatency: Math.random() * 10 + 1,
        errorRate: Math.random() * 0.05,
        queueSize: Math.floor(Math.random() * 100),
        lastProcessedTime: new Date().toISOString(),
        dataQuality: {
          completeness: 95 + Math.random() * 5,
          accuracy: 98 + Math.random() * 2,
          timeliness: 90 + Math.random() * 10,
        },
        marketStatus: {
          isOpen: isMarketOpen(),
          nextOpenTime: getNextMarketTime('open'),
          nextCloseTime: getNextMarketTime('close'),
          tradingSession: getCurrentTradingSession(),
        },
      };

      return mockMetrics;
    },
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  // Generate alerts based on metrics
  useEffect(() => {
    if (!dataStatus) return;

    const newAlerts: typeof alerts = [];

    // Check error rate
    if (dataStatus.errorRate > 0.02) {
      newAlerts.push({
        type: 'error',
        message: `数据处理错误率过高: ${(dataStatus.errorRate * 100).toFixed(2)}%`,
        timestamp: new Date().toISOString(),
      });
    }

    // Check processing latency
    if (dataStatus.processingLatency > 5) {
      newAlerts.push({
        type: 'warning',
        message: `数据处理延迟过高: ${dataStatus.processingLatency.toFixed(2)}ms`,
        timestamp: new Date().toISOString(),
      });
    }

    // Check queue size
    if (dataStatus.queueSize > 50) {
      newAlerts.push({
        type: 'warning',
        message: `数据队列积压: ${dataStatus.queueSize} 条待处理`,
        timestamp: new Date().toISOString(),
      });
    }

    // Check data quality
    if (dataStatus.dataQuality.completeness < 95) {
      newAlerts.push({
        type: 'warning',
        message: `数据完整性低于标准: ${dataStatus.dataQuality.completeness.toFixed(1)}%`,
        timestamp: new Date().toISOString(),
      });
    }

    setAlerts(newAlerts);
  }, [dataStatus]);

  // Helper functions
  function isMarketOpen(): boolean {
    const now = new Date();
    const hour = now.getHours();
    const minute = now.getMinutes();
    const currentTime = hour * 60 + minute;
    
    // A股交易时间: 9:30-11:30, 13:00-15:00
    const morningStart = 9 * 60 + 30; // 9:30
    const morningEnd = 11 * 60 + 30;   // 11:30
    const afternoonStart = 13 * 60;     // 13:00
    const afternoonEnd = 15 * 60;       // 15:00
    
    return (currentTime >= morningStart && currentTime <= morningEnd) ||
           (currentTime >= afternoonStart && currentTime <= afternoonEnd);
  }

  function getCurrentTradingSession(): string {
    const now = new Date();
    const hour = now.getHours();
    const minute = now.getMinutes();
    const currentTime = hour * 60 + minute;
    
    const morningStart = 9 * 60 + 30;
    const morningEnd = 11 * 60 + 30;
    const afternoonStart = 13 * 60;
    const afternoonEnd = 15 * 60;
    
    if (currentTime >= morningStart && currentTime <= morningEnd) {
      return '上午交易';
    } else if (currentTime >= afternoonStart && currentTime <= afternoonEnd) {
      return '下午交易';
    } else if (currentTime < morningStart) {
      return '开盘前';
    } else if (currentTime > morningEnd && currentTime < afternoonStart) {
      return '午间休市';
    } else {
      return '收盘后';
    }
  }

  function getNextMarketTime(type: 'open' | 'close'): string {
    const now = new Date();
    const tomorrow = new Date(now);
    tomorrow.setDate(tomorrow.getDate() + 1);
    
    if (type === 'open') {
      const nextOpen = new Date(now);
      nextOpen.setHours(9, 30, 0, 0);
      
      if (nextOpen <= now) {
        nextOpen.setDate(nextOpen.getDate() + 1);
      }
      
      return nextOpen.toISOString();
    } else {
      const nextClose = new Date(now);
      nextClose.setHours(15, 0, 0, 0);
      
      if (nextClose <= now) {
        nextClose.setDate(nextClose.getDate() + 1);
      }
      
      return nextClose.toISOString();
    }
  }

  // Render data quality progress
  const renderDataQuality = (label: string, value: number, color: string) => (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <Text type="secondary">{label}</Text>
        <Text strong>{value.toFixed(1)}%</Text>
      </div>
      <Progress
        percent={value}
        strokeColor={color}
        size="small"
        showInfo={false}
      />
    </div>
  );

  if (error) {
    return (
      <Card title="数据处理状态" size="small">
        <Alert
          message="获取数据处理状态失败"
          description={error.message}
          type="error"
          showIcon
          action={
            <Button size="small" onClick={() => refetch()}>
              重试
            </Button>
          }
        />
      </Card>
    );
  }

  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      {/* Market Status Card */}
      <Card
        title="市场状态"
        size="small"
        extra={
          <Badge
            status={dataStatus?.marketStatus.isOpen ? 'success' : 'default'}
            text={dataStatus?.marketStatus.isOpen ? '开市' : '闭市'}
          />
        }
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <Row gutter={16}>
            <Col span={24}>
              <Statistic
                title="当前交易时段"
                value={dataStatus?.marketStatus.tradingSession || '未知'}
                prefix={
                  dataStatus?.marketStatus.isOpen ? 
                    <CheckCircleOutlined style={{ color: '#52c41a' }} /> :
                    <ClockCircleOutlined style={{ color: '#666' }} />
                }
              />
            </Col>
          </Row>
          
          {dataStatus?.marketStatus.nextOpenTime && (
            <Text type="secondary">
              下次开市: {formatTime(dataStatus.marketStatus.nextOpenTime)}
            </Text>
          )}
          
          {dataStatus?.marketStatus.nextCloseTime && (
            <Text type="secondary">
              下次收市: {formatTime(dataStatus.marketStatus.nextCloseTime)}
            </Text>
          )}
        </Space>
      </Card>

      {/* Data Processing Metrics Card */}
      <Card
        title="数据处理统计"
        size="small"
        extra={
          <Button
            icon={<ReloadOutlined />}
            size="small"
            onClick={() => refetch()}
            loading={isLoading}
          />
        }
      >
        <Row gutter={[16, 16]}>
          <Col span={12}>
            <Statistic
              title="已处理Tick数"
              value={dataStatus?.ticksProcessed || 0}
              formatter={(value) => formatNumber(value as number)}
            />
          </Col>
          <Col span={12}>
            <Statistic
              title="处理速度"
              value={dataStatus?.ticksPerSecond || 0}
              suffix="条/秒"
              formatter={(value) => formatNumber(value as number)}
            />
          </Col>
          <Col span={12}>
            <Statistic
              title="处理延迟"
              value={dataStatus?.processingLatency || 0}
              suffix="ms"
              precision={2}
              valueStyle={{
                color: (dataStatus?.processingLatency || 0) > 5 ? '#ff4d4f' : '#52c41a'
              }}
            />
          </Col>
          <Col span={12}>
            <Statistic
              title="错误率"
              value={(dataStatus?.errorRate || 0) * 100}
              suffix="%"
              precision={3}
              valueStyle={{
                color: (dataStatus?.errorRate || 0) > 0.02 ? '#ff4d4f' : '#52c41a'
              }}
            />
          </Col>
        </Row>

        <div style={{ marginTop: 16 }}>
          <Text type="secondary">
            队列积压: {dataStatus?.queueSize || 0} 条
          </Text>
          <br />
          <Text type="secondary">
            最后处理时间: {dataStatus?.lastProcessedTime ? formatTime(dataStatus.lastProcessedTime) : '未知'}
          </Text>
        </div>
      </Card>

      {/* Data Quality Card */}
      <Card title="数据质量监控" size="small">
        {dataStatus?.dataQuality && (
          <Space direction="vertical" style={{ width: '100%' }}>
            {renderDataQuality('完整性', dataStatus.dataQuality.completeness, '#52c41a')}
            {renderDataQuality('准确性', dataStatus.dataQuality.accuracy, '#1890ff')}
            {renderDataQuality('及时性', dataStatus.dataQuality.timeliness, '#faad14')}
          </Space>
        )}
      </Card>

      {/* Alerts Card */}
      {alerts.length > 0 && (
        <Card
          title={
            <Space>
              <WarningOutlined style={{ color: '#faad14' }} />
              <Text>数据告警</Text>
            </Space>
          }
          size="small"
        >
          <Space direction="vertical" style={{ width: '100%' }}>
            {alerts.map((alert, index) => (
              <Alert
                key={index}
                message={alert.message}
                type={alert.type}
                size="small"
                showIcon
                closable
                onClose={() => {
                  setAlerts(prev => prev.filter((_, i) => i !== index));
                }}
              />
            ))}
          </Space>
        </Card>
      )}
    </Space>
  );
};

export { DataProcessingStatus };