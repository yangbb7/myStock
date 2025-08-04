import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Progress, Button, Space, Alert } from 'antd';
import { ReloadOutlined, WarningOutlined } from '@ant-design/icons';
import { performanceMonitor } from '../../utils/performanceMonitor';
import { apiCache, chartDataCache, getMemoryInfo, formatBytes } from '../../utils/cacheManager';

interface PerformanceStats {
  renderTime: number;
  memoryUsage?: {
    used: number;
    total: number;
    limit: number;
    usagePercentage: number;
  };
  cacheStats: {
    apiCache: any;
    chartCache: any;
  };
  networkRequests: number;
  cacheHitRate: number;
  recommendations: string[];
}

export const PerformanceDashboard: React.FC<{ 
  visible?: boolean;
  onClose?: () => void;
}> = ({ visible = false, onClose }) => {
  const [stats, setStats] = useState<PerformanceStats | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const refreshStats = async () => {
    setRefreshing(true);
    try {
      const report = performanceMonitor.getPerformanceReport();
      const memoryInfo = getMemoryInfo();
      
      const newStats: PerformanceStats = {
        renderTime: report.current.renderTime,
        memoryUsage: memoryInfo || undefined,
        cacheStats: {
          apiCache: apiCache.getStats(),
          chartCache: chartDataCache.getStats()
        },
        networkRequests: report.current.networkRequests,
        cacheHitRate: report.current.cacheHitRate,
        recommendations: report.recommendations
      };

      setStats(newStats);
    } catch (error) {
      console.error('Failed to refresh performance stats:', error);
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    if (visible) {
      refreshStats();
      const interval = setInterval(refreshStats, 5000); // Refresh every 5 seconds
      return () => clearInterval(interval);
    }
  }, [visible]);

  if (!visible || !stats) {
    return null;
  }

  const getPerformanceColor = (value: number, thresholds: { good: number; warning: number }) => {
    if (value <= thresholds.good) return '#52c41a';
    if (value <= thresholds.warning) return '#faad14';
    return '#ff4d4f';
  };

  const getMemoryColor = (percentage: number) => {
    if (percentage <= 60) return '#52c41a';
    if (percentage <= 80) return '#faad14';
    return '#ff4d4f';
  };

  return (
    <div style={{ 
      position: 'fixed',
      top: 20,
      right: 20,
      width: 400,
      maxHeight: '80vh',
      overflowY: 'auto',
      zIndex: 1000,
      background: 'white',
      borderRadius: 8,
      boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
      border: '1px solid #d9d9d9'
    }}>
      <Card
        title="性能监控面板"
        size="small"
        extra={
          <Space>
            <Button
              icon={<ReloadOutlined />}
              size="small"
              loading={refreshing}
              onClick={refreshStats}
            />
            <Button size="small" onClick={onClose}>
              关闭
            </Button>
          </Space>
        }
      >
        {/* Render Performance */}
        <Row gutter={[16, 16]}>
          <Col span={12}>
            <Statistic
              title="渲染时间"
              value={stats.renderTime}
              precision={2}
              suffix="ms"
              valueStyle={{ 
                color: getPerformanceColor(stats.renderTime, { good: 16, warning: 50 })
              }}
            />
          </Col>
          <Col span={12}>
            <Statistic
              title="网络请求"
              value={stats.networkRequests}
              valueStyle={{ 
                color: getPerformanceColor(stats.networkRequests, { good: 5, warning: 20 })
              }}
            />
          </Col>
        </Row>

        {/* Memory Usage */}
        {stats.memoryUsage && (
          <Card size="small" title="内存使用" style={{ marginTop: 16 }}>
            <Row gutter={[16, 8]}>
              <Col span={24}>
                <div style={{ marginBottom: 8 }}>
                  <span>使用量: {formatBytes(stats.memoryUsage.used)}</span>
                  <span style={{ float: 'right' }}>
                    {stats.memoryUsage.usagePercentage.toFixed(1)}%
                  </span>
                </div>
                <Progress
                  percent={stats.memoryUsage.usagePercentage}
                  strokeColor={getMemoryColor(stats.memoryUsage.usagePercentage)}
                  showInfo={false}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="总内存"
                  value={formatBytes(stats.memoryUsage.total)}
                  valueStyle={{ fontSize: 12 }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="内存限制"
                  value={formatBytes(stats.memoryUsage.limit)}
                  valueStyle={{ fontSize: 12 }}
                />
              </Col>
            </Row>
          </Card>
        )}

        {/* Cache Statistics */}
        <Card size="small" title="缓存统计" style={{ marginTop: 16 }}>
          <Row gutter={[16, 8]}>
            <Col span={12}>
              <Statistic
                title="API缓存"
                value={`${stats.cacheStats.apiCache.size}/${stats.cacheStats.apiCache.maxSize}`}
                valueStyle={{ fontSize: 12 }}
              />
            </Col>
            <Col span={12}>
              <Statistic
                title="图表缓存"
                value={`${stats.cacheStats.chartCache.size}/${stats.cacheStats.chartCache.maxSize}`}
                valueStyle={{ fontSize: 12 }}
              />
            </Col>
            <Col span={24}>
              <div style={{ marginTop: 8 }}>
                <span>缓存命中率: </span>
                <span style={{ 
                  color: getPerformanceColor(1 - stats.cacheHitRate, { good: 0.3, warning: 0.5 }),
                  fontWeight: 'bold'
                }}>
                  {(stats.cacheHitRate * 100).toFixed(1)}%
                </span>
              </div>
            </Col>
          </Row>
        </Card>

        {/* Performance Recommendations */}
        {stats.recommendations.length > 0 && (
          <Card size="small" title="优化建议" style={{ marginTop: 16 }}>
            {stats.recommendations.map((recommendation, index) => (
              <Alert
                key={index}
                message={recommendation}
                type="warning"
                showIcon
                icon={<WarningOutlined />}
                style={{ marginBottom: 8, fontSize: 12 }}
              />
            ))}
          </Card>
        )}

        {/* Quick Actions */}
        <Card size="small" title="快速操作" style={{ marginTop: 16 }}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <Button
              size="small"
              block
              onClick={() => {
                apiCache.clear();
                chartDataCache.clear();
                refreshStats();
              }}
            >
              清理缓存
            </Button>
            <Button
              size="small"
              block
              onClick={() => {
                if (window.gc) {
                  window.gc();
                  setTimeout(refreshStats, 1000);
                } else {
                  console.warn('Garbage collection not available');
                }
              }}
            >
              触发垃圾回收 (仅开发环境)
            </Button>
          </Space>
        </Card>
      </Card>
    </div>
  );
};

// Hook for toggling performance dashboard
export const usePerformanceDashboard = () => {
  const [visible, setVisible] = useState(false);

  const toggle = () => setVisible(!visible);
  const show = () => setVisible(true);
  const hide = () => setVisible(false);

  // Keyboard shortcut (Ctrl+Shift+P)
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey && event.shiftKey && event.key === 'P') {
        event.preventDefault();
        toggle();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return {
    visible,
    toggle,
    show,
    hide,
    PerformanceDashboard: () => (
      <PerformanceDashboard visible={visible} onClose={hide} />
    )
  };
};