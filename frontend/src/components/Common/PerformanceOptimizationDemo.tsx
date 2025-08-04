import React, { useState, useMemo, useCallback } from 'react';
import { Card, Row, Col, Switch, Slider, Button, Statistic, Alert, Space, Tabs } from 'antd';
import { OptimizedChart } from '../Charts/OptimizedChart';
import { VirtualScrollList } from './VirtualScrollList';
import { performanceOptimizer } from '../../utils/performanceOptimizer';
import { cacheManager } from '../../utils/cacheManager';
import { performanceMonitor } from '../../utils/performanceMonitor';

const { TabPane } = Tabs;

// Generate sample data for testing
const generateTimeSeriesData = (count: number) => {
  const data = [];
  const startTime = Date.now() - count * 60000; // 1 minute intervals
  
  for (let i = 0; i < count; i++) {
    data.push({
      x: startTime + i * 60000,
      y: Math.random() * 100 + Math.sin(i / 10) * 20,
      timestamp: startTime + i * 60000,
      value: Math.random() * 100 + Math.sin(i / 10) * 20
    });
  }
  
  return data;
};

const generateFinancialData = (count: number) => {
  const data = [];
  let price = 100;
  
  for (let i = 0; i < count; i++) {
    const change = (Math.random() - 0.5) * 4;
    price += change;
    
    data.push({
      timestamp: Date.now() - (count - i) * 60000,
      open: price,
      high: price + Math.random() * 2,
      low: price - Math.random() * 2,
      close: price + change,
      volume: Math.floor(Math.random() * 10000) + 1000,
      value: price
    });
  }
  
  return data;
};

const generateListData = (count: number) => {
  return Array.from({ length: count }, (_, i) => ({
    id: i,
    name: `Item ${i}`,
    value: Math.random() * 1000,
    category: `Category ${i % 10}`,
    timestamp: Date.now() - i * 1000
  }));
};

export const PerformanceOptimizationDemo: React.FC = () => {
  const [dataSize, setDataSize] = useState(1000);
  const [enableSampling, setEnableSampling] = useState(true);
  const [enableCaching, setEnableCaching] = useState(true);
  const [enableVirtualization, setEnableVirtualization] = useState(true);
  const [samplingStrategy, setSamplingStrategy] = useState<'uniform' | 'lttb' | 'peakPreserving' | 'smart'>('smart');
  const [performanceReport, setPerformanceReport] = useState<any>(null);

  // Generate test data
  const timeSeriesData = useMemo(() => generateTimeSeriesData(dataSize), [dataSize]);
  const financialData = useMemo(() => generateFinancialData(dataSize), [dataSize]);
  const listData = useMemo(() => generateListData(dataSize), [dataSize]);

  // Chart options
  const lineChartOption = useMemo(() => ({
    title: { text: '时间序列数据' },
    xAxis: { type: 'time' },
    yAxis: { type: 'value' },
    series: [{
      type: 'line',
      data: timeSeriesData.map(d => [d.x, d.y]),
      smooth: true
    }]
  }), [timeSeriesData]);

  const candlestickChartOption = useMemo(() => ({
    title: { text: '金融K线数据' },
    xAxis: { type: 'time' },
    yAxis: { type: 'value' },
    series: [{
      type: 'candlestick',
      data: financialData.map(d => [d.timestamp, d.open, d.close, d.low, d.high])
    }]
  }), [financialData]);

  // Performance monitoring
  const updatePerformanceReport = useCallback(() => {
    const optimizerReport = performanceOptimizer.getOptimizationReport();
    const cacheStats = cacheManager.getGlobalStats();
    const monitorMetrics = performanceMonitor.getMetrics();
    
    setPerformanceReport({
      optimizer: optimizerReport,
      cache: cacheStats,
      monitor: monitorMetrics
    });
  }, []);

  // Force optimization
  const forceOptimization = useCallback(() => {
    (performanceOptimizer as any).runOptimizations();
    setTimeout(updatePerformanceReport, 1000);
  }, [updatePerformanceReport]);

  // Clear caches
  const clearCaches = useCallback(() => {
    cacheManager.cleanup();
    updatePerformanceReport();
  }, [updatePerformanceReport]);

  // List item renderer
  const renderListItem = useCallback((item: any, index: number, style: React.CSSProperties) => (
    <div style={{ ...style, padding: '8px 16px', borderBottom: '1px solid #f0f0f0' }}>
      <div style={{ fontWeight: 'bold' }}>{item.name}</div>
      <div style={{ color: '#666', fontSize: '12px' }}>
        值: {item.value.toFixed(2)} | 分类: {item.category}
      </div>
    </div>
  ), []);

  return (
    <div style={{ padding: '24px' }}>
      <Card title="性能优化演示" style={{ marginBottom: '24px' }}>
        <Row gutter={[16, 16]}>
          <Col span={6}>
            <Card size="small" title="数据大小">
              <Slider
                min={100}
                max={50000}
                step={100}
                value={dataSize}
                onChange={setDataSize}
                marks={{
                  100: '100',
                  1000: '1K',
                  10000: '10K',
                  50000: '50K'
                }}
              />
              <div style={{ textAlign: 'center', marginTop: '8px' }}>
                {dataSize.toLocaleString()} 数据点
              </div>
            </Card>
          </Col>
          
          <Col span={6}>
            <Card size="small" title="优化选项">
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Switch 
                    checked={enableSampling} 
                    onChange={setEnableSampling} 
                  /> 数据采样
                </div>
                <div>
                  <Switch 
                    checked={enableCaching} 
                    onChange={setEnableCaching} 
                  /> 缓存优化
                </div>
                <div>
                  <Switch 
                    checked={enableVirtualization} 
                    onChange={setEnableVirtualization} 
                  /> 虚拟滚动
                </div>
              </Space>
            </Card>
          </Col>
          
          <Col span={6}>
            <Card size="small" title="性能控制">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Button onClick={updatePerformanceReport} block>
                  更新性能报告
                </Button>
                <Button onClick={forceOptimization} block>
                  强制优化
                </Button>
                <Button onClick={clearCaches} block danger>
                  清空缓存
                </Button>
              </Space>
            </Card>
          </Col>
          
          <Col span={6}>
            {performanceReport && (
              <Card size="small" title="性能指标">
                <Statistic
                  title="内存使用"
                  value={performanceReport.optimizer.memory?.usagePercentage || 0}
                  precision={1}
                  suffix="%"
                />
                <Statistic
                  title="缓存命中"
                  value={performanceReport.cache.totalSize}
                  suffix="项"
                />
              </Card>
            )}
          </Col>
        </Row>
      </Card>

      <Tabs defaultActiveKey="charts">
        <TabPane tab="图表优化" key="charts">
          <Row gutter={[16, 16]}>
            <Col span={12}>
              <OptimizedChart
                data={timeSeriesData}
                option={lineChartOption}
                title="时间序列图表"
                height={400}
                maxDataPoints={1000}
                enableSampling={enableSampling}
                enableCaching={enableCaching}
                samplingStrategy={samplingStrategy}
                dataType="timeseries"
                showPerformanceInfo={true}
                enableExport={true}
                onDataSampled={(original, sampled) => {
                  console.log(`数据采样: ${original} -> ${sampled}`);
                }}
              />
            </Col>
            
            <Col span={12}>
              <OptimizedChart
                data={financialData}
                option={candlestickChartOption}
                title="金融K线图表"
                height={400}
                maxDataPoints={1000}
                enableSampling={enableSampling}
                enableCaching={enableCaching}
                samplingStrategy="peakPreserving"
                dataType="financial"
                showPerformanceInfo={true}
                enableExport={true}
              />
            </Col>
          </Row>
        </TabPane>
        
        <TabPane tab="列表虚拟化" key="virtualization">
          <Row gutter={[16, 16]}>
            <Col span={24}>
              <Card title={`虚拟滚动列表 (${listData.length.toLocaleString()} 项)`}>
                <VirtualScrollList
                  items={listData}
                  height={500}
                  itemHeight={60}
                  threshold={enableVirtualization ? 100 : Infinity}
                  renderItem={renderListItem}
                  enableInfiniteScroll={false}
                />
              </Card>
            </Col>
          </Row>
        </TabPane>
        
        <TabPane tab="性能报告" key="performance">
          {performanceReport && (
            <Row gutter={[16, 16]}>
              <Col span={8}>
                <Card title="优化器状态" size="small">
                  <pre style={{ fontSize: '12px', maxHeight: '300px', overflow: 'auto' }}>
                    {JSON.stringify(performanceReport.optimizer, null, 2)}
                  </pre>
                </Card>
              </Col>
              
              <Col span={8}>
                <Card title="缓存统计" size="small">
                  <pre style={{ fontSize: '12px', maxHeight: '300px', overflow: 'auto' }}>
                    {JSON.stringify(performanceReport.cache, null, 2)}
                  </pre>
                </Card>
              </Col>
              
              <Col span={8}>
                <Card title="监控指标" size="small">
                  <pre style={{ fontSize: '12px', maxHeight: '300px', overflow: 'auto' }}>
                    {JSON.stringify(performanceReport.monitor, null, 2)}
                  </pre>
                </Card>
              </Col>
            </Row>
          )}
          
          <Alert
            message="性能优化提示"
            description="当数据量超过阈值时，系统会自动启用数据采样、虚拟滚动和缓存优化。您可以通过上述控制面板调整优化策略。"
            type="info"
            showIcon
            style={{ marginTop: '16px' }}
          />
        </TabPane>
      </Tabs>
    </div>
  );
};