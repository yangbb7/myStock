import React, { useState, useMemo } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Table,
  Button,
  Space,
  Typography,
  Tabs,
  Empty,
  Tag,
  Tooltip,
  Progress,
  Alert,
  Divider,
  message,
} from 'antd';
import {
  BarChartOutlined,
  LineChartOutlined,
  TableOutlined,
  DownloadOutlined,
  ShareAltOutlined,
  TrophyOutlined,
  RiseOutlined,
  FallOutlined,
  InfoCircleOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import dayjs from 'dayjs';
import { api } from '../../services/api';
import type { BacktestResult, TradeRecord, EquityPoint, DrawdownPoint } from '../../services/types';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface BacktestResultsProps {
  result: BacktestResult | null;
  onNewBacktest?: () => void;
  className?: string;
}

export const BacktestResults: React.FC<BacktestResultsProps> = ({
  result,
  onNewBacktest,
  className,
}) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [exportLoading, setExportLoading] = useState(false);

  // Early return with error boundary if result is malformed
  if (result && typeof result !== 'object') {
    console.error('BacktestResults: Invalid result object type:', typeof result);
    return (
      <div className={className}>
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="回测结果数据格式错误"
        >
          <Button type="primary" onClick={onNewBacktest}>
            重新开始回测
          </Button>
        </Empty>
      </div>
    );
  }

  // Performance metrics calculations
  const performanceStats = useMemo(() => {
    if (!result || !result.performance) return null;

    const { performance } = result;
    
    // Additional safety check for performance object
    if (!performance || typeof performance !== 'object') return null;
    return [
      {
        title: '总收益率',
        value: performance.totalReturnPercent || 0,
        suffix: '%',
        precision: 2,
        valueStyle: { color: (performance.totalReturnPercent || 0) >= 0 ? '#3f8600' : '#cf1322' },
        prefix: (performance.totalReturnPercent || 0) >= 0 ? <RiseOutlined /> : <FallOutlined />,
      },
      {
        title: '年化收益率',
        value: performance.annualizedReturn || 0,
        suffix: '%',
        precision: 2,
        valueStyle: { color: (performance.annualizedReturn || 0) >= 0 ? '#3f8600' : '#cf1322' },
      },
      {
        title: '夏普比率',
        value: performance.sharpeRatio || 0,
        precision: 3,
        valueStyle: { 
          color: (performance.sharpeRatio || 0) >= 1 ? '#3f8600' : 
                 (performance.sharpeRatio || 0) >= 0.5 ? '#faad14' : '#cf1322' 
        },
      },
      {
        title: '最大回撤',
        value: performance.maxDrawdown || 0,
        suffix: '%',
        precision: 2,
        valueStyle: { color: '#cf1322' },
        prefix: <FallOutlined />,
      },
      {
        title: '胜率',
        value: performance.winRate || 0,
        suffix: '%',
        precision: 1,
        valueStyle: { color: (performance.winRate || 0) >= 50 ? '#3f8600' : '#cf1322' },
      },
      {
        title: '盈亏比',
        value: performance.profitFactor || 0,
        precision: 2,
        valueStyle: { 
          color: (performance.profitFactor || 0) >= 1.5 ? '#3f8600' : 
                 (performance.profitFactor || 0) >= 1 ? '#faad14' : '#cf1322' 
        },
      },
    ];
  }, [result]);

  // Equity curve chart options
  const equityChartOptions = useMemo(() => {
    if (!result?.equity || !Array.isArray(result.equity) || result.equity.length === 0) return {};

    const data = result.equity.map(point => {
      if (!point || typeof point !== 'object') return [0, 0];
      return [
        point.timestamp ? dayjs(point.timestamp).valueOf() : 0,
        point.equity || 0,
      ];
    });

    return {
      title: {
        text: '资金曲线',
        left: 'center',
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const point = params[0];
          return `
            <div>
              <div>时间: ${dayjs(point.data[0]).format('YYYY-MM-DD')}</div>
              <div>资金: ¥${point.data[1].toLocaleString()}</div>
            </div>
          `;
        },
      },
      xAxis: {
        type: 'time',
        axisLabel: {
          formatter: (value: number) => dayjs(value).format('MM-DD'),
        },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: (value: number) => `¥${(value / 10000).toFixed(0)}万`,
        },
      },
      series: [
        {
          name: '资金',
          type: 'line',
          data,
          smooth: true,
          lineStyle: {
            color: '#1890ff',
            width: 2,
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(24, 144, 255, 0.3)' },
                { offset: 1, color: 'rgba(24, 144, 255, 0.1)' },
              ],
            },
          },
        },
      ],
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true,
      },
    };
  }, [result]);

  // Drawdown chart options
  const drawdownChartOptions = useMemo(() => {
    if (!result?.drawdown || !Array.isArray(result.drawdown) || result.drawdown.length === 0) return {};

    const data = result.drawdown.map(point => {
      if (!point || typeof point !== 'object') return [0, 0];
      return [
        point.timestamp ? dayjs(point.timestamp).valueOf() : 0,
        point.drawdownPercent || 0,
      ];
    });

    return {
      title: {
        text: '回撤曲线',
        left: 'center',
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const point = params[0];
          return `
            <div>
              <div>时间: ${dayjs(point.data[0]).format('YYYY-MM-DD')}</div>
              <div>回撤: ${point.data[1].toFixed(2)}%</div>
            </div>
          `;
        },
      },
      xAxis: {
        type: 'time',
        axisLabel: {
          formatter: (value: number) => dayjs(value).format('MM-DD'),
        },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: (value: number) => `${value.toFixed(1)}%`,
        },
        max: 0,
      },
      series: [
        {
          name: '回撤',
          type: 'line',
          data,
          smooth: true,
          lineStyle: {
            color: '#ff4d4f',
            width: 2,
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(255, 77, 79, 0.3)' },
                { offset: 1, color: 'rgba(255, 77, 79, 0.1)' },
              ],
            },
          },
        },
      ],
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true,
      },
    };
  }, [result]);

  // Trade table columns
  const tradeColumns = [
    {
      title: '股票代码',
      dataIndex: 'symbol',
      key: 'symbol',
      width: 100,
    },
    {
      title: '方向',
      dataIndex: 'side',
      key: 'side',
      width: 80,
      render: (side: string) => (
        <Tag color={side === 'BUY' ? 'green' : 'red'}>
          {side === 'BUY' ? '买入' : '卖出'}
        </Tag>
      ),
    },
    {
      title: '数量',
      dataIndex: 'quantity',
      key: 'quantity',
      width: 100,
      render: (value: number) => value.toLocaleString(),
    },
    {
      title: '开仓价格',
      dataIndex: 'entryPrice',
      key: 'entryPrice',
      width: 100,
      render: (value: number) => `¥${value.toFixed(2)}`,
    },
    {
      title: '平仓价格',
      dataIndex: 'exitPrice',
      key: 'exitPrice',
      width: 100,
      render: (value: number) => `¥${value.toFixed(2)}`,
    },
    {
      title: '盈亏',
      dataIndex: 'pnl',
      key: 'pnl',
      width: 120,
      render: (value: number) => (
        <span style={{ color: value >= 0 ? '#3f8600' : '#cf1322' }}>
          {value >= 0 ? '+' : ''}¥{value.toFixed(2)}
        </span>
      ),
    },
    {
      title: '盈亏率',
      dataIndex: 'pnlPercent',
      key: 'pnlPercent',
      width: 100,
      render: (value: number) => (
        <span style={{ color: value >= 0 ? '#3f8600' : '#cf1322' }}>
          {value >= 0 ? '+' : ''}{value.toFixed(2)}%
        </span>
      ),
    },
    {
      title: '开仓时间',
      dataIndex: 'entryTime',
      key: 'entryTime',
      width: 120,
      render: (value: string) => dayjs(value).format('MM-DD HH:mm'),
    },
    {
      title: '平仓时间',
      dataIndex: 'exitTime',
      key: 'exitTime',
      width: 120,
      render: (value: string) => dayjs(value).format('MM-DD HH:mm'),
    },
    {
      title: '持仓时长',
      dataIndex: 'duration',
      key: 'duration',
      width: 100,
      render: (value: number) => {
        const hours = Math.floor(value / 3600);
        const minutes = Math.floor((value % 3600) / 60);
        return `${hours}h ${minutes}m`;
      },
    },
  ];

  const handleExport = async (format: 'pdf' | 'excel') => {
    if (!result) return;

    try {
      setExportLoading(true);
      const blob = await api.analytics.exportData('backtest', format, {
        backtestId: result.config?.strategyName || 'unknown',
        startDate: result.config?.startDate || '',
        endDate: result.config?.endDate || '',
      });

      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `backtest_${result.config?.strategyName || 'unknown'}_${dayjs().format('YYYYMMDD')}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      message.success(`${format.toUpperCase()}报告导出成功`);
    } catch (error: any) {
      message.error(`导出失败: ${error.message}`);
    } finally {
      setExportLoading(false);
    }
  };

  const handleShare = () => {
    if (!result) return;

    const shareData = {
      strategy: result.config?.strategyName || '未知策略',
      period: `${result.config?.startDate || ''} ~ ${result.config?.endDate || ''}`,
      return: `${(result.performance?.totalReturnPercent || 0).toFixed(2)}%`,
      sharpe: (result.performance?.sharpeRatio || 0).toFixed(3),
      maxDrawdown: `${(result.performance?.maxDrawdown || 0).toFixed(2)}%`,
    };

    const shareText = `回测结果分享：
策略：${shareData.strategy}
时间：${shareData.period}
总收益率：${shareData.return}
夏普比率：${shareData.sharpe}
最大回撤：${shareData.maxDrawdown}`;

    navigator.clipboard.writeText(shareText).then(() => {
      message.success('回测结果已复制到剪贴板');
    });
  };

  if (!result) {
    return (
      <div className={className}>
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="暂无回测结果"
        >
          <Button type="primary" onClick={onNewBacktest}>
            开始新的回测
          </Button>
        </Empty>
      </div>
    );
  }

  return (
    <div className={className}>
      <Row gutter={[16, 16]}>
        {/* Header */}
        <Col span={24}>
          <Card>
            <Row justify="space-between" align="middle">
              <Col>
                <Space direction="vertical" size="small">
                  <Title level={3} style={{ margin: 0 }}>
                    <TrophyOutlined style={{ marginRight: 8 }} />
                    {result.config?.strategyName || '未知策略'} 回测结果
                  </Title>
                  <Text type="secondary">
                    {result.config?.startDate || ''} ~ {result.config?.endDate || ''} | 
                    初始资金: ¥{(result.config?.initialCapital || 0).toLocaleString()}
                  </Text>
                </Space>
              </Col>
              <Col>
                <Space>
                  <Button
                    icon={<DownloadOutlined />}
                    loading={exportLoading}
                    onClick={() => handleExport('pdf')}
                  >
                    导出PDF
                  </Button>
                  <Button
                    icon={<DownloadOutlined />}
                    loading={exportLoading}
                    onClick={() => handleExport('excel')}
                  >
                    导出Excel
                  </Button>
                  <Button
                    icon={<ShareAltOutlined />}
                    onClick={handleShare}
                  >
                    分享结果
                  </Button>
                  <Button
                    icon={<ReloadOutlined />}
                    onClick={onNewBacktest}
                  >
                    新建回测
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Performance Overview */}
        <Col span={24}>
          <Card title="绩效概览">
            <Row gutter={16}>
              {performanceStats?.map((stat, index) => (
                <Col span={4} key={index}>
                  <Statistic
                    title={stat.title}
                    value={stat.value}
                    suffix={stat.suffix}
                    precision={stat.precision}
                    valueStyle={stat.valueStyle}
                    prefix={stat.prefix}
                  />
                </Col>
              ))}
            </Row>
          </Card>
        </Col>

        {/* Detailed Analysis */}
        <Col span={24}>
          <Card>
            <Tabs activeKey={activeTab} onChange={setActiveTab}>
              <TabPane
                tab={
                  <span>
                    <LineChartOutlined />
                    图表分析
                  </span>
                }
                key="charts"
              >
                <Row gutter={[16, 16]}>
                  <Col span={24}>
                    <Card>
                      <ReactECharts
                        option={equityChartOptions}
                        style={{ height: '400px' }}
                        notMerge={true}
                        lazyUpdate={true}
                      />
                    </Card>
                  </Col>
                  <Col span={24}>
                    <Card>
                      <ReactECharts
                        option={drawdownChartOptions}
                        style={{ height: '300px' }}
                        notMerge={true}
                        lazyUpdate={true}
                      />
                    </Card>
                  </Col>
                </Row>
              </TabPane>

              <TabPane
                tab={
                  <span>
                    <TableOutlined />
                    交易记录
                  </span>
                }
                key="trades"
              >
                <Table
                  columns={tradeColumns}
                  dataSource={result.trades || []}
                  rowKey={(record, index) => `${record.symbol}_${record.entryTime}_${index}`}
                  scroll={{ x: 1200 }}
                  pagination={{
                    pageSize: 20,
                    showSizeChanger: true,
                    showQuickJumper: true,
                    showTotal: (total) => `共 ${total} 笔交易`,
                  }}
                />
              </TabPane>

              <TabPane
                tab={
                  <span>
                    <BarChartOutlined />
                    详细指标
                  </span>
                }
                key="metrics"
              >
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Card title="收益指标" size="small">
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text>总收益:</Text>
                          <Text strong>¥{(result.performance?.totalReturn || 0).toLocaleString()}</Text>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text>总收益率:</Text>
                          <Text strong style={{ color: (result.performance?.totalReturnPercent || 0) >= 0 ? '#3f8600' : '#cf1322' }}>
                            {(result.performance?.totalReturnPercent || 0).toFixed(2)}%
                          </Text>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text>年化收益率:</Text>
                          <Text strong>{(result.performance?.annualizedReturn || 0).toFixed(2)}%</Text>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text>波动率:</Text>
                          <Text strong>{(result.performance?.volatility || 0).toFixed(2)}%</Text>
                        </div>
                      </Space>
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card title="风险指标" size="small">
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text>夏普比率:</Text>
                          <Text strong>{(result.performance?.sharpeRatio || 0).toFixed(3)}</Text>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text>最大回撤:</Text>
                          <Text strong style={{ color: '#cf1322' }}>
                            {(result.performance?.maxDrawdown || 0).toFixed(2)}%
                          </Text>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text>卡玛比率:</Text>
                          <Text strong>{(result.performance?.calmarRatio || 0).toFixed(3)}</Text>
                        </div>
                      </Space>
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card title="交易统计" size="small">
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text>总交易次数:</Text>
                          <Text strong>{result.performance?.totalTrades || 0}</Text>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text>盈利交易:</Text>
                          <Text strong style={{ color: '#3f8600' }}>{result.performance?.winningTrades || 0}</Text>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text>亏损交易:</Text>
                          <Text strong style={{ color: '#cf1322' }}>{result.performance?.losingTrades || 0}</Text>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text>胜率:</Text>
                          <Text strong>{(result.performance?.winRate || 0).toFixed(1)}%</Text>
                        </div>
                      </Space>
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card title="盈亏分析" size="small">
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text>平均盈利:</Text>
                          <Text strong style={{ color: '#3f8600' }}>¥{(result.performance?.avgWin || 0).toFixed(2)}</Text>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text>平均亏损:</Text>
                          <Text strong style={{ color: '#cf1322' }}>¥{(result.performance?.avgLoss || 0).toFixed(2)}</Text>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text>盈亏比:</Text>
                          <Text strong>{(result.performance?.profitFactor || 0).toFixed(2)}</Text>
                        </div>
                      </Space>
                    </Card>
                  </Col>
                </Row>
              </TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default BacktestResults;