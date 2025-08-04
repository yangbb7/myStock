import React, { useState, useMemo } from 'react';
import {
  Card,
  Row,
  Col,
  Table,
  Button,
  Space,
  Typography,
  Select,
  Checkbox,
  Tag,
  Tooltip,
  Progress,
  Alert,
  Divider,
  InputNumber,
  Slider,
  message,
  Modal,
} from 'antd';
import {
  SwapOutlined,
  TrophyOutlined,
  LineChartOutlined,
  SettingOutlined,
  StarOutlined,
  StarFilled,
  EyeOutlined,
  ToolOutlined,
  PieChartOutlined,
} from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import dayjs from 'dayjs';
import type { BacktestResult, PerformanceMetrics } from '../../services/types';

const { Title, Text } = Typography;
const { Option } = Select;

interface StrategyComparisonProps {
  backtestHistory: BacktestResult[];
  onViewResult?: (result: BacktestResult) => void;
  className?: string;
}

interface ComparisonMetric {
  key: string;
  name: string;
  format: (value: number) => string;
  higher: boolean; // true if higher is better
}

interface PortfolioWeight {
  strategy: string;
  weight: number;
}

const COMPARISON_METRICS: ComparisonMetric[] = [
  { key: 'totalReturnPercent', name: '总收益率', format: (v) => `${v.toFixed(2)}%`, higher: true },
  { key: 'annualizedReturn', name: '年化收益率', format: (v) => `${v.toFixed(2)}%`, higher: true },
  { key: 'sharpeRatio', name: '夏普比率', format: (v) => v.toFixed(3), higher: true },
  { key: 'maxDrawdown', name: '最大回撤', format: (v) => `${v.toFixed(2)}%`, higher: false },
  { key: 'volatility', name: '波动率', format: (v) => `${v.toFixed(2)}%`, higher: false },
  { key: 'calmarRatio', name: '卡玛比率', format: (v) => v.toFixed(3), higher: true },
  { key: 'winRate', name: '胜率', format: (v) => `${v.toFixed(1)}%`, higher: true },
  { key: 'profitFactor', name: '盈亏比', format: (v) => v.toFixed(2), higher: true },
];

export const StrategyComparison: React.FC<StrategyComparisonProps> = ({
  backtestHistory,
  onViewResult,
  className,
}) => {
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([]);
  const [comparisonMetric, setComparisonMetric] = useState('totalReturnPercent');
  const [favorites, setFavorites] = useState<string[]>([]);
  const [portfolioWeights, setPortfolioWeights] = useState<PortfolioWeight[]>([]);
  const [showPortfolioModal, setShowPortfolioModal] = useState(false);

  // Group backtest results by strategy name
  const strategiesByName = useMemo(() => {
    const grouped: Record<string, BacktestResult[]> = {};
    backtestHistory.forEach(result => {
      const key = result.config.strategyName;
      if (!grouped[key]) {
        grouped[key] = [];
      }
      grouped[key].push(result);
    });
    return grouped;
  }, [backtestHistory]);

  // Get latest result for each strategy
  const latestResults = useMemo(() => {
    return Object.entries(strategiesByName).map(([strategyName, results]) => {
      const latest = results.sort((a, b) => 
        dayjs(b.config.endDate).valueOf() - dayjs(a.config.endDate).valueOf()
      )[0];
      return { strategyName, result: latest, totalRuns: results.length };
    });
  }, [strategiesByName]);

  // Filtered and sorted results for comparison
  const comparisonData = useMemo(() => {
    let data = latestResults;
    
    if (selectedStrategies.length > 0) {
      data = data.filter(item => selectedStrategies.includes(item.strategyName));
    }

    // Sort by selected metric
    const metric = COMPARISON_METRICS.find(m => m.key === comparisonMetric);
    if (metric) {
      data = data.sort((a, b) => {
        const aValue = (a.result.performance as any)[metric.key];
        const bValue = (b.result.performance as any)[metric.key];
        return metric.higher ? bValue - aValue : aValue - bValue;
      });
    }

    return data;
  }, [latestResults, selectedStrategies, comparisonMetric]);

  // Performance comparison chart
  const comparisonChartOptions = useMemo(() => {
    if (comparisonData.length === 0) return {};

    const strategies = comparisonData.map(item => item.strategyName);
    const values = comparisonData.map(item => 
      (item.result.performance as any)[comparisonMetric]
    );

    const metric = COMPARISON_METRICS.find(m => m.key === comparisonMetric);

    return {
      title: {
        text: `策略${metric?.name}对比`,
        left: 'center',
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow',
        },
        formatter: (params: any) => {
          const point = params[0];
          return `
            <div>
              <div><strong>${point.name}</strong></div>
              <div>${metric?.name}: ${metric?.format(point.value)}</div>
            </div>
          `;
        },
      },
      xAxis: {
        type: 'category',
        data: strategies,
        axisLabel: {
          rotate: 45,
          interval: 0,
        },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: (value: number) => metric?.format(value) || value.toString(),
        },
      },
      series: [
        {
          name: metric?.name,
          type: 'bar',
          data: values.map((value, index) => ({
            value,
            itemStyle: {
              color: index === 0 ? '#52c41a' : 
                     index === 1 ? '#1890ff' : 
                     index === 2 ? '#faad14' : '#d9d9d9',
            },
          })),
          label: {
            show: true,
            position: 'top',
            formatter: (params: any) => metric?.format(params.value) || params.value,
          },
        },
      ],
      grid: {
        left: '3%',
        right: '4%',
        bottom: '15%',
        containLabel: true,
      },
    };
  }, [comparisonData, comparisonMetric]);

  // Equity curves comparison chart
  const equityCurvesOptions = useMemo(() => {
    if (comparisonData.length === 0) return {};

    const colors = ['#1890ff', '#52c41a', '#faad14', '#f5222d', '#722ed1'];
    const series = comparisonData.slice(0, 5).map((item, index) => ({
      name: item.strategyName,
      type: 'line',
      data: item.result.equity.map(point => [
        dayjs(point.timestamp).valueOf(),
        point.equity,
      ]),
      smooth: true,
      lineStyle: {
        color: colors[index],
        width: 2,
      },
    }));

    return {
      title: {
        text: '资金曲线对比',
        left: 'center',
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          let content = `<div>时间: ${dayjs(params[0].data[0]).format('YYYY-MM-DD')}</div>`;
          params.forEach((param: any) => {
            content += `<div>${param.seriesName}: ¥${param.data[1].toLocaleString()}</div>`;
          });
          return content;
        },
      },
      legend: {
        top: 30,
        data: comparisonData.slice(0, 5).map(item => item.strategyName),
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
      series,
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        top: '15%',
        containLabel: true,
      },
    };
  }, [comparisonData]);

  // Table columns for strategy comparison
  const comparisonColumns = [
    {
      title: '排名',
      key: 'rank',
      width: 60,
      render: (_: any, __: any, index: number) => (
        <div style={{ textAlign: 'center' }}>
          {index === 0 ? <TrophyOutlined style={{ color: '#faad14' }} /> : index + 1}
        </div>
      ),
    },
    {
      title: '策略名称',
      key: 'strategyName',
      width: 150,
      render: (item: any) => (
        <Space>
          <Text strong>{item.strategyName}</Text>
          <Button
            type="text"
            size="small"
            icon={favorites.includes(item.strategyName) ? <StarFilled /> : <StarOutlined />}
            onClick={() => {
              if (favorites.includes(item.strategyName)) {
                setFavorites(prev => prev.filter(f => f !== item.strategyName));
              } else {
                setFavorites(prev => [...prev, item.strategyName]);
              }
            }}
          />
        </Space>
      ),
    },
    ...COMPARISON_METRICS.map(metric => ({
      title: metric.name,
      key: metric.key,
      width: 120,
      render: (item: any) => {
        const value = (item.result.performance as any)[metric.key];
        const isSelected = comparisonMetric === metric.key;
        return (
          <Text 
            strong={isSelected}
            style={{ 
              color: isSelected ? '#1890ff' : undefined,
              backgroundColor: isSelected ? '#f0f8ff' : undefined,
              padding: isSelected ? '2px 4px' : undefined,
              borderRadius: isSelected ? '4px' : undefined,
            }}
          >
            {metric.format(value)}
          </Text>
        );
      },
    })),
    {
      title: '回测次数',
      key: 'totalRuns',
      width: 100,
      render: (item: any) => (
        <Tag color="blue">{item.totalRuns}次</Tag>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 120,
      render: (item: any) => (
        <Space>
          <Button
            type="link"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => onViewResult?.(item.result)}
          >
            查看
          </Button>
        </Space>
      ),
    },
  ];

  const handleCreatePortfolio = () => {
    if (selectedStrategies.length < 2) {
      message.warning('请至少选择2个策略进行组合');
      return;
    }

    const weights = selectedStrategies.map(strategy => ({
      strategy,
      weight: Math.round(100 / selectedStrategies.length),
    }));
    
    setPortfolioWeights(weights);
    setShowPortfolioModal(true);
  };

  const handlePortfolioWeightChange = (strategy: string, weight: number) => {
    setPortfolioWeights(prev => 
      prev.map(item => 
        item.strategy === strategy ? { ...item, weight } : item
      )
    );
  };

  const calculatePortfolioMetrics = () => {
    if (portfolioWeights.length === 0) return null;

    const totalWeight = portfolioWeights.reduce((sum, item) => sum + item.weight, 0);
    if (totalWeight !== 100) return null;

    // Simplified portfolio calculation (weighted average of metrics)
    const portfolioMetrics: any = {};
    COMPARISON_METRICS.forEach(metric => {
      let weightedSum = 0;
      portfolioWeights.forEach(({ strategy, weight }) => {
        const strategyData = comparisonData.find(item => item.strategyName === strategy);
        if (strategyData) {
          const value = (strategyData.result.performance as any)[metric.key];
          weightedSum += value * (weight / 100);
        }
      });
      portfolioMetrics[metric.key] = weightedSum;
    });

    return portfolioMetrics;
  };

  if (backtestHistory.length === 0) {
    return (
      <div className={className}>
        <Alert
          message="暂无回测历史"
          description="请先运行一些回测，然后回到这里进行策略对比分析。"
          type="info"
          showIcon
        />
      </div>
    );
  }

  return (
    <div className={className}>
      <Row gutter={[16, 16]}>
        {/* Controls */}
        <Col span={24}>
          <Card>
            <Row gutter={16} align="middle">
              <Col span={8}>
                <Space direction="vertical" size="small" style={{ width: '100%' }}>
                  <Text strong>选择策略:</Text>
                  <Select
                    mode="multiple"
                    placeholder="选择要对比的策略"
                    style={{ width: '100%' }}
                    value={selectedStrategies}
                    onChange={setSelectedStrategies}
                  >
                    {latestResults.map(item => (
                      <Option key={item.strategyName} value={item.strategyName}>
                        <Space>
                          {item.strategyName}
                          {favorites.includes(item.strategyName) && <StarFilled style={{ color: '#faad14' }} />}
                          <Tag size="small">{item.totalRuns}次</Tag>
                        </Space>
                      </Option>
                    ))}
                  </Select>
                </Space>
              </Col>
              <Col span={6}>
                <Space direction="vertical" size="small" style={{ width: '100%' }}>
                  <Text strong>排序指标:</Text>
                  <Select
                    value={comparisonMetric}
                    onChange={setComparisonMetric}
                    style={{ width: '100%' }}
                  >
                    {COMPARISON_METRICS.map(metric => (
                      <Option key={metric.key} value={metric.key}>
                        {metric.name}
                      </Option>
                    ))}
                  </Select>
                </Space>
              </Col>
              <Col span={10}>
                <Space>
                  <Button
                    type="primary"
                    icon={<PieChartOutlined />}
                    onClick={handleCreatePortfolio}
                    disabled={selectedStrategies.length < 2}
                  >
                    创建策略组合
                  </Button>
                  <Button
                    icon={<ToolOutlined />}
                    disabled={selectedStrategies.length === 0}
                  >
                    参数优化建议
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Comparison Table */}
        <Col span={24}>
          <Card title="策略对比表">
            <Table
              columns={comparisonColumns}
              dataSource={comparisonData}
              rowKey="strategyName"
              pagination={false}
              scroll={{ x: 1400 }}
              rowSelection={{
                selectedRowKeys: selectedStrategies,
                onChange: (keys) => setSelectedStrategies(keys as string[]),
                getCheckboxProps: (record) => ({
                  name: record.strategyName,
                }),
              }}
            />
          </Card>
        </Col>

        {/* Charts */}
        {comparisonData.length > 0 && (
          <>
            <Col span={12}>
              <Card title="指标对比">
                <ReactECharts
                  option={comparisonChartOptions}
                  style={{ height: '400px' }}
                  notMerge={true}
                  lazyUpdate={true}
                />
              </Card>
            </Col>
            <Col span={12}>
              <Card title="资金曲线对比">
                <ReactECharts
                  option={equityCurvesOptions}
                  style={{ height: '400px' }}
                  notMerge={true}
                  lazyUpdate={true}
                />
              </Card>
            </Col>
          </>
        )}
      </Row>

      {/* Portfolio Creation Modal */}
      <Modal
        title="创建策略组合"
        open={showPortfolioModal}
        onCancel={() => setShowPortfolioModal(false)}
        onOk={() => {
          const portfolioMetrics = calculatePortfolioMetrics();
          if (portfolioMetrics) {
            message.success('策略组合创建成功');
            setShowPortfolioModal(false);
          } else {
            message.error('权重总和必须为100%');
          }
        }}
        width={600}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <Alert
            message="策略组合"
            description="通过设置不同策略的权重来创建组合策略，权重总和必须为100%。"
            type="info"
            showIcon
          />
          
          <Divider />
          
          {portfolioWeights.map(({ strategy, weight }) => (
            <div key={strategy}>
              <Row gutter={16} align="middle">
                <Col span={8}>
                  <Text strong>{strategy}</Text>
                </Col>
                <Col span={12}>
                  <Slider
                    min={0}
                    max={100}
                    value={weight}
                    onChange={(value) => handlePortfolioWeightChange(strategy, value)}
                  />
                </Col>
                <Col span={4}>
                  <InputNumber
                    min={0}
                    max={100}
                    value={weight}
                    onChange={(value) => handlePortfolioWeightChange(strategy, value || 0)}
                    formatter={(value) => `${value}%`}
                    parser={(value) => value!.replace('%', '')}
                    style={{ width: '100%' }}
                  />
                </Col>
              </Row>
            </div>
          ))}
          
          <Divider />
          
          <div>
            <Text strong>
              权重总和: {portfolioWeights.reduce((sum, item) => sum + item.weight, 0)}%
            </Text>
            <Progress
              percent={portfolioWeights.reduce((sum, item) => sum + item.weight, 0)}
              status={portfolioWeights.reduce((sum, item) => sum + item.weight, 0) === 100 ? 'success' : 'active'}
            />
          </div>

          {calculatePortfolioMetrics() && (
            <>
              <Divider />
              <div>
                <Text strong>预期组合指标:</Text>
                <Row gutter={16} style={{ marginTop: 8 }}>
                  {COMPARISON_METRICS.slice(0, 4).map(metric => {
                    const value = calculatePortfolioMetrics()![metric.key];
                    return (
                      <Col span={6} key={metric.key}>
                        <div style={{ textAlign: 'center' }}>
                          <div style={{ fontSize: '16px', fontWeight: 'bold' }}>
                            {metric.format(value)}
                          </div>
                          <div style={{ fontSize: '12px', color: '#666' }}>
                            {metric.name}
                          </div>
                        </div>
                      </Col>
                    );
                  })}
                </Row>
              </div>
            </>
          )}
        </Space>
      </Modal>
    </div>
  );
};

export default StrategyComparison;