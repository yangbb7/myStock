import React, { useState, useMemo } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Select,
  DatePicker,
  Button,
  Space,
  Table,
  Tag,
  Tooltip,
  Progress,
  Alert,
  Divider,
} from 'antd';
import type { ColumnsType } from 'antd/es/table';
import {
  ReloadOutlined,
  WarningOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined,
  FireOutlined,
} from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import dayjs from 'dayjs';
import { api } from '../../../services/api';
import type { RiskMetrics, PortfolioSummary } from '../../../services/types';
import { StatisticCard } from '../../../components/Common/StatisticCard';
import { LoadingState } from '../../../components/Common/LoadingState';
import { BaseChart } from '../../../components/Charts/BaseChart';
import type { EChartsOption } from 'echarts';

const { RangePicker } = DatePicker;

interface RiskContribution {
  symbol: string;
  weight: number;
  volatility: number;
  var95: number;
  var99: number;
  beta: number;
  riskContribution: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
}

interface CorrelationData {
  symbol1: string;
  symbol2: string;
  correlation: number;
  riskLevel: 'low' | 'medium' | 'high';
}

interface RiskAlert {
  type: 'warning' | 'error' | 'critical';
  title: string;
  message: string;
  value: number;
  threshold: number;
  recommendation: string;
}

export const RiskAnalysis: React.FC = () => {
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs] | null>([
    dayjs().subtract(3, 'months'),
    dayjs(),
  ]);
  const [confidenceLevel, setConfidenceLevel] = useState<number>(95);
  const [riskHorizon, setRiskHorizon] = useState<number>(1); // days

  // Fetch risk metrics
  const {
    data: riskMetrics,
    isLoading: riskLoading,
    error: riskError,
    refetch: refetchRisk,
  } = useQuery({
    queryKey: ['riskMetrics'],
    queryFn: () => api.risk.getMetrics(),
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  // Fetch portfolio summary for risk calculation
  const {
    data: portfolioSummary,
    isLoading: portfolioLoading,
  } = useQuery({
    queryKey: ['portfolioSummary'],
    queryFn: () => api.portfolio.getSummary(),
    refetchInterval: 30000,
  });

  // Calculate individual position risks
  const positionRisks = useMemo((): RiskContribution[] => {
    if (!portfolioSummary?.positions) return [];

    return Object.entries(portfolioSummary.positions).map(([symbol, position]) => {
      const weight = portfolioSummary.totalValue > 0 ? 
        (position.quantity * position.currentPrice) / portfolioSummary.totalValue * 100 : 0;
      
      // Mock risk calculations (in real implementation, these would be calculated from historical data)
      const volatility = Math.random() * 30 + 10; // 10-40% volatility
      const beta = Math.random() * 2 + 0.5; // 0.5-2.5 beta
      const var95 = volatility * 1.645 / Math.sqrt(252); // Daily VaR at 95%
      const var99 = volatility * 2.326 / Math.sqrt(252); // Daily VaR at 99%
      const riskContribution = weight * volatility / 100;
      
      let riskLevel: 'low' | 'medium' | 'high' | 'critical' = 'low';
      if (volatility > 30) riskLevel = 'critical';
      else if (volatility > 25) riskLevel = 'high';
      else if (volatility > 20) riskLevel = 'medium';

      return {
        symbol,
        weight,
        volatility,
        var95,
        var99,
        beta,
        riskContribution,
        riskLevel,
      };
    }).sort((a, b) => b.riskContribution - a.riskContribution);
  }, [portfolioSummary]);

  // Generate correlation matrix (mock data)
  const correlationData = useMemo((): CorrelationData[] => {
    if (positionRisks.length < 2) return [];

    const correlations: CorrelationData[] = [];
    for (let i = 0; i < positionRisks.length; i++) {
      for (let j = i + 1; j < positionRisks.length; j++) {
        const correlation = (Math.random() - 0.5) * 2; // -1 to 1
        let riskLevel: 'low' | 'medium' | 'high' = 'low';
        if (Math.abs(correlation) > 0.8) riskLevel = 'high';
        else if (Math.abs(correlation) > 0.6) riskLevel = 'medium';

        correlations.push({
          symbol1: positionRisks[i].symbol,
          symbol2: positionRisks[j].symbol,
          correlation,
          riskLevel,
        });
      }
    }
    return correlations.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
  }, [positionRisks]);

  // Generate risk alerts
  const riskAlerts = useMemo((): RiskAlert[] => {
    const alerts: RiskAlert[] = [];

    if (riskMetrics) {
      // Daily P&L alert
      if (riskMetrics.dailyPnl < -riskMetrics.riskLimits.maxDailyLoss * 0.8) {
        alerts.push({
          type: 'warning',
          title: '日损失接近限制',
          message: '当日损失已接近最大日损失限制的80%',
          value: Math.abs(riskMetrics.dailyPnl),
          threshold: riskMetrics.riskLimits.maxDailyLoss,
          recommendation: '建议减少高风险仓位或增加对冲',
        });
      }

      // Drawdown alert
      if (Math.abs(riskMetrics.currentDrawdown) > riskMetrics.riskLimits.maxDrawdownLimit * 0.7) {
        alerts.push({
          type: 'error',
          title: '回撤风险较高',
          message: '当前回撤已超过最大回撤限制的70%',
          value: Math.abs(riskMetrics.currentDrawdown),
          threshold: riskMetrics.riskLimits.maxDrawdownLimit,
          recommendation: '建议降低整体仓位或调整投资策略',
        });
      }

      // Risk utilization alert
      if (riskMetrics.riskUtilization.dailyLossRatio > 0.9) {
        alerts.push({
          type: 'critical',
          title: '风险利用率过高',
          message: '日损失风险利用率已超过90%',
          value: riskMetrics.riskUtilization.dailyLossRatio * 100,
          threshold: 90,
          recommendation: '立即减少高风险仓位，控制风险敞口',
        });
      }
    }

    // High concentration risk
    const maxWeight = Math.max(...positionRisks.map(p => p.weight));
    if (maxWeight > 30) {
      alerts.push({
        type: 'warning',
        title: '集中度风险',
        message: '单一持仓权重过高，存在集中度风险',
        value: maxWeight,
        threshold: 30,
        recommendation: '建议分散投资，降低单一持仓权重',
      });
    }

    // High correlation risk
    const highCorrelations = correlationData.filter(c => Math.abs(c.correlation) > 0.8);
    if (highCorrelations.length > 0) {
      alerts.push({
        type: 'warning',
        title: '相关性风险',
        message: `发现${highCorrelations.length}对高相关性资产`,
        value: highCorrelations.length,
        threshold: 0,
        recommendation: '建议降低高相关性资产的权重，增加投资组合多样性',
      });
    }

    return alerts;
  }, [riskMetrics, positionRisks, correlationData]);

  // Risk contribution chart
  const riskChartOption = useMemo((): EChartsOption => {
    if (positionRisks.length === 0) {
      return {
        title: {
          text: '暂无数据',
          left: 'center',
          top: 'middle',
          textStyle: { color: '#999', fontSize: 14 },
        },
      };
    }

    return {
      title: {
        text: '风险贡献分析',
        left: 'center',
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow',
        },
        formatter: (params: any) => {
          const data = params[0];
          const item = positionRisks.find(p => p.symbol === data.name);
          return `
            <div>
              <strong>${data.name}</strong><br/>
              风险贡献: ${data.value.toFixed(2)}%<br/>
              权重: ${item?.weight.toFixed(2)}%<br/>
              波动率: ${item?.volatility.toFixed(2)}%<br/>
              Beta: ${item?.beta.toFixed(2)}
            </div>
          `;
        },
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: positionRisks.map(p => p.symbol),
        axisLabel: {
          rotate: 45,
          interval: 0,
        },
      },
      yAxis: {
        type: 'value',
        name: '风险贡献 (%)',
      },
      series: [
        {
          name: '风险贡献',
          type: 'bar',
          data: positionRisks.map(item => ({
            value: item.riskContribution,
            itemStyle: {
              color: item.riskLevel === 'critical' ? '#ff4d4f' :
                     item.riskLevel === 'high' ? '#fa8c16' :
                     item.riskLevel === 'medium' ? '#faad14' : '#52c41a',
            },
          })),
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowOffsetX: 0,
              shadowColor: 'rgba(0, 0, 0, 0.5)',
            },
          },
        },
      ],
    };
  }, [positionRisks]);

  // Risk metrics table columns
  const riskColumns: ColumnsType<RiskContribution> = [
    {
      title: '股票代码',
      dataIndex: 'symbol',
      key: 'symbol',
      width: 120,
      render: (symbol: string) => <strong>{symbol}</strong>,
    },
    {
      title: '权重',
      dataIndex: 'weight',
      key: 'weight',
      width: 100,
      align: 'right',
      render: (weight: number) => `${weight.toFixed(2)}%`,
      sorter: (a, b) => a.weight - b.weight,
    },
    {
      title: '波动率',
      dataIndex: 'volatility',
      key: 'volatility',
      width: 100,
      align: 'right',
      render: (volatility: number) => `${volatility.toFixed(2)}%`,
      sorter: (a, b) => a.volatility - b.volatility,
    },
    {
      title: 'Beta系数',
      dataIndex: 'beta',
      key: 'beta',
      width: 100,
      align: 'right',
      render: (beta: number) => beta.toFixed(2),
      sorter: (a, b) => a.beta - b.beta,
    },
    {
      title: `VaR(${confidenceLevel}%)`,
      dataIndex: confidenceLevel === 95 ? 'var95' : 'var99',
      key: 'var',
      width: 120,
      align: 'right',
      render: (var_: number) => `${var_.toFixed(2)}%`,
      sorter: (a, b) => (confidenceLevel === 95 ? a.var95 - b.var95 : a.var99 - b.var99),
    },
    {
      title: '风险贡献',
      dataIndex: 'riskContribution',
      key: 'riskContribution',
      width: 120,
      align: 'right',
      render: (contribution: number) => `${contribution.toFixed(2)}%`,
      sorter: (a, b) => a.riskContribution - b.riskContribution,
    },
    {
      title: '风险等级',
      dataIndex: 'riskLevel',
      key: 'riskLevel',
      width: 100,
      align: 'center',
      render: (level: string) => {
        const config = {
          low: { color: 'green', text: '低' },
          medium: { color: 'orange', text: '中' },
          high: { color: 'red', text: '高' },
          critical: { color: 'red', text: '极高' },
        };
        return <Tag color={config[level as keyof typeof config].color}>
          {config[level as keyof typeof config].text}
        </Tag>;
      },
      filters: [
        { text: '低风险', value: 'low' },
        { text: '中风险', value: 'medium' },
        { text: '高风险', value: 'high' },
        { text: '极高风险', value: 'critical' },
      ],
      onFilter: (value, record) => record.riskLevel === value,
    },
  ];

  // Mock portfolio-level risk metrics
  const portfolioRiskMetrics = {
    portfolioVar95: 2.5,
    portfolioVar99: 3.8,
    expectedShortfall: 4.2,
    portfolioVolatility: 18.5,
    trackingError: 3.2,
    informationRatio: 0.65,
    diversificationRatio: 0.78,
  };

  return (
    <LoadingState loading={riskLoading || portfolioLoading} error={riskError}>
      <div style={{ padding: '0 0 24px 0' }}>
        {/* Risk Alerts */}
        {riskAlerts.length > 0 && (
          <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
            <Col span={24}>
              <Card title={
                <Space>
                  <WarningOutlined style={{ color: '#faad14' }} />
                  风险预警
                </Space>
              } size="small">
                <Space direction="vertical" style={{ width: '100%' }}>
                  {riskAlerts.map((alert, index) => (
                    <Alert
                      key={index}
                      type={alert.type}
                      message={alert.title}
                      description={
                        <div>
                          <p>{alert.message}</p>
                          <p><strong>建议：</strong>{alert.recommendation}</p>
                        </div>
                      }
                      showIcon
                      closable
                    />
                  ))}
                </Space>
              </Card>
            </Col>
          </Row>
        )}

        {/* Controls */}
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={24} sm={8} md={6}>
            <Select
              value={confidenceLevel}
              onChange={setConfidenceLevel}
              style={{ width: '100%' }}
            >
              <Select.Option value={95}>95% 置信度</Select.Option>
              <Select.Option value={99}>99% 置信度</Select.Option>
            </Select>
          </Col>
          <Col xs={24} sm={8} md={6}>
            <Select
              value={riskHorizon}
              onChange={setRiskHorizon}
              style={{ width: '100%' }}
            >
              <Select.Option value={1}>1天</Select.Option>
              <Select.Option value={5}>5天</Select.Option>
              <Select.Option value={10}>10天</Select.Option>
            </Select>
          </Col>
          <Col xs={24} sm={8} md={12}>
            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button
                icon={<ReloadOutlined />}
                onClick={() => refetchRisk()}
                loading={riskLoading}
              >
                刷新
              </Button>
            </Space>
          </Col>
        </Row>

        {/* Portfolio Risk Metrics */}
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={12} sm={8} md={6}>
            <StatisticCard
              title={
                <Space>
                  组合VaR({confidenceLevel}%)
                  <Tooltip title="在给定置信度下，投资组合可能的最大损失">
                    <InfoCircleOutlined style={{ color: '#999' }} />
                  </Tooltip>
                </Space>
              }
              value={confidenceLevel === 95 ? portfolioRiskMetrics.portfolioVar95 : portfolioRiskMetrics.portfolioVar99}
              precision={2}
              suffix="%"
              valueStyle={{ color: '#cf1322' }}
            />
          </Col>
          <Col xs={12} sm={8} md={6}>
            <StatisticCard
              title={
                <Space>
                  预期损失
                  <Tooltip title="超过VaR时的平均损失">
                    <InfoCircleOutlined style={{ color: '#999' }} />
                  </Tooltip>
                </Space>
              }
              value={portfolioRiskMetrics.expectedShortfall}
              precision={2}
              suffix="%"
              valueStyle={{ color: '#ff4d4f' }}
            />
          </Col>
          <Col xs={12} sm={8} md={6}>
            <StatisticCard
              title="组合波动率"
              value={portfolioRiskMetrics.portfolioVolatility}
              precision={2}
              suffix="%"
              valueStyle={{ color: '#722ed1' }}
            />
          </Col>
          <Col xs={12} sm={8} md={6}>
            <StatisticCard
              title="跟踪误差"
              value={portfolioRiskMetrics.trackingError}
              precision={2}
              suffix="%"
              valueStyle={{ color: '#fa8c16' }}
            />
          </Col>
        </Row>

        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={12} sm={8} md={6}>
            <StatisticCard
              title="信息比率"
              value={portfolioRiskMetrics.informationRatio}
              precision={2}
              valueStyle={{ color: '#13c2c2' }}
            />
          </Col>
          <Col xs={12} sm={8} md={6}>
            <StatisticCard
              title="分散化比率"
              value={portfolioRiskMetrics.diversificationRatio}
              precision={2}
              valueStyle={{ color: '#52c41a' }}
            />
          </Col>
          <Col xs={12} sm={8} md={6}>
            <StatisticCard
              title="当前回撤"
              value={Math.abs(riskMetrics?.currentDrawdown || 0)}
              precision={2}
              suffix="%"
              valueStyle={{ color: '#cf1322' }}
            />
          </Col>
          <Col xs={12} sm={8} md={6}>
            <StatisticCard
              title="风险利用率"
              value={(riskMetrics?.riskUtilization.dailyLossRatio || 0) * 100}
              precision={1}
              suffix="%"
              valueStyle={{ 
                color: (riskMetrics?.riskUtilization.dailyLossRatio || 0) > 0.8 ? '#ff4d4f' : 
                       (riskMetrics?.riskUtilization.dailyLossRatio || 0) > 0.6 ? '#faad14' : '#52c41a'
              }}
            />
          </Col>
        </Row>

        {/* Risk Contribution Chart */}
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col span={24}>
            <BaseChart
              option={riskChartOption}
              loading={riskLoading}
              height={400}
              title="风险贡献分析"
            />
          </Col>
        </Row>

        {/* Risk Details Table */}
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <Card title="个股风险分析" size="small">
              <Table
                columns={riskColumns}
                dataSource={positionRisks}
                rowKey="symbol"
                pagination={{
                  pageSize: 10,
                  showSizeChanger: true,
                  showTotal: (total) => `共 ${total} 只股票`,
                }}
                size="small"
                loading={riskLoading}
              />
            </Card>
          </Col>
        </Row>

        {/* Correlation Analysis */}
        {correlationData.length > 0 && (
          <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
            <Col span={24}>
              <Card title="相关性分析" size="small">
                <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                  {correlationData.slice(0, 10).map((corr, index) => (
                    <div key={index} style={{ marginBottom: 8 }}>
                      <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                        <Space>
                          <span>{corr.symbol1} - {corr.symbol2}</span>
                          <Tag color={corr.riskLevel === 'high' ? 'red' : corr.riskLevel === 'medium' ? 'orange' : 'green'}>
                            {corr.riskLevel === 'high' ? '高相关' : corr.riskLevel === 'medium' ? '中相关' : '低相关'}
                          </Tag>
                        </Space>
                        <span style={{ 
                          color: Math.abs(corr.correlation) > 0.7 ? '#cf1322' : 
                                 Math.abs(corr.correlation) > 0.5 ? '#faad14' : '#52c41a' 
                        }}>
                          {corr.correlation.toFixed(3)}
                        </span>
                      </Space>
                      <Progress
                        percent={Math.abs(corr.correlation) * 100}
                        size="small"
                        showInfo={false}
                        strokeColor={Math.abs(corr.correlation) > 0.7 ? '#cf1322' : 
                                   Math.abs(corr.correlation) > 0.5 ? '#faad14' : '#52c41a'}
                      />
                    </div>
                  ))}
                </div>
              </Card>
            </Col>
          </Row>
        )}
      </div>
    </LoadingState>
  );
};

export default RiskAnalysis;