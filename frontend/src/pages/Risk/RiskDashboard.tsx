import React, { useState, useEffect, useMemo } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Alert,
  Spin,
  Button,
  Modal,
  Form,
  InputNumber,
  message,
  Space,
  Typography,
  Divider,
  Tag,
  Tooltip
} from 'antd';
import {
  ExclamationCircleOutlined,
  WarningOutlined,
  SettingOutlined,
  ReloadOutlined,
  RiseOutlined,
  ArrowDownOutlined
} from '@ant-design/icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import ReactECharts from 'echarts-for-react';
import { api } from '../../services/api';
import { RiskMetrics, RiskLimits } from '../../services/types';
import { useRiskAlerts } from '../../hooks/useRealTime';

const { Text, Title } = Typography;

interface RiskDashboardProps {
  refreshInterval?: number;
}

const RiskDashboard: React.FC<RiskDashboardProps> = ({ 
  refreshInterval = 5000 
}) => {
  const [isLimitsModalVisible, setIsLimitsModalVisible] = useState(false);
  const [form] = Form.useForm();
  const queryClient = useQueryClient();

  // Fetch risk metrics
  const { 
    data: riskMetrics, 
    isLoading: isLoadingMetrics, 
    error: metricsError,
    refetch: refetchMetrics
  } = useQuery({
    queryKey: ['riskMetrics'],
    queryFn: () => api.risk.getMetrics(),
    refetchInterval: refreshInterval,
    refetchOnWindowFocus: true,
  });

  // Fetch risk configuration
  const { 
    data: riskConfig, 
    isLoading: isLoadingConfig 
  } = useQuery({
    queryKey: ['riskConfig'],
    queryFn: () => api.risk.getConfig(),
  });

  // Real-time risk alerts
  const { alerts, latestAlert } = useRiskAlerts({
    autoSubscribe: true,
    maxAlertsSize: 10
  });

  // Update risk limits mutation
  const updateLimitsMutation = useMutation({
    mutationFn: (limits: Partial<RiskLimits>) => api.risk.updateLimits(limits),
    onSuccess: () => {
      message.success('风险限制更新成功');
      setIsLimitsModalVisible(false);
      queryClient.invalidateQueries({ queryKey: ['riskMetrics'] });
      queryClient.invalidateQueries({ queryKey: ['riskConfig'] });
    },
    onError: (error: any) => {
      message.error(`更新失败: ${error.message}`);
    },
  });

  // Risk trend chart data
  const riskTrendData = useMemo(() => {
    if (!riskMetrics) return [];
    
    // Generate mock historical data for demonstration
    // In real implementation, this would come from API
    const now = new Date();
    const data = [];
    for (let i = 23; i >= 0; i--) {
      const time = new Date(now.getTime() - i * 60 * 60 * 1000);
      data.push({
        time: time.toISOString(),
        dailyPnl: riskMetrics.dailyPnl + (Math.random() - 0.5) * 1000,
        drawdown: riskMetrics.currentDrawdown + (Math.random() - 0.5) * 2,
        riskUtilization: riskMetrics.riskUtilization.dailyLossRatio + (Math.random() - 0.5) * 0.1
      });
    }
    return data;
  }, [riskMetrics]);

  const chartOption = useMemo(() => {
    if (riskTrendData.length === 0) return {};

    return {
      title: {
        text: '风险指标趋势',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'normal'
        }
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross'
        }
      },
      legend: {
        data: ['日盈亏', '回撤率', '风险利用率'],
        bottom: 10
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '15%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: riskTrendData.map(item => 
          new Date(item.time).toLocaleTimeString('zh-CN', { 
            hour: '2-digit', 
            minute: '2-digit' 
          })
        ),
        axisLabel: {
          rotate: 45
        }
      },
      yAxis: [
        {
          type: 'value',
          name: '金额 (¥)',
          position: 'left',
          axisLabel: {
            formatter: '{value}'
          }
        },
        {
          type: 'value',
          name: '百分比 (%)',
          position: 'right',
          axisLabel: {
            formatter: '{value}%'
          }
        }
      ],
      series: [
        {
          name: '日盈亏',
          type: 'line',
          yAxisIndex: 0,
          data: riskTrendData.map(item => item.dailyPnl.toFixed(2)),
          itemStyle: {
            color: '#1890ff'
          },
          areaStyle: {
            opacity: 0.3
          }
        },
        {
          name: '回撤率',
          type: 'line',
          yAxisIndex: 1,
          data: riskTrendData.map(item => item.drawdown.toFixed(2)),
          itemStyle: {
            color: '#ff4d4f'
          }
        },
        {
          name: '风险利用率',
          type: 'line',
          yAxisIndex: 1,
          data: riskTrendData.map(item => (item.riskUtilization * 100).toFixed(1)),
          itemStyle: {
            color: '#faad14'
          }
        }
      ]
    };
  }, [riskTrendData]);

  const handleUpdateLimits = async (values: any) => {
    const limits: Partial<RiskLimits> = {
      maxPositionSize: values.maxPositionSize / 100, // Convert percentage to decimal
      maxDrawdownLimit: values.maxDrawdownLimit / 100,
      maxDailyLoss: values.maxDailyLoss
    };
    
    updateLimitsMutation.mutate(limits);
  };

  const openLimitsModal = () => {
    if (riskMetrics?.riskLimits) {
      form.setFieldsValue({
        maxPositionSize: riskMetrics.riskLimits.maxPositionSize * 100,
        maxDrawdownLimit: riskMetrics.riskLimits.maxDrawdownLimit * 100,
        maxDailyLoss: riskMetrics.riskLimits.maxDailyLoss
      });
    }
    setIsLimitsModalVisible(true);
  };

  const getRiskLevel = (utilization: number): { level: string; color: string } => {
    if (utilization >= 0.9) return { level: '极高', color: '#ff4d4f' };
    if (utilization >= 0.7) return { level: '高', color: '#ff7a45' };
    if (utilization >= 0.5) return { level: '中', color: '#faad14' };
    return { level: '低', color: '#52c41a' };
  };

  if (isLoadingMetrics || isLoadingConfig) {
    return (
      <Card title="风险指标仪表板">
        <div style={{ textAlign: 'center', padding: '50px' }}>
          <Spin size="large" />
        </div>
      </Card>
    );
  }

  if (metricsError) {
    return (
      <Card title="风险指标仪表板">
        <Alert
          message="数据加载失败"
          description={metricsError.message}
          type="error"
          showIcon
          action={
            <Button size="small" onClick={() => refetchMetrics()}>
              重试
            </Button>
          }
        />
      </Card>
    );
  }

  if (!riskMetrics) {
    return (
      <Card title="风险指标仪表板">
        <Alert
          message="暂无风险数据"
          description="系统尚未获取到风险指标数据"
          type="info"
          showIcon
        />
      </Card>
    );
  }

  const dailyLossRisk = getRiskLevel(riskMetrics.riskUtilization.dailyLossRatio);
  const drawdownRisk = getRiskLevel(riskMetrics.riskUtilization.drawdownRatio);

  return (
    <>
      <Card
        title="风险指标仪表板"
        extra={
          <Space>
            <Button
              icon={<ReloadOutlined />}
              onClick={() => refetchMetrics()}
              size="small"
            >
              刷新
            </Button>
            <Button
              icon={<SettingOutlined />}
              onClick={openLimitsModal}
              size="small"
            >
              设置限制
            </Button>
          </Space>
        }
      >
        {/* Latest Alert Banner */}
        {latestAlert && (
          <Alert
            message={`最新风险告警: ${latestAlert.message}`}
            type={latestAlert.level === 'critical' ? 'error' : 
                  latestAlert.level === 'error' ? 'warning' : 'info'}
            showIcon
            closable
            style={{ marginBottom: '16px' }}
          />
        )}

        {/* Risk Metrics Overview */}
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col xs={24} sm={12} md={6}>
            <Card size="small">
              <Statistic
                title="日盈亏"
                value={riskMetrics.dailyPnl}
                precision={2}
                prefix="¥"
                valueStyle={{ 
                  color: riskMetrics.dailyPnl >= 0 ? '#3f8600' : '#cf1322',
                  fontSize: '20px'
                }}
                suffix={
                  riskMetrics.dailyPnl >= 0 ? 
                    <RiseOutlined style={{ color: '#3f8600' }} /> :
                    <ArrowDownOutlined style={{ color: '#cf1322' }} />
                }
              />
            </Card>
          </Col>
          
          <Col xs={24} sm={12} md={6}>
            <Card size="small">
              <Statistic
                title="当前回撤"
                value={riskMetrics.currentDrawdown}
                precision={2}
                suffix="%"
                valueStyle={{ 
                  color: '#cf1322',
                  fontSize: '20px'
                }}
              />
            </Card>
          </Col>
          
          <Col xs={24} sm={12} md={6}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <Text type="secondary" style={{ fontSize: '14px' }}>日损失风险</Text>
                <div style={{ marginTop: '8px' }}>
                  <Tag color={dailyLossRisk.color} style={{ fontSize: '16px', padding: '4px 12px' }}>
                    {dailyLossRisk.level}
                  </Tag>
                </div>
                <Progress
                  percent={riskMetrics.riskUtilization.dailyLossRatio * 100}
                  strokeColor={dailyLossRisk.color}
                  size="small"
                  style={{ marginTop: '8px' }}
                />
              </div>
            </Card>
          </Col>
          
          <Col xs={24} sm={12} md={6}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <Text type="secondary" style={{ fontSize: '14px' }}>回撤风险</Text>
                <div style={{ marginTop: '8px' }}>
                  <Tag color={drawdownRisk.color} style={{ fontSize: '16px', padding: '4px 12px' }}>
                    {drawdownRisk.level}
                  </Tag>
                </div>
                <Progress
                  percent={riskMetrics.riskUtilization.drawdownRatio * 100}
                  strokeColor={drawdownRisk.color}
                  size="small"
                  style={{ marginTop: '8px' }}
                />
              </div>
            </Card>
          </Col>
        </Row>

        {/* Risk Limits and Utilization */}
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col xs={24} lg={12}>
            <Card title="风险限制" size="small">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Row justify="space-between">
                  <Text>最大仓位比例:</Text>
                  <Text strong>{(riskMetrics.riskLimits.maxPositionSize * 100).toFixed(1)}%</Text>
                </Row>
                <Row justify="space-between">
                  <Text>最大回撤限制:</Text>
                  <Text strong>{(riskMetrics.riskLimits.maxDrawdownLimit * 100).toFixed(1)}%</Text>
                </Row>
                <Row justify="space-between">
                  <Text>最大日损失:</Text>
                  <Text strong>{((riskMetrics.riskLimits.maxDailyLoss || 0) * 100).toFixed(1)}%</Text>
                </Row>
              </Space>
            </Card>
          </Col>
          
          <Col xs={24} lg={12}>
            <Card title="风险利用率" size="small">
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Row justify="space-between" style={{ marginBottom: '8px' }}>
                    <Text>日损失利用率:</Text>
                    <Text strong style={{ color: dailyLossRisk.color }}>
                      {(riskMetrics.riskUtilization.dailyLossRatio * 100).toFixed(1)}%
                    </Text>
                  </Row>
                  <Progress
                    percent={riskMetrics.riskUtilization.dailyLossRatio * 100}
                    strokeColor={dailyLossRisk.color}
                    size="small"
                  />
                </div>
                
                <div>
                  <Row justify="space-between" style={{ marginBottom: '8px' }}>
                    <Text>回撤利用率:</Text>
                    <Text strong style={{ color: drawdownRisk.color }}>
                      {(riskMetrics.riskUtilization.drawdownRatio * 100).toFixed(1)}%
                    </Text>
                  </Row>
                  <Progress
                    percent={riskMetrics.riskUtilization.drawdownRatio * 100}
                    strokeColor={drawdownRisk.color}
                    size="small"
                  />
                </div>
              </Space>
            </Card>
          </Col>
        </Row>

        {/* Risk Trend Chart */}
        <Card title="风险趋势分析" size="small">
          <ReactECharts
            option={chartOption}
            style={{ height: '300px' }}
            notMerge={true}
            lazyUpdate={true}
          />
        </Card>
      </Card>

      {/* Risk Limits Configuration Modal */}
      <Modal
        title="风险限制设置"
        open={isLimitsModalVisible}
        onCancel={() => setIsLimitsModalVisible(false)}
        onOk={() => form.submit()}
        confirmLoading={updateLimitsMutation.isPending}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleUpdateLimits}
        >
          <Form.Item
            name="maxPositionSize"
            label="最大仓位比例 (%)"
            rules={[
              { required: true, message: '请输入最大仓位比例' },
              { type: 'number', min: 1, max: 100, message: '仓位比例必须在1-100%之间' }
            ]}
          >
            <InputNumber
              style={{ width: '100%' }}
              min={1}
              max={100}
              step={1}
              formatter={value => `${value}%`}
              parser={value => parseFloat(value!.replace('%', '')) || 0}
            />
          </Form.Item>

          <Form.Item
            name="maxDrawdownLimit"
            label="最大回撤限制 (%)"
            rules={[
              { required: true, message: '请输入最大回撤限制' },
              { type: 'number', min: 1, max: 50, message: '回撤限制必须在1-50%之间' }
            ]}
          >
            <InputNumber
              style={{ width: '100%' }}
              min={1}
              max={50}
              step={0.1}
              formatter={value => `${value}%`}
              parser={value => parseFloat(value!.replace('%', '')) || 0}
            />
          </Form.Item>

          <Form.Item
            name="maxDailyLoss"
            label="最大日损失 (¥)"
            rules={[
              { required: true, message: '请输入最大日损失' },
              { type: 'number', min: 100, message: '最大日损失不能小于100元' }
            ]}
          >
            <InputNumber
              style={{ width: '100%' }}
              min={100}
              step={100}
              formatter={value => `¥ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
              parser={value => parseFloat(value!.replace(/¥\s?|(,*)/g, '')) || 0}
            />
          </Form.Item>
        </Form>
      </Modal>
    </>
  );
};

export default RiskDashboard;