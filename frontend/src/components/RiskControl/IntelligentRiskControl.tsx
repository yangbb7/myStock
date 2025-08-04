import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Tabs,
  Badge,
  List,
  Tag,
  Progress,
  Button,
  Space,
  Statistic,
  Alert,
  Switch,
  InputNumber,
  Form,
  Select,
  Divider,
  Timeline,
  Tooltip,
  Modal,
  Table,
  message,
} from 'antd';
import {
  SafetyOutlined,
  AlertOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  StopOutlined,
  RiseOutlined,
  FallOutlined,
  SettingOutlined,
  BellOutlined,
  DashboardOutlined,
} from '@ant-design/icons';
import { Line, Pie } from '@ant-design/charts';
import { riskApi } from '@/api/risk';

const { TabPane } = Tabs;
const { Option } = Select;

interface RiskAlert {
  alert_id: string;
  timestamp: string;
  level: 'low' | 'medium' | 'high' | 'critical';
  type: string;
  message: string;
  position_id?: string;
  suggested_action?: string;
  metadata?: any;
}

interface RiskConfig {
  max_position_size: number;
  max_position_percent: number;
  max_total_exposure: number;
  max_sector_exposure: number;
  stop_loss_enabled: boolean;
  stop_loss_percent: number;
  trailing_stop_enabled: boolean;
  trailing_stop_percent: number;
  take_profit_enabled: boolean;
  take_profit_levels: Array<{ percent: number; close_ratio: number }>;
}

interface RiskMetrics {
  total_positions: number;
  risk_level: string;
  active_alerts: number;
  max_drawdown: number;
  value_at_risk: number;
  portfolio_volatility: number;
}

const IntelligentRiskControl: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [riskAlerts, setRiskAlerts] = useState<RiskAlert[]>([]);
  const [riskConfig, setRiskConfig] = useState<RiskConfig>({
    max_position_size: 100000,
    max_position_percent: 0.1,
    max_total_exposure: 0.95,
    max_sector_exposure: 0.3,
    stop_loss_enabled: true,
    stop_loss_percent: 0.02,
    trailing_stop_enabled: true,
    trailing_stop_percent: 0.03,
    take_profit_enabled: true,
    take_profit_levels: [
      { percent: 0.05, close_ratio: 0.3 },
      { percent: 0.10, close_ratio: 0.3 },
      { percent: 0.15, close_ratio: 0.4 },
    ],
  });
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics>({
    total_positions: 5,
    risk_level: 'medium',
    active_alerts: 3,
    max_drawdown: -0.08,
    value_at_risk: -15000,
    portfolio_volatility: 0.18,
  });
  const [autoExecute, setAutoExecute] = useState(false);
  const [loading, setLoading] = useState(false);
  const [form] = Form.useForm();

  // 模拟实时风险监控
  useEffect(() => {
    const interval = setInterval(() => {
      // 模拟新的风险警报
      if (Math.random() > 0.7) {
        const newAlert: RiskAlert = {
          alert_id: `alert_${Date.now()}`,
          timestamp: new Date().toISOString(),
          level: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)] as any,
          type: ['stop_loss', 'concentration', 'volatility'][Math.floor(Math.random() * 3)],
          message: '模拟风险警报',
          suggested_action: 'review_position',
        };
        setRiskAlerts(prev => [newAlert, ...prev].slice(0, 10));
      }
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  // 风险等级颜色
  const getRiskLevelColor = (level: string) => {
    const colors = {
      low: '#52c41a',
      medium: '#faad14',
      high: '#ff7a45',
      critical: '#f5222d',
    };
    return colors[level as keyof typeof colors] || '#1890ff';
  };

  // 渲染风险概览
  const renderRiskOverview = () => {
    const pieData = [
      { type: '低风险', value: 60 },
      { type: '中风险', value: 30 },
      { type: '高风险', value: 10 },
    ];

    const pieConfig = {
      data: pieData,
      angleField: 'value',
      colorField: 'type',
      radius: 0.8,
      label: {
        type: 'outer',
        content: '{name} {percentage}',
      },
      interactions: [{ type: 'element-active' }],
    };

    return (
      <Row gutter={[16, 16]}>
        <Col span={6}>
          <Card>
            <Statistic
              title="风险等级"
              value={riskMetrics.risk_level.toUpperCase()}
              valueStyle={{ color: getRiskLevelColor(riskMetrics.risk_level) }}
              prefix={<DashboardOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃警报"
              value={riskMetrics.active_alerts}
              valueStyle={{ color: riskMetrics.active_alerts > 5 ? '#f5222d' : '#1890ff' }}
              prefix={<AlertOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="最大回撤"
              value={riskMetrics.max_drawdown * 100}
              precision={2}
              suffix="%"
              valueStyle={{ color: '#f5222d' }}
              prefix={<FallOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="风险价值(VaR)"
              value={Math.abs(riskMetrics.value_at_risk)}
              prefix="¥"
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        
        <Col span={12}>
          <Card title="持仓风险分布">
            <Pie {...pieConfig} height={200} />
          </Card>
        </Col>
        
        <Col span={12}>
          <Card title="风险指标趋势">
            <Line
              data={[
                { time: '09:30', value: 0.15, type: '波动率' },
                { time: '10:00', value: 0.18, type: '波动率' },
                { time: '10:30', value: 0.16, type: '波动率' },
                { time: '11:00', value: 0.20, type: '波动率' },
                { time: '11:30', value: 0.19, type: '波动率' },
                { time: '09:30', value: -0.05, type: '回撤' },
                { time: '10:00', value: -0.06, type: '回撤' },
                { time: '10:30', value: -0.04, type: '回撤' },
                { time: '11:00', value: -0.08, type: '回撤' },
                { time: '11:30', value: -0.07, type: '回撤' },
              ]}
              xField="time"
              yField="value"
              seriesField="type"
              height={200}
            />
          </Card>
        </Col>
      </Row>
    );
  };

  // 渲染风险警报
  const renderRiskAlerts = () => {
    const columns = [
      {
        title: '时间',
        dataIndex: 'timestamp',
        key: 'timestamp',
        width: 150,
        render: (timestamp: string) => new Date(timestamp).toLocaleTimeString(),
      },
      {
        title: '级别',
        dataIndex: 'level',
        key: 'level',
        width: 100,
        render: (level: string) => (
          <Tag color={getRiskLevelColor(level)}>{level.toUpperCase()}</Tag>
        ),
      },
      {
        title: '类型',
        dataIndex: 'type',
        key: 'type',
        width: 120,
      },
      {
        title: '警报信息',
        dataIndex: 'message',
        key: 'message',
      },
      {
        title: '建议操作',
        dataIndex: 'suggested_action',
        key: 'suggested_action',
        width: 120,
        render: (action: string) => {
          const actionMap: any = {
            close_position: { text: '平仓', color: 'red' },
            reduce_position: { text: '减仓', color: 'orange' },
            review_position: { text: '审查', color: 'blue' },
          };
          const config = actionMap[action] || { text: action, color: 'default' };
          return <Tag color={config.color}>{config.text}</Tag>;
        },
      },
      {
        title: '操作',
        key: 'action',
        width: 150,
        render: (_: any, record: RiskAlert) => (
          <Space>
            <Button size="small" type="primary" danger>
              执行
            </Button>
            <Button size="small">忽略</Button>
          </Space>
        ),
      },
    ];

    return (
      <Card
        title="实时风险警报"
        extra={
          <Space>
            <span>自动执行：</span>
            <Switch
              checked={autoExecute}
              onChange={setAutoExecute}
              checkedChildren="开启"
              unCheckedChildren="关闭"
            />
            <Button icon={<BellOutlined />}>警报设置</Button>
          </Space>
        }
      >
        <Table
          columns={columns}
          dataSource={riskAlerts}
          rowKey="alert_id"
          size="small"
          pagination={{ pageSize: 10 }}
        />
      </Card>
    );
  };

  // 渲染止损止盈设置
  const renderStopLossSettings = () => {
    return (
      <Card title="止损止盈策略">
        <Form
          form={form}
          layout="vertical"
          initialValues={riskConfig}
          onFinish={(values) => {
            setRiskConfig(values);
            message.success('风控参数已更新');
          }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Card title="止损设置" type="inner">
                <Form.Item label="启用固定止损" name="stop_loss_enabled" valuePropName="checked">
                  <Switch checkedChildren="开启" unCheckedChildren="关闭" />
                </Form.Item>
                
                <Form.Item
                  label="止损比例"
                  name="stop_loss_percent"
                  rules={[{ required: true }]}
                >
                  <InputNumber
                    min={0.01}
                    max={0.1}
                    step={0.01}
                    style={{ width: '100%' }}
                    formatter={value => `${(Number(value) * 100).toFixed(0)}%`}
                    parser={value => Number(value?.replace('%', '')) / 100}
                  />
                </Form.Item>
                
                <Form.Item label="启用移动止损" name="trailing_stop_enabled" valuePropName="checked">
                  <Switch checkedChildren="开启" unCheckedChildren="关闭" />
                </Form.Item>
                
                <Form.Item
                  label="移动止损比例"
                  name="trailing_stop_percent"
                  rules={[{ required: true }]}
                >
                  <InputNumber
                    min={0.01}
                    max={0.1}
                    step={0.01}
                    style={{ width: '100%' }}
                    formatter={value => `${(Number(value) * 100).toFixed(0)}%`}
                    parser={value => Number(value?.replace('%', '')) / 100}
                  />
                </Form.Item>
              </Card>
            </Col>
            
            <Col span={12}>
              <Card title="止盈设置" type="inner">
                <Form.Item label="启用分批止盈" name="take_profit_enabled" valuePropName="checked">
                  <Switch checkedChildren="开启" unCheckedChildren="关闭" />
                </Form.Item>
                
                <Timeline>
                  <Timeline.Item color="green">
                    <Space>
                      <span>盈利 5% 时</span>
                      <Tag color="green">平仓 30%</Tag>
                    </Space>
                  </Timeline.Item>
                  <Timeline.Item color="green">
                    <Space>
                      <span>盈利 10% 时</span>
                      <Tag color="green">平仓 30%</Tag>
                    </Space>
                  </Timeline.Item>
                  <Timeline.Item color="green">
                    <Space>
                      <span>盈利 15% 时</span>
                      <Tag color="green">平仓 40%</Tag>
                    </Space>
                  </Timeline.Item>
                </Timeline>
                
                <Alert
                  message="智能止盈策略"
                  description="系统会根据市场状态和持仓情况自动调整止盈点位"
                  type="info"
                  showIcon
                />
              </Card>
            </Col>
          </Row>
          
          <Divider />
          
          <Row gutter={16}>
            <Col span={24}>
              <Card title="仓位管理" type="inner">
                <Row gutter={16}>
                  <Col span={6}>
                    <Form.Item label="单仓位最大金额" name="max_position_size">
                      <InputNumber
                        min={10000}
                        max={1000000}
                        step={10000}
                        style={{ width: '100%' }}
                        formatter={value => `¥ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                        parser={value => value?.replace(/\¥\s?|(,*)/g, '') as any}
                      />
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item label="单仓位最大比例" name="max_position_percent">
                      <InputNumber
                        min={0.05}
                        max={0.3}
                        step={0.05}
                        style={{ width: '100%' }}
                        formatter={value => `${(Number(value) * 100).toFixed(0)}%`}
                        parser={value => Number(value?.replace('%', '')) / 100}
                      />
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item label="总仓位上限" name="max_total_exposure">
                      <InputNumber
                        min={0.5}
                        max={1.0}
                        step={0.05}
                        style={{ width: '100%' }}
                        formatter={value => `${(Number(value) * 100).toFixed(0)}%`}
                        parser={value => Number(value?.replace('%', '')) / 100}
                      />
                    </Form.Item>
                  </Col>
                  <Col span={6}>
                    <Form.Item label="板块集中度上限" name="max_sector_exposure">
                      <InputNumber
                        min={0.2}
                        max={0.5}
                        step={0.05}
                        style={{ width: '100%' }}
                        formatter={value => `${(Number(value) * 100).toFixed(0)}%`}
                        parser={value => Number(value?.replace('%', '')) / 100}
                      />
                    </Form.Item>
                  </Col>
                </Row>
              </Card>
            </Col>
          </Row>
          
          <Form.Item style={{ marginTop: 16 }}>
            <Space>
              <Button type="primary" htmlType="submit">
                保存设置
              </Button>
              <Button onClick={() => form.resetFields()}>
                重置
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Card>
    );
  };

  // 渲染风险规则
  const renderRiskRules = () => {
    const rules = [
      {
        id: 'rule_1',
        name: '仓位集中度检查',
        condition: 'position_percent > 15%',
        action: '发出警报',
        status: 'active',
        priority: 'high',
      },
      {
        id: 'rule_2',
        name: '亏损扩大检查',
        condition: 'loss > 10% AND trend = down',
        action: '自动减仓50%',
        status: 'active',
        priority: 'critical',
      },
      {
        id: 'rule_3',
        name: '波动率激增检查',
        condition: 'volatility > 2 * avg_volatility',
        action: '收紧止损',
        status: 'active',
        priority: 'high',
      },
      {
        id: 'rule_4',
        name: '相关性检查',
        condition: 'correlation > 0.8',
        action: '提示分散投资',
        status: 'active',
        priority: 'medium',
      },
    ];

    return (
      <Card
        title="风险控制规则"
        extra={
          <Button type="primary" icon={<SettingOutlined />}>
            添加规则
          </Button>
        }
      >
        <List
          dataSource={rules}
          renderItem={rule => (
            <List.Item
              actions={[
                <Switch
                  checked={rule.status === 'active'}
                  checkedChildren="启用"
                  unCheckedChildren="停用"
                />,
                <Button size="small">编辑</Button>,
                <Button size="small" danger>
                  删除
                </Button>,
              ]}
            >
              <List.Item.Meta
                avatar={
                  <Badge
                    status={
                      rule.priority === 'critical'
                        ? 'error'
                        : rule.priority === 'high'
                        ? 'warning'
                        : 'processing'
                    }
                  />
                }
                title={rule.name}
                description={
                  <Space direction="vertical" size="small">
                    <div>
                      <Text type="secondary">触发条件：</Text>
                      <Tag>{rule.condition}</Tag>
                    </div>
                    <div>
                      <Text type="secondary">执行动作：</Text>
                      <Tag color="blue">{rule.action}</Tag>
                    </div>
                  </Space>
                }
              />
            </List.Item>
          )}
        />
      </Card>
    );
  };

  return (
    <div className="intelligent-risk-control">
      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane
          tab={
            <span>
              <DashboardOutlined />
              风险概览
            </span>
          }
          key="overview"
        >
          {renderRiskOverview()}
        </TabPane>
        
        <TabPane
          tab={
            <span>
              <AlertOutlined />
              风险警报
              <Badge count={riskAlerts.filter(a => a.level === 'high').length} offset={[10, 0]} />
            </span>
          }
          key="alerts"
        >
          {renderRiskAlerts()}
        </TabPane>
        
        <TabPane
          tab={
            <span>
              <StopOutlined />
              止损止盈
            </span>
          }
          key="stop"
        >
          {renderStopLossSettings()}
        </TabPane>
        
        <TabPane
          tab={
            <span>
              <SafetyOutlined />
              风控规则
            </span>
          }
          key="rules"
        >
          {renderRiskRules()}
        </TabPane>
      </Tabs>
    </div>
  );
};

export default IntelligentRiskControl;