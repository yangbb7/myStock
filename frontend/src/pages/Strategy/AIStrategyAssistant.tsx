import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Space,
  Tag,
  Progress,
  List,
  Statistic,
  Alert,
  Spin,
  Collapse,
  Timeline,
  Badge,
  Tooltip,
  Select,
  Radio,
  Divider,
  Typography,
} from 'antd';
import {
  BulbOutlined,
  RobotOutlined,
  LineChartOutlined,
  SafetyOutlined,
  ThunderboltOutlined,
  QuestionCircleOutlined,
  SyncOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  RiseOutlined,
  FallOutlined,
} from '@ant-design/icons';
import { strategyApi } from '@/api/strategy';
import { marketApi } from '@/api/market';

const { Panel } = Collapse;
const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

interface MarketCondition {
  trend: string;
  volatility: string;
  volume_trend: string;
  confidence: number;
}

interface StrategyRecommendation {
  strategy_type: string;
  reason: string;
  expected_performance: {
    expected_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
  };
  risk_level: string;
  suitable_symbols: string[];
  parameters: Record<string, any>;
  confidence: number;
}

interface OptimizationSuggestion {
  parameter: string;
  current_value: any;
  suggested_value: any;
  expected_improvement: number;
  reason: string;
}

const AIStrategyAssistant: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [marketCondition, setMarketCondition] = useState<MarketCondition | null>(null);
  const [recommendations, setRecommendations] = useState<StrategyRecommendation[]>([]);
  const [optimizationSuggestions, setOptimizationSuggestions] = useState<OptimizationSuggestion[]>([]);
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>(['000001.SZ', '000002.SZ']);
  const [riskPreference, setRiskPreference] = useState<string>('medium');
  const [activeTab, setActiveTab] = useState<'recommend' | 'optimize' | 'analysis'>('recommend');

  // 获取市场分析和策略推荐
  const fetchRecommendations = async () => {
    setLoading(true);
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // 设置模拟数据
      setMarketCondition({
        trend: 'bullish',
        volatility: 'medium',
        volume_trend: 'increasing',
        confidence: 0.85,
      });

      setRecommendations([
        {
          strategy_type: 'trend_following',
          reason: '市场处于上升趋势，趋势跟踪策略能够捕捉主要行情',
          expected_performance: {
            expected_return: 0.15,
            sharpe_ratio: 1.2,
            max_drawdown: -0.08,
            win_rate: 0.55,
          },
          risk_level: 'medium',
          suitable_symbols: ['000001.SZ', '000002.SZ'],
          parameters: {
            fast_period: 10,
            slow_period: 30,
            stop_loss: 0.02,
          },
          confidence: 0.8,
        },
        {
          strategy_type: 'momentum',
          reason: '市场动能强劲，动量策略能够跟随强势股',
          expected_performance: {
            expected_return: 0.18,
            sharpe_ratio: 1.1,
            max_drawdown: -0.10,
            win_rate: 0.52,
          },
          risk_level: 'medium',
          suitable_symbols: ['000001.SZ'],
          parameters: {
            momentum_period: 10,
            volume_threshold: 1.5,
          },
          confidence: 0.75,
        },
      ]);

      setOptimizationSuggestions([
        {
          parameter: 'fast_period',
          current_value: 20,
          suggested_value: 10,
          expected_improvement: 0.1,
          reason: '历史数据显示10日均线响应更及时',
        },
        {
          parameter: 'stop_loss',
          current_value: 0.05,
          suggested_value: 0.03,
          expected_improvement: 0.05,
          reason: '当前市场波动适中，可以收紧止损以保护利润',
        },
      ]);
    } catch (error) {
      console.error('Failed to fetch recommendations:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecommendations();
  }, [selectedSymbols, riskPreference]);

  // 渲染市场状态卡片
  const renderMarketConditionCard = () => {
    if (!marketCondition) return null;

    const trendIcon = marketCondition.trend === 'bullish' ? <RiseOutlined /> : 
                     marketCondition.trend === 'bearish' ? <FallOutlined /> : 
                     <LineChartOutlined />;
    
    const trendColor = marketCondition.trend === 'bullish' ? '#52c41a' : 
                      marketCondition.trend === 'bearish' ? '#f5222d' : 
                      '#faad14';

    return (
      <Card title="当前市场状态" extra={<SyncOutlined spin={loading} />}>
        <Row gutter={[16, 16]}>
          <Col span={6}>
            <Statistic
              title="市场趋势"
              value={marketCondition.trend === 'bullish' ? '上涨' : 
                     marketCondition.trend === 'bearish' ? '下跌' : '横盘'}
              prefix={trendIcon}
              valueStyle={{ color: trendColor }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="波动率"
              value={marketCondition.volatility === 'high' ? '高' :
                     marketCondition.volatility === 'low' ? '低' : '中等'}
              prefix={<ThunderboltOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="成交量趋势"
              value={marketCondition.volume_trend === 'increasing' ? '增加' :
                     marketCondition.volume_trend === 'decreasing' ? '减少' : '稳定'}
            />
          </Col>
          <Col span={6}>
            <Progress
              type="circle"
              percent={marketCondition.confidence * 100}
              format={percent => `${percent}%`}
              strokeColor="#1890ff"
              width={80}
            />
            <div style={{ textAlign: 'center', marginTop: 8 }}>置信度</div>
          </Col>
        </Row>
      </Card>
    );
  };

  // 渲染策略推荐
  const renderStrategyRecommendations = () => {
    return (
      <Card title="AI策略推荐" loading={loading}>
        <List
          dataSource={recommendations}
          renderItem={(item, index) => (
            <List.Item
              actions={[
                <Button type="primary" size="small">使用策略</Button>,
                <Button size="small">查看详情</Button>,
              ]}
            >
              <List.Item.Meta
                avatar={
                  <div style={{ 
                    width: 60, 
                    height: 60, 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center',
                    background: '#f0f2f5',
                    borderRadius: 8,
                    fontSize: 24,
                  }}>
                    {index + 1}
                  </div>
                }
                title={
                  <Space>
                    <span>{item.strategy_type === 'trend_following' ? '趋势跟踪' : '动量策略'}</span>
                    <Tag color={item.risk_level === 'high' ? 'red' : 
                               item.risk_level === 'low' ? 'green' : 'orange'}>
                      {item.risk_level === 'high' ? '高风险' : 
                       item.risk_level === 'low' ? '低风险' : '中等风险'}
                    </Tag>
                    <Badge count={`${(item.confidence * 100).toFixed(0)}%`} 
                           style={{ backgroundColor: '#52c41a' }} />
                  </Space>
                }
                description={
                  <div>
                    <Paragraph style={{ marginBottom: 8 }}>{item.reason}</Paragraph>
                    <Row gutter={16}>
                      <Col span={6}>
                        <Text type="secondary">预期收益：</Text>
                        <Text strong style={{ color: '#52c41a' }}>
                          {(item.expected_performance.expected_return * 100).toFixed(1)}%
                        </Text>
                      </Col>
                      <Col span={6}>
                        <Text type="secondary">夏普比率：</Text>
                        <Text strong>{item.expected_performance.sharpe_ratio.toFixed(2)}</Text>
                      </Col>
                      <Col span={6}>
                        <Text type="secondary">最大回撤：</Text>
                        <Text strong style={{ color: '#f5222d' }}>
                          {(item.expected_performance.max_drawdown * 100).toFixed(1)}%
                        </Text>
                      </Col>
                      <Col span={6}>
                        <Text type="secondary">胜率：</Text>
                        <Text strong>{(item.expected_performance.win_rate * 100).toFixed(0)}%</Text>
                      </Col>
                    </Row>
                  </div>
                }
              />
            </List.Item>
          )}
        />
      </Card>
    );
  };

  // 渲染优化建议
  const renderOptimizationSuggestions = () => {
    return (
      <Card title="参数优化建议" loading={loading}>
        <Timeline>
          {optimizationSuggestions.map((suggestion, index) => (
            <Timeline.Item
              key={index}
              color="green"
              dot={<CheckCircleOutlined />}
            >
              <Card size="small">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Row justify="space-between">
                    <Col>
                      <Text strong>{suggestion.parameter}</Text>
                    </Col>
                    <Col>
                      <Tag color="blue">
                        预期提升 {(suggestion.expected_improvement * 100).toFixed(0)}%
                      </Tag>
                    </Col>
                  </Row>
                  <Row>
                    <Col span={8}>
                      <Text type="secondary">当前值：</Text>
                      <Text code>{suggestion.current_value}</Text>
                    </Col>
                    <Col span={8}>
                      <Text type="secondary">建议值：</Text>
                      <Text code style={{ color: '#52c41a' }}>
                        {suggestion.suggested_value}
                      </Text>
                    </Col>
                  </Row>
                  <Alert
                    message={suggestion.reason}
                    type="info"
                    showIcon
                    icon={<BulbOutlined />}
                  />
                </Space>
              </Card>
            </Timeline.Item>
          ))}
        </Timeline>
      </Card>
    );
  };

  // 渲染风险控制建议
  const renderRiskControlSuggestions = () => {
    return (
      <Card title="智能风控建议">
        <Collapse defaultActiveKey={['1']}>
          <Panel header="仓位管理" key="1" extra={<SafetyOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Alert
                message="建议单一持仓不超过总资金的10%"
                description="根据您的中等风险偏好，建议采用分散投资策略"
                type="warning"
                showIcon
              />
              <Row gutter={16}>
                <Col span={8}>
                  <Statistic title="最大仓位" value="10%" />
                </Col>
                <Col span={8}>
                  <Statistic title="建议持仓数" value="5-10只" />
                </Col>
                <Col span={8}>
                  <Statistic title="行业集中度" value="< 30%" />
                </Col>
              </Row>
            </Space>
          </Panel>
          
          <Panel header="止损止盈" key="2" extra={<WarningOutlined />}>
            <List
              size="small"
              dataSource={[
                { type: '初始止损', value: '2%', desc: '进场后的最大亏损限制' },
                { type: '移动止损', value: '3%', desc: '盈利后的回撤保护' },
                { type: '时间止损', value: '30天', desc: '长期横盘后退出' },
                { type: '止盈目标', value: '10%', desc: '分批止盈点位' },
              ]}
              renderItem={item => (
                <List.Item>
                  <List.Item.Meta
                    title={item.type}
                    description={item.desc}
                  />
                  <Tag color="orange">{item.value}</Tag>
                </List.Item>
              )}
            />
          </Panel>
        </Collapse>
      </Card>
    );
  };

  return (
    <div className="ai-strategy-assistant">
      <Card>
        <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
          <Col span={12}>
            <Space>
              <span>选择标的：</span>
              <Select
                mode="multiple"
                value={selectedSymbols}
                onChange={setSelectedSymbols}
                style={{ width: 300 }}
                placeholder="请选择股票"
              >
                <Option value="000001.SZ">平安银行</Option>
                <Option value="000002.SZ">万科A</Option>
                <Option value="000858.SZ">五粮液</Option>
                <Option value="002415.SZ">海康威视</Option>
              </Select>
            </Space>
          </Col>
          <Col span={12}>
            <Space>
              <span>风险偏好：</span>
              <Radio.Group value={riskPreference} onChange={e => setRiskPreference(e.target.value)}>
                <Radio.Button value="low">保守型</Radio.Button>
                <Radio.Button value="medium">稳健型</Radio.Button>
                <Radio.Button value="high">激进型</Radio.Button>
              </Radio.Group>
            </Space>
          </Col>
        </Row>

        {renderMarketConditionCard()}
        
        <Divider />

        <Row gutter={[16, 16]}>
          <Col span={16}>
            {activeTab === 'recommend' && renderStrategyRecommendations()}
            {activeTab === 'optimize' && renderOptimizationSuggestions()}
          </Col>
          <Col span={8}>
            {renderRiskControlSuggestions()}
          </Col>
        </Row>

        <div style={{ marginTop: 16, textAlign: 'center' }}>
          <Space>
            <Button
              type={activeTab === 'recommend' ? 'primary' : 'default'}
              onClick={() => setActiveTab('recommend')}
              icon={<RobotOutlined />}
            >
              策略推荐
            </Button>
            <Button
              type={activeTab === 'optimize' ? 'primary' : 'default'}
              onClick={() => setActiveTab('optimize')}
              icon={<BulbOutlined />}
            >
              参数优化
            </Button>
            <Button
              onClick={fetchRecommendations}
              icon={<SyncOutlined />}
              loading={loading}
            >
              刷新建议
            </Button>
          </Space>
        </div>
      </Card>
    </div>
  );
};

export default AIStrategyAssistant;