import React, { useState } from 'react';
import { Card, Tabs, Row, Col, Space, Button, Typography, Divider } from 'antd';
import { 
  ExperimentOutlined, 
  BarChartOutlined, 
  SwapOutlined, 
  PlayCircleOutlined,
  HistoryOutlined,
  SettingOutlined
} from '@ant-design/icons';
import { BacktestConfigForm } from './BacktestConfigForm';
import { BacktestResults } from './BacktestResults';
import { StrategyComparison } from './StrategyComparison';
import { SimulationTrading } from './SimulationTrading';
import type { BacktestConfig, BacktestResult } from '../../services/types';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface BacktestPageProps {
  className?: string;
}

export const BacktestPage: React.FC<BacktestPageProps> = ({ className }) => {
  const [activeTab, setActiveTab] = useState('config');
  const [currentBacktest, setCurrentBacktest] = useState<BacktestResult | null>(null);
  const [backtestHistory, setBacktestHistory] = useState<BacktestResult[]>([]);

  const handleBacktestComplete = (result: BacktestResult) => {
    setCurrentBacktest(result);
    setBacktestHistory(prev => [result, ...prev]);
    setActiveTab('results');
  };

  const handleViewResult = (result: BacktestResult) => {
    setCurrentBacktest(result);
    setActiveTab('results');
  };

  const tabItems = [
    {
      key: 'config',
      label: (
        <span>
          <SettingOutlined />
          回测配置
        </span>
      ),
      children: (
        <BacktestConfigForm 
          onBacktestComplete={handleBacktestComplete}
        />
      ),
    },
    {
      key: 'results',
      label: (
        <span>
          <BarChartOutlined />
          回测结果
        </span>
      ),
      children: (
        <BacktestResults 
          result={currentBacktest}
          onNewBacktest={() => setActiveTab('config')}
        />
      ),
    },
    {
      key: 'comparison',
      label: (
        <span>
          <SwapOutlined />
          策略对比
        </span>
      ),
      children: (
        <StrategyComparison 
          backtestHistory={backtestHistory}
          onViewResult={handleViewResult}
        />
      ),
    },
    {
      key: 'simulation',
      label: (
        <span>
          <PlayCircleOutlined />
          模拟交易
        </span>
      ),
      children: (
        <SimulationTrading />
      ),
    },
  ];

  return (
    <div className={className}>
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card>
            <Space direction="vertical" size="small" style={{ width: '100%' }}>
              <Title level={2}>
                <ExperimentOutlined style={{ marginRight: 8 }} />
                回测分析
              </Title>
              <Text type="secondary">
                通过历史数据验证策略有效性，进行策略对比分析和模拟交易
              </Text>
            </Space>
          </Card>
        </Col>

        <Col span={24}>
          <Card>
            <Tabs
              activeKey={activeTab}
              onChange={setActiveTab}
              items={tabItems}
              size="large"
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default BacktestPage;