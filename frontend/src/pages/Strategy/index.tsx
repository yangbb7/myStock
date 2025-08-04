import React, { useState } from 'react';
import { Tabs, Card, Row, Col, Statistic, Space, Button, Alert } from 'antd';
import {
  SettingOutlined,
  BarChartOutlined,
  ControlOutlined,
  PlusOutlined,
  TrophyOutlined,
  RocketOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import { api } from '../../services/api';
import StrategyConfigForm from './StrategyConfigForm';
import StrategyPerformanceMonitor from './StrategyPerformanceMonitor';
import StrategyOperations from './StrategyOperations';

const { TabPane } = Tabs;

const StrategyManagementPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);

  // 获取策略性能数据用于概览
  const { data: strategyPerformance, isLoading } = useQuery({
    queryKey: ['strategyPerformance'],
    queryFn: () => api.strategy.getPerformance(),
    refetchInterval: 5000,
  });

  // 获取策略列表
  const { data: strategyList } = useQuery({
    queryKey: ['strategyList'],
    queryFn: () => api.strategy.getStrategies(),
    refetchInterval: 10000,
  });

  // 计算概览统计数据
  const overviewStats = React.useMemo(() => {
    if (!strategyPerformance || !strategyList) {
      return {
        totalStrategies: 0,
        activeStrategies: 0,
        totalPnl: 0,
        totalSignals: 0,
        avgWinRate: 0,
        bestStrategy: null,
        worstStrategy: null,
      };
    }

    const strategies = Object.entries(strategyPerformance);
    const totalPnl = strategies.reduce((sum, [, data]) => sum + data.totalPnl, 0);
    const totalSignals = strategies.reduce((sum, [, data]) => sum + data.signalsGenerated, 0);
    const activeStrategies = strategies.filter(([, data]) => data.signalsGenerated > 0).length;
    
    const avgWinRate = strategies.length > 0 
      ? strategies.reduce((sum, [, data]) => {
          const winRate = data.successfulTrades > 0 ? (data.successfulTrades / data.signalsGenerated) * 100 : 0;
          return sum + winRate;
        }, 0) / strategies.length
      : 0;

    const sortedByPnl = strategies.sort(([, a], [, b]) => b.totalPnl - a.totalPnl);
    const bestStrategy = sortedByPnl.length > 0 ? sortedByPnl[0] : null;
    const worstStrategy = sortedByPnl.length > 0 ? sortedByPnl[sortedByPnl.length - 1] : null;

    return {
      totalStrategies: strategyList.length,
      activeStrategies,
      totalPnl,
      totalSignals,
      avgWinRate,
      bestStrategy,
      worstStrategy,
    };
  }, [strategyPerformance, strategyList]);

  // 处理策略编辑
  const handleEditStrategy = (strategyName: string) => {
    setSelectedStrategy(strategyName);
    setActiveTab('config');
  };

  // 处理新策略添加成功
  const handleStrategyAdded = () => {
    setActiveTab('performance');
  };

  // 渲染概览页面
  const renderOverview = () => (
    <div>
      {/* 关键指标卡片 */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="策略总数"
              value={overviewStats.totalStrategies}
              suffix="个"
              prefix={<SettingOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃策略"
              value={overviewStats.activeStrategies}
              suffix={`/ ${overviewStats.totalStrategies}`}
              prefix={<RocketOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="总盈亏"
              value={overviewStats.totalPnl}
              precision={2}
              prefix="¥"
              valueStyle={{ 
                color: overviewStats.totalPnl >= 0 ? '#3f8600' : '#cf1322' 
              }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均胜率"
              value={overviewStats.avgWinRate}
              precision={1}
              suffix="%"
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 策略状态概览 */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={12}>
          <Card title="最佳表现策略" extra={<TrophyOutlined style={{ color: '#faad14' }} />}>
            {overviewStats.bestStrategy ? (
              <div>
                <div style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: 8 }}>
                  {overviewStats.bestStrategy[0]}
                </div>
                <Row gutter={16}>
                  <Col span={12}>
                    <Statistic
                      title="总盈亏"
                      value={overviewStats.bestStrategy[1].totalPnl}
                      precision={2}
                      prefix="¥"
                      valueStyle={{ color: '#3f8600' }}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="成功交易"
                      value={overviewStats.bestStrategy[1].successfulTrades}
                      suffix="笔"
                    />
                  </Col>
                </Row>
              </div>
            ) : (
              <Alert message="暂无策略数据" type="info" />
            )}
          </Card>
        </Col>
        <Col span={12}>
          <Card title="需要关注的策略" extra={<ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />}>
            {overviewStats.worstStrategy && overviewStats.worstStrategy[1].totalPnl < 0 ? (
              <div>
                <div style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: 8 }}>
                  {overviewStats.worstStrategy[0]}
                </div>
                <Row gutter={16}>
                  <Col span={12}>
                    <Statistic
                      title="总盈亏"
                      value={overviewStats.worstStrategy[1].totalPnl}
                      precision={2}
                      prefix="¥"
                      valueStyle={{ color: '#cf1322' }}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="信号数量"
                      value={overviewStats.worstStrategy[1].signalsGenerated}
                      suffix="个"
                    />
                  </Col>
                </Row>
                <Alert
                  message="建议检查策略参数或暂停运行"
                  type="warning"
                  style={{ marginTop: 12 }}
                />
              </div>
            ) : (
              <Alert message="所有策略表现良好" type="success" />
            )}
          </Card>
        </Col>
      </Row>

      {/* 快速操作 */}
      <Card title="快速操作">
        <Space size="large">
          <Button
            type="primary"
            icon={<PlusOutlined />}
            size="large"
            onClick={() => setActiveTab('config')}
          >
            添加新策略
          </Button>
          <Button
            icon={<BarChartOutlined />}
            size="large"
            onClick={() => setActiveTab('performance')}
          >
            查看性能分析
          </Button>
          <Button
            icon={<ControlOutlined />}
            size="large"
            onClick={() => setActiveTab('operations')}
          >
            策略操作管理
          </Button>
        </Space>
      </Card>
    </div>
  );

  return (
    <div style={{ padding: '24px' }}>
      <Card
        title={
          <Space>
            <SettingOutlined />
            策略管理中心
          </Space>
        }
        extra={
          <Space>
            <Statistic
              title="总信号数"
              value={overviewStats.totalSignals}
              valueStyle={{ fontSize: '14px' }}
            />
          </Space>
        }
      >
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          size="large"
          tabBarStyle={{ marginBottom: 24 }}
        >
          <TabPane
            tab={
              <Space>
                <BarChartOutlined />
                概览
              </Space>
            }
            key="overview"
          >
            {renderOverview()}
          </TabPane>

          <TabPane
            tab={
              <Space>
                <PlusOutlined />
                策略配置
              </Space>
            }
            key="config"
          >
            <StrategyConfigForm
              onSuccess={handleStrategyAdded}
              mode={selectedStrategy ? 'edit' : 'create'}
              strategyName={selectedStrategy || undefined}
            />
          </TabPane>

          <TabPane
            tab={
              <Space>
                <TrophyOutlined />
                性能监控
              </Space>
            }
            key="performance"
          >
            <StrategyPerformanceMonitor />
          </TabPane>

          <TabPane
            tab={
              <Space>
                <ControlOutlined />
                操作管理
              </Space>
            }
            key="operations"
          >
            <StrategyOperations onEditStrategy={handleEditStrategy} />
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
};

export default StrategyManagementPage;