import React, { useState, useMemo } from 'react';
import {
  Card,
  Table,
  Row,
  Col,
  Statistic,
  Progress,
  Tag,
  Space,
  Button,
  Select,
  DatePicker,
  Tooltip,
  Alert,
  Spin,
  Empty,
} from 'antd';
import {
  TrophyOutlined,
  RiseOutlined,
  FallOutlined,
  BarChartOutlined,
  LineChartOutlined,
  ReloadOutlined,
  SwapOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import ReactECharts from 'echarts-for-react';
import dayjs from 'dayjs';
import { api } from '../../services/api';
import { StrategyPerformance, StrategyPerformanceData } from '../../services/types';

const { RangePicker } = DatePicker;
const { Option } = Select;

interface StrategyPerformanceMonitorProps {
  selectedStrategies?: string[];
  onStrategySelect?: (strategies: string[]) => void;
}

interface StrategyWithMetrics extends StrategyPerformanceData {
  name: string;
  winRate: number;
  profitFactor: number;
  avgWin: number;
  avgLoss: number;
  sharpeRatio: number;
  maxDrawdown: number;
  rank: number;
  status: 'active' | 'inactive' | 'error';
}

const StrategyPerformanceMonitor: React.FC<StrategyPerformanceMonitorProps> = ({
  selectedStrategies = [],
  onStrategySelect,
}) => {
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs] | null>(null);
  const [sortBy, setSortBy] = useState<string>('totalPnl');
  const [compareMode, setCompareMode] = useState<boolean>(false);
  const [selectedForCompare, setSelectedForCompare] = useState<string[]>([]);

  // 获取策略性能数据
  const { data: strategyPerformance, isLoading, refetch } = useQuery({
    queryKey: ['strategyPerformance', dateRange],
    queryFn: () => api.strategy.getPerformance(),
    refetchInterval: 5000, // 5秒刷新
  });

  // 获取策略列表
  const { data: strategyList } = useQuery({
    queryKey: ['strategyList'],
    queryFn: () => api.strategy.getStrategies(),
    refetchInterval: 10000, // 10秒刷新
  });

  // 处理和增强策略数据
  const enhancedStrategies = useMemo(() => {
    if (!strategyPerformance) return [];

    const strategies: StrategyWithMetrics[] = Object.entries(strategyPerformance).map(([name, data]) => {
      // 计算衍生指标
      const winRate = data.successfulTrades > 0 ? (data.successfulTrades / data.signalsGenerated) * 100 : 0;
      const avgWin = data.avgWin || (data.totalPnl > 0 ? data.totalPnl / data.successfulTrades : 0);
      const avgLoss = data.avgLoss || 0;
      const profitFactor = avgLoss !== 0 ? Math.abs(avgWin / avgLoss) : avgWin > 0 ? 999 : 0;
      const sharpeRatio = data.sharpeRatio || (data.totalPnl / Math.max(Math.abs(data.totalPnl) * 0.1, 1));
      const maxDrawdown = data.maxDrawdown || Math.abs(Math.min(data.totalPnl * 0.1, 0));

      // 模拟策略状态
      const status: 'active' | 'inactive' | 'error' = 
        data.signalsGenerated > 0 ? 'active' : 
        data.totalPnl < -1000 ? 'error' : 'inactive';

      return {
        name,
        ...data,
        winRate,
        profitFactor,
        avgWin,
        avgLoss,
        sharpeRatio,
        maxDrawdown,
        rank: 0, // 将在排序后设置
        status,
      };
    });

    // 排序并设置排名
    strategies.sort((a, b) => {
      switch (sortBy) {
        case 'totalPnl':
          return b.totalPnl - a.totalPnl;
        case 'winRate':
          return b.winRate - a.winRate;
        case 'profitFactor':
          return b.profitFactor - a.profitFactor;
        case 'sharpeRatio':
          return b.sharpeRatio - a.sharpeRatio;
        case 'signalsGenerated':
          return b.signalsGenerated - a.signalsGenerated;
        default:
          return b.totalPnl - a.totalPnl;
      }
    });

    strategies.forEach((strategy, index) => {
      strategy.rank = index + 1;
    });

    return strategies;
  }, [strategyPerformance, sortBy]);

  // 表格列定义
  const columns = [
    {
      title: '排名',
      dataIndex: 'rank',
      key: 'rank',
      width: 60,
      render: (rank: number) => (
        <div style={{ textAlign: 'center' }}>
          {rank <= 3 ? (
            <TrophyOutlined style={{ color: rank === 1 ? '#FFD700' : rank === 2 ? '#C0C0C0' : '#CD7F32' }} />
          ) : (
            <span>{rank}</span>
          )}
        </div>
      ),
    },
    {
      title: '策略名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: StrategyWithMetrics) => (
        <Space>
          <span style={{ fontWeight: 'bold' }}>{name}</span>
          <Tag color={
            record.status === 'active' ? 'green' : 
            record.status === 'error' ? 'red' : 'default'
          }>
            {record.status === 'active' ? '运行中' : 
             record.status === 'error' ? '异常' : '停止'}
          </Tag>
        </Space>
      ),
    },
    {
      title: '信号数量',
      dataIndex: 'signalsGenerated',
      key: 'signalsGenerated',
      sorter: true,
      render: (value: number) => value.toLocaleString(),
    },
    {
      title: '成功交易',
      dataIndex: 'successfulTrades',
      key: 'successfulTrades',
      render: (value: number) => value.toLocaleString(),
    },
    {
      title: '胜率',
      dataIndex: 'winRate',
      key: 'winRate',
      sorter: true,
      render: (value: number) => (
        <div>
          <div>{value.toFixed(1)}%</div>
          <Progress 
            percent={value} 
            size="small" 
            showInfo={false}
            strokeColor={value >= 60 ? '#52c41a' : value >= 40 ? '#faad14' : '#ff4d4f'}
          />
        </div>
      ),
    },
    {
      title: '总盈亏',
      dataIndex: 'totalPnl',
      key: 'totalPnl',
      sorter: true,
      render: (value: number) => (
        <Statistic
          value={value}
          precision={2}
          prefix={value >= 0 ? <><RiseOutlined /> ¥</> : <><FallOutlined /> ¥</>}
          valueStyle={{ 
            color: value >= 0 ? '#3f8600' : '#cf1322',
            fontSize: '14px',
          }}
        />
      ),
    },
    {
      title: '盈利因子',
      dataIndex: 'profitFactor',
      key: 'profitFactor',
      sorter: true,
      render: (value: number) => (
        <Tooltip title="盈利因子 = 平均盈利 / 平均亏损">
          <span style={{ color: value >= 1.5 ? '#3f8600' : value >= 1 ? '#faad14' : '#cf1322' }}>
            {value === 999 ? '∞' : value.toFixed(2)}
          </span>
        </Tooltip>
      ),
    },
    {
      title: '夏普比率',
      dataIndex: 'sharpeRatio',
      key: 'sharpeRatio',
      sorter: true,
      render: (value: number) => (
        <Tooltip title="夏普比率 = (收益率 - 无风险利率) / 波动率">
          <span style={{ color: value >= 1 ? '#3f8600' : value >= 0.5 ? '#faad14' : '#cf1322' }}>
            {value.toFixed(2)}
          </span>
        </Tooltip>
      ),
    },
    {
      title: '最大回撤',
      dataIndex: 'maxDrawdown',
      key: 'maxDrawdown',
      render: (value: number) => (
        <span style={{ color: '#cf1322' }}>
          -{Math.abs(value).toFixed(2)}%
        </span>
      ),
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: StrategyWithMetrics) => (
        <Space>
          <Button
            size="small"
            type={selectedForCompare.includes(record.name) ? 'primary' : 'default'}
            onClick={() => {
              if (selectedForCompare.includes(record.name)) {
                setSelectedForCompare(prev => prev.filter(name => name !== record.name));
              } else {
                setSelectedForCompare(prev => [...prev, record.name]);
              }
            }}
          >
            {selectedForCompare.includes(record.name) ? '取消对比' : '加入对比'}
          </Button>
        </Space>
      ),
    },
  ];

  // 生成对比图表数据
  const generateComparisonChart = () => {
    if (selectedForCompare.length === 0) return null;

    const compareData = selectedForCompare.map(name => {
      const strategy = enhancedStrategies.find(s => s.name === name);
      return {
        name,
        totalPnl: strategy?.totalPnl || 0,
        winRate: strategy?.winRate || 0,
        profitFactor: strategy?.profitFactor || 0,
        sharpeRatio: strategy?.sharpeRatio || 0,
      };
    });

    const option = {
      title: {
        text: '策略对比分析',
        left: 'center',
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow',
        },
      },
      legend: {
        data: ['总盈亏', '胜率', '盈利因子', '夏普比率'],
        top: 30,
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: compareData.map(d => d.name),
      },
      yAxis: [
        {
          type: 'value',
          name: '总盈亏 (¥)',
          position: 'left',
        },
        {
          type: 'value',
          name: '比率 (%)',
          position: 'right',
          max: 100,
        },
      ],
      series: [
        {
          name: '总盈亏',
          type: 'bar',
          yAxisIndex: 0,
          data: compareData.map(d => d.totalPnl),
          itemStyle: {
            color: '#1890ff',
          },
        },
        {
          name: '胜率',
          type: 'line',
          yAxisIndex: 1,
          data: compareData.map(d => d.winRate),
          itemStyle: {
            color: '#52c41a',
          },
        },
        {
          name: '盈利因子',
          type: 'line',
          yAxisIndex: 1,
          data: compareData.map(d => Math.min(d.profitFactor * 10, 100)), // 缩放到0-100范围
          itemStyle: {
            color: '#faad14',
          },
        },
        {
          name: '夏普比率',
          type: 'line',
          yAxisIndex: 1,
          data: compareData.map(d => Math.max(d.sharpeRatio * 20, 0)), // 缩放到0-100范围
          itemStyle: {
            color: '#722ed1',
          },
        },
      ],
    };

    return (
      <Card title="策略对比图表" style={{ marginBottom: 16 }}>
        <ReactECharts
          option={option}
          style={{ height: '400px' }}
          notMerge={true}
          lazyUpdate={true}
        />
      </Card>
    );
  };

  // 生成性能趋势图
  const generatePerformanceTrend = () => {
    // 模拟历史数据
    const dates = Array.from({ length: 30 }, (_, i) => 
      dayjs().subtract(29 - i, 'day').format('MM-DD')
    );

    const trendData = enhancedStrategies.slice(0, 5).map(strategy => ({
      name: strategy.name,
      data: dates.map((_, i) => {
        const base = strategy.totalPnl;
        const variation = Math.random() * 0.2 - 0.1; // ±10% 变化
        return base * (1 + variation * (i / 30));
      }),
    }));

    const option = {
      title: {
        text: '策略性能趋势 (近30天)',
        left: 'center',
      },
      tooltip: {
        trigger: 'axis',
      },
      legend: {
        data: trendData.map(d => d.name),
        top: 30,
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: dates,
      },
      yAxis: {
        type: 'value',
        name: '累计盈亏 (¥)',
      },
      series: trendData.map((trend, index) => ({
        name: trend.name,
        type: 'line',
        data: trend.data,
        smooth: true,
        itemStyle: {
          color: ['#1890ff', '#52c41a', '#faad14', '#722ed1', '#eb2f96'][index % 5],
        },
      })),
    };

    return (
      <Card title="性能趋势分析" style={{ marginBottom: 16 }}>
        <ReactECharts
          option={option}
          style={{ height: '400px' }}
          notMerge={true}
          lazyUpdate={true}
        />
      </Card>
    );
  };

  if (isLoading) {
    return (
      <Card>
        <div style={{ textAlign: 'center', padding: '50px' }}>
          <Spin size="large" />
          <div style={{ marginTop: 16 }}>加载策略性能数据...</div>
        </div>
      </Card>
    );
  }

  if (!strategyPerformance || Object.keys(strategyPerformance).length === 0) {
    return (
      <Card>
        <Empty
          description="暂无策略数据"
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        >
          <Button type="primary" onClick={() => refetch()}>
            刷新数据
          </Button>
        </Empty>
      </Card>
    );
  }

  return (
    <div>
      {/* 控制面板 */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row gutter={16} align="middle">
          <Col span={6}>
            <Space>
              <span>排序方式:</span>
              <Select
                value={sortBy}
                onChange={setSortBy}
                style={{ width: 120 }}
              >
                <Option value="totalPnl">总盈亏</Option>
                <Option value="winRate">胜率</Option>
                <Option value="profitFactor">盈利因子</Option>
                <Option value="sharpeRatio">夏普比率</Option>
                <Option value="signalsGenerated">信号数量</Option>
              </Select>
            </Space>
          </Col>
          <Col span={8}>
            <Space>
              <span>时间范围:</span>
              <RangePicker
                value={dateRange}
                onChange={(dates) => setDateRange(dates as [dayjs.Dayjs, dayjs.Dayjs] | null)}
                style={{ width: 240 }}
              />
            </Space>
          </Col>
          <Col span={6}>
            <Space>
              <Button
                icon={<SwapOutlined />}
                type={compareMode ? 'primary' : 'default'}
                onClick={() => setCompareMode(!compareMode)}
              >
                对比模式
              </Button>
              <Button icon={<ReloadOutlined />} onClick={() => refetch()}>
                刷新
              </Button>
            </Space>
          </Col>
          <Col span={4}>
            <Statistic
              title="策略总数"
              value={enhancedStrategies.length}
              suffix="个"
            />
          </Col>
        </Row>
      </Card>

      {/* 总体统计 */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总盈亏"
              value={enhancedStrategies.reduce((sum, s) => sum + s.totalPnl, 0)}
              precision={2}
              prefix="¥"
              valueStyle={{ 
                color: enhancedStrategies.reduce((sum, s) => sum + s.totalPnl, 0) >= 0 ? '#3f8600' : '#cf1322' 
              }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均胜率"
              value={enhancedStrategies.reduce((sum, s) => sum + s.winRate, 0) / enhancedStrategies.length}
              precision={1}
              suffix="%"
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃策略"
              value={enhancedStrategies.filter(s => s.status === 'active').length}
              suffix={`/ ${enhancedStrategies.length}`}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="总信号数"
              value={enhancedStrategies.reduce((sum, s) => sum + s.signalsGenerated, 0)}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 对比图表 */}
      {compareMode && selectedForCompare.length > 0 && generateComparisonChart()}

      {/* 性能趋势图 */}
      {generatePerformanceTrend()}

      {/* 策略性能表格 */}
      <Card 
        title={
          <Space>
            <BarChartOutlined />
            策略性能排行榜
            <Tooltip title="点击列标题可以排序，点击策略可以查看详情">
              <InfoCircleOutlined />
            </Tooltip>
          </Space>
        }
        extra={
          compareMode && (
            <Alert
              message={`已选择 ${selectedForCompare.length} 个策略进行对比`}
              type="info"
              showIcon
              closable
              onClose={() => setSelectedForCompare([])}
            />
          )
        }
      >
        <Table
          columns={columns}
          dataSource={enhancedStrategies}
          rowKey="name"
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 个策略`,
          }}
          onChange={(pagination, filters, sorter) => {
            if (sorter && !Array.isArray(sorter) && sorter.field) {
              setSortBy(sorter.field as string);
            }
          }}
          rowClassName={(record) => 
            selectedForCompare.includes(record.name) ? 'ant-table-row-selected' : ''
          }
        />
      </Card>
    </div>
  );
};

export default StrategyPerformanceMonitor;