import React, { useState, useMemo } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Select,
  DatePicker,
  Space,
  Typography,
  Alert,
  Spin,
} from 'antd';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import {
  RiseOutlined,
  ArrowDownOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  BarChartOutlined,
  PieChartOutlined,
} from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import dayjs from 'dayjs';
import { api } from '../../services/api';
import { DateRangeFilter, OrderStatus } from '../../services/types';

const { Option } = Select;
const { RangePicker } = DatePicker;
const { Title, Text } = Typography;

interface OrderAnalyticsProps {
  defaultTimeRange?: [dayjs.Dayjs, dayjs.Dayjs];
}

interface OrderStatsData {
  totalOrders: number;
  successRate: number;
  avgExecutionTime: number;
  totalVolume: number;
  totalValue: number;
  rejectionRate: number;
  cancellationRate: number;
  partialFillRate: number;
}

interface TimeSeriesData {
  date: string;
  orders: number;
  volume: number;
  value: number;
  successRate: number;
  avgExecutionTime: number;
}

interface SymbolStatsData {
  symbol: string;
  orders: number;
  volume: number;
  value: number;
  successRate: number;
  avgExecutionTime: number;
}

const OrderAnalytics: React.FC<OrderAnalyticsProps> = ({
  defaultTimeRange,
}) => {
  const [timeRange, setTimeRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>(
    defaultTimeRange || [dayjs().subtract(30, 'day'), dayjs()]
  );
  const [groupBy, setGroupBy] = useState<'day' | 'hour'>('day');

  const dateFilter: DateRangeFilter = {
    startDate: timeRange[0].format('YYYY-MM-DD'),
    endDate: timeRange[1].format('YYYY-MM-DD'),
  };

  // Get order statistics
  const { data: orderStats, isLoading: statsLoading } = useQuery({
    queryKey: ['orderStats', dateFilter],
    queryFn: () => api.order.getOrderStats(dateFilter),
  });

  // Get order history for detailed analysis
  const { data: orderHistory = [], isLoading: historyLoading } = useQuery({
    queryKey: ['orderHistory', dateFilter],
    queryFn: () => api.order.getOrderHistory(dateFilter),
  });

  // Process data for charts
  const chartData = useMemo(() => {
    if (!orderHistory.length) return { timeSeries: [], symbolStats: [], statusDistribution: [] };

    // Time series data
    const timeSeriesMap = new Map<string, {
      orders: number;
      volume: number;
      value: number;
      executionTimes: number[];
      successful: number;
    }>();

    // Symbol statistics
    const symbolStatsMap = new Map<string, {
      orders: number;
      volume: number;
      value: number;
      executionTimes: number[];
      successful: number;
    }>();

    // Status distribution
    const statusCount = new Map<string, number>();

    orderHistory.forEach((order: OrderStatus) => {
      const date = groupBy === 'day' 
        ? dayjs(order.timestamp).format('YYYY-MM-DD')
        : dayjs(order.timestamp).format('YYYY-MM-DD HH:00');

      // Time series
      if (!timeSeriesMap.has(date)) {
        timeSeriesMap.set(date, {
          orders: 0,
          volume: 0,
          value: 0,
          executionTimes: [],
          successful: 0,
        });
      }
      const timeData = timeSeriesMap.get(date)!;
      timeData.orders += 1;
      timeData.volume += order.executedQuantity || 0;
      timeData.value += (order.executedQuantity || 0) * (order.executedPrice || order.price || 0);
      if (order.status === 'FILLED') {
        timeData.successful += 1;
        // Simulate execution time (in real app, this would come from the API)
        timeData.executionTimes.push(Math.random() * 1000 + 100);
      }

      // Symbol statistics
      if (!symbolStatsMap.has(order.symbol)) {
        symbolStatsMap.set(order.symbol, {
          orders: 0,
          volume: 0,
          value: 0,
          executionTimes: [],
          successful: 0,
        });
      }
      const symbolData = symbolStatsMap.get(order.symbol)!;
      symbolData.orders += 1;
      symbolData.volume += order.executedQuantity || 0;
      symbolData.value += (order.executedQuantity || 0) * (order.executedPrice || order.price || 0);
      if (order.status === 'FILLED') {
        symbolData.successful += 1;
        symbolData.executionTimes.push(Math.random() * 1000 + 100);
      }

      // Status distribution
      statusCount.set(order.status, (statusCount.get(order.status) || 0) + 1);
    });

    // Convert to arrays
    const timeSeries: TimeSeriesData[] = Array.from(timeSeriesMap.entries()).map(([date, data]) => ({
      date,
      orders: data.orders,
      volume: data.volume,
      value: data.value,
      successRate: data.orders > 0 ? (data.successful / data.orders) * 100 : 0,
      avgExecutionTime: data.executionTimes.length > 0 
        ? data.executionTimes.reduce((a, b) => a + b, 0) / data.executionTimes.length 
        : 0,
    })).sort((a, b) => a.date.localeCompare(b.date));

    const symbolStats: SymbolStatsData[] = Array.from(symbolStatsMap.entries()).map(([symbol, data]) => ({
      symbol,
      orders: data.orders,
      volume: data.volume,
      value: data.value,
      successRate: data.orders > 0 ? (data.successful / data.orders) * 100 : 0,
      avgExecutionTime: data.executionTimes.length > 0 
        ? data.executionTimes.reduce((a, b) => a + b, 0) / data.executionTimes.length 
        : 0,
    })).sort((a, b) => b.orders - a.orders);

    const statusDistribution = Array.from(statusCount.entries()).map(([status, count]) => ({
      name: getStatusName(status),
      value: count,
      color: getStatusColor(status),
    }));

    return { timeSeries, symbolStats, statusDistribution };
  }, [orderHistory, groupBy]);

  const getStatusName = (status: string) => {
    const statusMap = {
      PENDING: '待处理',
      FILLED: '已成交',
      PARTIALLY_FILLED: '部分成交',
      REJECTED: '已拒绝',
      CANCELLED: '已取消',
      ERROR: '错误',
    };
    return statusMap[status as keyof typeof statusMap] || status;
  };

  const getStatusColor = (status: string) => {
    const colorMap = {
      PENDING: '#1890ff',
      FILLED: '#52c41a',
      PARTIALLY_FILLED: '#faad14',
      REJECTED: '#ff4d4f',
      CANCELLED: '#d9d9d9',
      ERROR: '#ff7875',
    };
    return colorMap[status as keyof typeof colorMap] || '#d9d9d9';
  };

  const isLoading = statsLoading || historyLoading;

  return (
    <div>
      {/* Controls */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row gutter={16} align="middle">
          <Col span={8}>
            <Space>
              <Text>时间范围:</Text>
              <RangePicker
                value={timeRange}
                onChange={(dates) => {
                  if (dates) {
                    setTimeRange([dates[0]!, dates[1]!]);
                  }
                }}
                format="YYYY-MM-DD"
              />
            </Space>
          </Col>
          <Col span={4}>
            <Space>
              <Text>分组:</Text>
              <Select
                value={groupBy}
                onChange={setGroupBy}
                style={{ width: 100 }}
              >
                <Option value="day">按天</Option>
                <Option value="hour">按小时</Option>
              </Select>
            </Space>
          </Col>
        </Row>
      </Card>

      {isLoading ? (
        <Card>
          <div style={{ textAlign: 'center', padding: '50px 0' }}>
            <Spin size="large" />
            <div style={{ marginTop: 16 }}>
              <Text>正在加载数据...</Text>
            </div>
          </div>
        </Card>
      ) : (
        <>
          {/* Key Metrics */}
          <Row gutter={16} style={{ marginBottom: 16 }}>
            <Col span={6}>
              <Card>
                <Statistic
                  title="总订单数"
                  value={orderStats?.totalOrders || 0}
                  prefix={<BarChartOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="成功率"
                  value={orderStats?.successRate || 0}
                  suffix="%"
                  precision={1}
                  prefix={<CheckCircleOutlined />}
                  valueStyle={{ 
                    color: (orderStats?.successRate || 0) >= 90 ? '#3f8600' : '#faad14' 
                  }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="平均执行时间"
                  value={orderStats?.avgExecutionTime || 0}
                  suffix="ms"
                  precision={0}
                  prefix={<ClockCircleOutlined />}
                  valueStyle={{ 
                    color: (orderStats?.avgExecutionTime || 0) <= 500 ? '#3f8600' : '#faad14' 
                  }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="总交易金额"
                  value={orderStats?.totalValue || 0}
                  prefix="¥"
                  precision={0}
                  formatter={(value) => `${Number(value).toLocaleString()}`}
                />
              </Card>
            </Col>
          </Row>

          <Row gutter={16} style={{ marginBottom: 16 }}>
            <Col span={6}>
              <Card>
                <Statistic
                  title="拒绝率"
                  value={orderStats?.rejectionRate || 0}
                  suffix="%"
                  precision={1}
                  prefix={<ArrowDownOutlined />}
                  valueStyle={{ 
                    color: (orderStats?.rejectionRate || 0) <= 5 ? '#3f8600' : '#ff4d4f' 
                  }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="取消率"
                  value={orderStats?.cancellationRate || 0}
                  suffix="%"
                  precision={1}
                  valueStyle={{ 
                    color: (orderStats?.cancellationRate || 0) <= 10 ? '#3f8600' : '#faad14' 
                  }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="部分成交率"
                  value={orderStats?.partialFillRate || 0}
                  suffix="%"
                  precision={1}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="总交易量"
                  value={orderStats?.totalVolume || 0}
                  formatter={(value) => `${Number(value).toLocaleString()}`}
                />
              </Card>
            </Col>
          </Row>

          {/* Charts */}
          <Row gutter={16} style={{ marginBottom: 16 }}>
            {/* Order Volume Trend */}
            <Col span={12}>
              <Card title="订单量趋势" size="small">
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={chartData.timeSeries}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="date" 
                      tick={{ fontSize: 12 }}
                      tickFormatter={(value) => 
                        groupBy === 'day' 
                          ? dayjs(value).format('MM-DD')
                          : dayjs(value).format('MM-DD HH:mm')
                      }
                    />
                    <YAxis tick={{ fontSize: 12 }} />
                    <Tooltip 
                      labelFormatter={(value) => 
                        groupBy === 'day' 
                          ? dayjs(value).format('YYYY-MM-DD')
                          : dayjs(value).format('YYYY-MM-DD HH:mm')
                      }
                    />
                    <Area 
                      type="monotone" 
                      dataKey="orders" 
                      stroke="#1890ff" 
                      fill="#1890ff" 
                      fillOpacity={0.3}
                      name="订单数"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </Card>
            </Col>

            {/* Success Rate Trend */}
            <Col span={12}>
              <Card title="成功率趋势" size="small">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData.timeSeries}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="date" 
                      tick={{ fontSize: 12 }}
                      tickFormatter={(value) => 
                        groupBy === 'day' 
                          ? dayjs(value).format('MM-DD')
                          : dayjs(value).format('MM-DD HH:mm')
                      }
                    />
                    <YAxis 
                      tick={{ fontSize: 12 }}
                      domain={[0, 100]}
                      tickFormatter={(value) => `${value}%`}
                    />
                    <Tooltip 
                      labelFormatter={(value) => 
                        groupBy === 'day' 
                          ? dayjs(value).format('YYYY-MM-DD')
                          : dayjs(value).format('YYYY-MM-DD HH:mm')
                      }
                      formatter={(value: number) => [`${value.toFixed(1)}%`, '成功率']}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="successRate" 
                      stroke="#52c41a" 
                      strokeWidth={2}
                      dot={{ r: 4 }}
                      name="成功率"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>

          <Row gutter={16} style={{ marginBottom: 16 }}>
            {/* Execution Time Trend */}
            <Col span={12}>
              <Card title="执行时间趋势" size="small">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData.timeSeries}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="date" 
                      tick={{ fontSize: 12 }}
                      tickFormatter={(value) => 
                        groupBy === 'day' 
                          ? dayjs(value).format('MM-DD')
                          : dayjs(value).format('MM-DD HH:mm')
                      }
                    />
                    <YAxis 
                      tick={{ fontSize: 12 }}
                      tickFormatter={(value) => `${value}ms`}
                    />
                    <Tooltip 
                      labelFormatter={(value) => 
                        groupBy === 'day' 
                          ? dayjs(value).format('YYYY-MM-DD')
                          : dayjs(value).format('YYYY-MM-DD HH:mm')
                      }
                      formatter={(value: number) => [`${value.toFixed(0)}ms`, '平均执行时间']}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="avgExecutionTime" 
                      stroke="#faad14" 
                      strokeWidth={2}
                      dot={{ r: 4 }}
                      name="平均执行时间"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </Col>

            {/* Order Status Distribution */}
            <Col span={12}>
              <Card title="订单状态分布" size="small">
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={chartData.statusDistribution}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {chartData.statusDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>

          {/* Symbol Statistics */}
          <Card title="股票交易统计" size="small">
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={chartData.symbolStats.slice(0, 10)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="symbol" tick={{ fontSize: 12 }} />
                <YAxis yAxisId="left" tick={{ fontSize: 12 }} />
                <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 12 }} />
                <Tooltip />
                <Legend />
                <Bar yAxisId="left" dataKey="orders" fill="#1890ff" name="订单数" />
                <Bar yAxisId="right" dataKey="successRate" fill="#52c41a" name="成功率(%)" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Performance Alerts */}
          {orderStats && (
            <div style={{ marginTop: 16 }}>
              {(orderStats.successRate || 0) < 90 && (
                <Alert
                  message="成功率告警"
                  description={`当前订单成功率为 ${(orderStats.successRate || 0).toFixed(1)}%，低于90%的标准`}
                  type="warning"
                  showIcon
                  style={{ marginBottom: 8 }}
                />
              )}
              {(orderStats.avgExecutionTime || 0) > 1000 && (
                <Alert
                  message="执行时间告警"
                  description={`平均执行时间为 ${(orderStats.avgExecutionTime || 0).toFixed(0)}ms，超过1秒标准`}
                  type="warning"
                  showIcon
                  style={{ marginBottom: 8 }}
                />
              )}
              {(orderStats.rejectionRate || 0) > 5 && (
                <Alert
                  message="拒绝率告警"
                  description={`订单拒绝率为 ${(orderStats.rejectionRate || 0).toFixed(1)}%，超过5%的标准`}
                  type="error"
                  showIcon
                  style={{ marginBottom: 8 }}
                />
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default OrderAnalytics;