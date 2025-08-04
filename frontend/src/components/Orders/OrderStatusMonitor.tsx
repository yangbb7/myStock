import React, { useState, useEffect } from 'react';
import {
  Table,
  Card,
  Tag,
  Button,
  Space,
  Tooltip,
  Progress,
  Modal,
  message,
  Input,
  Select,
  DatePicker,
  Row,
  Col,
  Statistic,
  Alert,
  Typography,
} from 'antd';
import {
  ReloadOutlined,
  EyeOutlined,
  StopOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
} from '@ant-design/icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import dayjs from 'dayjs';
import { api } from '../../services/api';
import { OrderStatus, OrderSide, OrderFilter } from '../../services/types';
import { useWebSocket } from '../../hooks/useWebSocket';
// Removed stockMapping import - now using API for stock data

const { Search } = Input;
const { Option } = Select;
const { RangePicker } = DatePicker;
const { Text, Title } = Typography;

interface OrderStatusMonitorProps {
  refreshInterval?: number;
  showFilters?: boolean;
  showStats?: boolean;
}

const OrderStatusMonitor: React.FC<OrderStatusMonitorProps> = ({
  refreshInterval = 5000,
  showFilters = true,
  showStats = true,
}) => {
  const [selectedOrder, setSelectedOrder] = useState<OrderStatus | null>(null);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [filters, setFilters] = useState<OrderFilter>({});
  const [searchText, setSearchText] = useState('');
  
  const queryClient = useQueryClient();

  // Get order history with filters
  const { 
    data: orders = [], 
    isLoading,
    error,
    refetch 
  } = useQuery({
    queryKey: ['orderHistory', filters],
    queryFn: () => api.order.getOrderHistory(filters),
    refetchInterval: refreshInterval,
  });

  // Get active orders
  const { data: activeOrders = [] } = useQuery({
    queryKey: ['activeOrders'],
    queryFn: () => api.order.getActiveOrders(),
    refetchInterval: 1000, // More frequent updates for active orders
  });

  // Get order statistics
  const { data: orderStats } = useQuery({
    queryKey: ['orderStats', filters],
    queryFn: () => api.order.getOrderStats(filters),
    enabled: showStats,
  });

  // Cancel order mutation
  const cancelOrderMutation = useMutation({
    mutationFn: (orderId: string) => api.order.cancelOrder(orderId),
    onSuccess: () => {
      message.success('订单取消成功');
      queryClient.invalidateQueries({ queryKey: ['orderHistory'] });
      queryClient.invalidateQueries({ queryKey: ['activeOrders'] });
    },
    onError: (error: any) => {
      message.error(`订单取消失败: ${error.message}`);
    },
  });

  // WebSocket for real-time order updates
  useWebSocket('/ws/orders', {
    onMessage: (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'order_update') {
          // Update order data in cache
          queryClient.invalidateQueries({ queryKey: ['orderHistory'] });
          queryClient.invalidateQueries({ queryKey: ['activeOrders'] });
          
          // Show notification for important status changes
          const order = message.data as OrderStatus;
          if (order.status === 'FILLED') {
            message.success(`订单 ${order.orderId} 已成交`);
          } else if (order.status === 'REJECTED') {
            message.error(`订单 ${order.orderId} 被拒绝: ${order.errorMessage}`);
          }
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    },
  });

  // Get status color and icon
  const getStatusDisplay = (status: OrderStatus['status']) => {
    const statusConfig = {
      PENDING: { color: 'processing', icon: <ClockCircleOutlined />, text: '待处理' },
      FILLED: { color: 'success', icon: <CheckCircleOutlined />, text: '已成交' },
      PARTIALLY_FILLED: { color: 'warning', icon: <SyncOutlined />, text: '部分成交' },
      REJECTED: { color: 'error', icon: <CloseCircleOutlined />, text: '已拒绝' },
      CANCELLED: { color: 'default', icon: <StopOutlined />, text: '已取消' },
      ERROR: { color: 'error', icon: <ExclamationCircleOutlined />, text: '错误' },
    };

    const config = statusConfig[status] || statusConfig.ERROR;
    return (
      <Tag color={config.color} icon={config.icon}>
        {config.text}
      </Tag>
    );
  };

  // Calculate execution progress
  const getExecutionProgress = (order: OrderStatus) => {
    if (!order.executedQuantity || !order.quantity) return 0;
    return (order.executedQuantity / order.quantity) * 100;
  };

  // Handle order cancellation
  const handleCancelOrder = (orderId: string) => {
    Modal.confirm({
      title: '确认取消订单',
      content: `确定要取消订单 ${orderId} 吗？`,
      icon: <ExclamationCircleOutlined />,
      onOk: () => cancelOrderMutation.mutate(orderId),
    });
  };

  // Handle view order details
  const handleViewDetails = (order: OrderStatus) => {
    setSelectedOrder(order);
    setDetailModalVisible(true);
  };

  // Filter orders based on search text
  const filteredOrders = orders.filter(order => 
    !searchText || 
    order.orderId.toLowerCase().includes(searchText.toLowerCase()) ||
    order.symbol.toLowerCase().includes(searchText.toLowerCase()) ||
    order.symbol.toLowerCase().includes(searchText.toLowerCase())
  );

  // Table columns
  const columns = [
    {
      title: '订单ID',
      dataIndex: 'orderId',
      key: 'orderId',
      width: 120,
      render: (orderId: string) => (
        <Text code copyable={{ text: orderId }}>
          {orderId.slice(-8)}
        </Text>
      ),
    },
    {
      title: '股票',
      dataIndex: 'symbol',
      key: 'symbol',
      width: 150,
      render: (symbol: string) => (
        <div>
          <Text strong>{symbol}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {symbol}
          </Text>
        </div>
      ),
    },
    {
      title: '方向',
      dataIndex: 'side',
      key: 'side',
      width: 80,
      render: (side: OrderSide) => (
        <Tag color={side === 'BUY' ? 'green' : 'red'}>
          {side === 'BUY' ? '买入' : '卖出'}
        </Tag>
      ),
    },
    {
      title: '类型',
      dataIndex: 'orderType',
      key: 'orderType',
      width: 100,
      render: (type: string) => {
        const typeMap = {
          MARKET: '市价',
          LIMIT: '限价',
          STOP: '止损',
          STOP_LIMIT: '止损限价',
        };
        return typeMap[type as keyof typeof typeMap] || type;
      },
    },
    {
      title: '数量',
      dataIndex: 'quantity',
      key: 'quantity',
      width: 100,
      render: (quantity: number) => quantity.toLocaleString(),
    },
    {
      title: '价格',
      dataIndex: 'price',
      key: 'price',
      width: 100,
      render: (price: number) => price ? `¥${price.toFixed(2)}` : '-',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: OrderStatus['status']) => getStatusDisplay(status),
    },
    {
      title: '执行进度',
      key: 'progress',
      width: 120,
      render: (_, record: OrderStatus) => {
        const progress = getExecutionProgress(record);
        return progress > 0 ? (
          <Progress 
            percent={progress} 
            size="small" 
            format={(percent) => `${record.executedQuantity}/${record.quantity}`}
          />
        ) : '-';
      },
    },
    {
      title: '成交价',
      dataIndex: 'executedPrice',
      key: 'executedPrice',
      width: 100,
      render: (price: number) => price ? `¥${price.toFixed(2)}` : '-',
    },
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 150,
      render: (timestamp: string) => dayjs(timestamp).format('MM-DD HH:mm:ss'),
    },
    {
      title: '操作',
      key: 'actions',
      width: 120,
      render: (_, record: OrderStatus) => (
        <Space size="small">
          <Tooltip title="查看详情">
            <Button
              type="text"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => handleViewDetails(record)}
            />
          </Tooltip>
          {(record.status === 'PENDING' || record.status === 'PARTIALLY_FILLED') && (
            <Tooltip title="取消订单">
              <Button
                type="text"
                size="small"
                danger
                icon={<StopOutlined />}
                loading={cancelOrderMutation.isPending}
                onClick={() => handleCancelOrder(record.orderId)}
              />
            </Tooltip>
          )}
        </Space>
      ),
    },
  ];

  return (
    <div>
      {/* Statistics */}
      {showStats && orderStats && (
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={6}>
            <Card>
              <Statistic
                title="总订单数"
                value={orderStats.totalOrders || 0}
                prefix={<ClockCircleOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="成功率"
                value={orderStats.successRate || 0}
                suffix="%"
                precision={1}
                prefix={<CheckCircleOutlined />}
                valueStyle={{ color: '#3f8600' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="平均执行时间"
                value={orderStats.avgExecutionTime || 0}
                suffix="ms"
                prefix={<SyncOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="活跃订单"
                value={activeOrders.length}
                prefix={<ClockCircleOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* Active Orders Alert */}
      {activeOrders.length > 0 && (
        <Alert
          message={`当前有 ${activeOrders.length} 个活跃订单`}
          description="这些订单正在处理中，请注意监控执行状态"
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      {/* Filters */}
      {showFilters && (
        <Card size="small" style={{ marginBottom: 16 }}>
          <Row gutter={16} align="middle">
            <Col span={6}>
              <Search
                placeholder="搜索订单ID、股票代码或股票名称"
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                allowClear
              />
            </Col>
            <Col span={4}>
              <Select
                placeholder="订单状态"
                style={{ width: '100%' }}
                allowClear
                onChange={(status) => setFilters(prev => ({ ...prev, status: status ? [status] : undefined }))}
              >
                <Option value="PENDING">待处理</Option>
                <Option value="FILLED">已成交</Option>
                <Option value="PARTIALLY_FILLED">部分成交</Option>
                <Option value="REJECTED">已拒绝</Option>
                <Option value="CANCELLED">已取消</Option>
                <Option value="ERROR">错误</Option>
              </Select>
            </Col>
            <Col span={4}>
              <Select
                placeholder="买卖方向"
                style={{ width: '100%' }}
                allowClear
                onChange={(side) => setFilters(prev => ({ ...prev, side: side ? [side] : undefined }))}
              >
                <Option value="BUY">买入</Option>
                <Option value="SELL">卖出</Option>
              </Select>
            </Col>
            <Col span={6}>
              <RangePicker
                style={{ width: '100%' }}
                onChange={(dates) => {
                  if (dates) {
                    setFilters(prev => ({
                      ...prev,
                      startDate: dates[0]?.format('YYYY-MM-DD'),
                      endDate: dates[1]?.format('YYYY-MM-DD'),
                    }));
                  } else {
                    setFilters(prev => ({
                      ...prev,
                      startDate: undefined,
                      endDate: undefined,
                    }));
                  }
                }}
              />
            </Col>
            <Col span={4}>
              <Button
                icon={<ReloadOutlined />}
                onClick={() => refetch()}
                loading={isLoading}
              >
                刷新
              </Button>
            </Col>
          </Row>
        </Card>
      )}

      {/* Orders Table */}
      <Card
        title={
          <Space>
            <Title level={4} style={{ margin: 0 }}>订单监控</Title>
            {isLoading && <SyncOutlined spin />}
          </Space>
        }
        extra={
          <Button
            icon={<ReloadOutlined />}
            onClick={() => refetch()}
            loading={isLoading}
          >
            刷新
          </Button>
        }
      >
        {error && (
          <Alert
            message="数据加载失败"
            description={error.message}
            type="error"
            showIcon
            style={{ marginBottom: 16 }}
            action={
              <Button size="small" onClick={() => refetch()}>
                重试
              </Button>
            }
          />
        )}

        <Table
          columns={columns}
          dataSource={filteredOrders}
          rowKey="orderId"
          loading={isLoading}
          pagination={{
            total: filteredOrders.length,
            pageSize: 20,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条，共 ${total} 条`,
          }}
          scroll={{ x: 1200 }}
          size="small"
        />
      </Card>

      {/* Order Detail Modal */}
      <Modal
        title={`订单详情 - ${selectedOrder?.orderId}`}
        open={detailModalVisible}
        onCancel={() => {
          setDetailModalVisible(false);
          setSelectedOrder(null);
        }}
        footer={[
          <Button key="close" onClick={() => setDetailModalVisible(false)}>
            关闭
          </Button>,
          selectedOrder && (selectedOrder.status === 'PENDING' || selectedOrder.status === 'PARTIALLY_FILLED') && (
            <Button
              key="cancel"
              danger
              loading={cancelOrderMutation.isPending}
              onClick={() => {
                handleCancelOrder(selectedOrder.orderId);
                setDetailModalVisible(false);
              }}
            >
              取消订单
            </Button>
          ),
        ]}
        width={600}
      >
        {selectedOrder && (
          <div>
            <Row gutter={16}>
              <Col span={12}>
                <Text strong>订单ID:</Text>
                <br />
                <Text code copyable>{selectedOrder.orderId}</Text>
              </Col>
              <Col span={12}>
                <Text strong>状态:</Text>
                <br />
                {getStatusDisplay(selectedOrder.status)}
              </Col>
            </Row>
            <br />
            <Row gutter={16}>
              <Col span={12}>
                <Text strong>股票:</Text>
                <br />
                <Text strong>{selectedOrder.symbol}</Text>
                <br />
                <Text type="secondary">{selectedOrder.symbol}</Text>
              </Col>
              <Col span={12}>
                <Text strong>买卖方向:</Text>
                <br />
                <Tag color={selectedOrder.side === 'BUY' ? 'green' : 'red'}>
                  {selectedOrder.side === 'BUY' ? '买入' : '卖出'}
                </Tag>
              </Col>
            </Row>
            <br />
            <Row gutter={16}>
              <Col span={12}>
                <Text strong>订单数量:</Text>
                <br />
                <Text>{selectedOrder.quantity.toLocaleString()}</Text>
              </Col>
              <Col span={12}>
                <Text strong>订单价格:</Text>
                <br />
                <Text>{selectedOrder.price ? `¥${selectedOrder.price.toFixed(2)}` : '市价'}</Text>
              </Col>
            </Row>
            <br />
            <Row gutter={16}>
              <Col span={12}>
                <Text strong>已成交数量:</Text>
                <br />
                <Text>{selectedOrder.executedQuantity || 0}</Text>
              </Col>
              <Col span={12}>
                <Text strong>平均成交价:</Text>
                <br />
                <Text>{selectedOrder.avgExecutionPrice ? `¥${selectedOrder.avgExecutionPrice.toFixed(2)}` : '-'}</Text>
              </Col>
            </Row>
            <br />
            <Row gutter={16}>
              <Col span={12}>
                <Text strong>手续费:</Text>
                <br />
                <Text>{selectedOrder.commission ? `¥${selectedOrder.commission.toFixed(2)}` : '-'}</Text>
              </Col>
              <Col span={12}>
                <Text strong>创建时间:</Text>
                <br />
                <Text>{dayjs(selectedOrder.timestamp).format('YYYY-MM-DD HH:mm:ss')}</Text>
              </Col>
            </Row>
            {selectedOrder.errorMessage && (
              <>
                <br />
                <Text strong>错误信息:</Text>
                <br />
                <Alert
                  message={selectedOrder.errorMessage}
                  type="error"
                  showIcon
                />
              </>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default OrderStatusMonitor;