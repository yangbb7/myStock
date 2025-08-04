import React, { useState, useEffect } from 'react';
import {
  Form,
  Input,
  InputNumber,
  Select,
  Button,
  Card,
  Row,
  Col,
  message,
  Divider,
  Alert,
  Space,
  Typography,
  Tooltip,
  Modal,
} from 'antd';
import {
  DollarOutlined,
  StockOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons';
import { useMutation, useQuery } from '@tanstack/react-query';
import { api } from '../../services/api';
import { OrderRequest, OrderSide, OrderType, RiskMetrics, PortfolioSummary } from '../../services/types';
// Removed stockMapping import - now using API for stock data

const { Option } = Select;
const { Text, Title } = Typography;

interface OrderCreateFormProps {
  onSuccess?: (orderId: string) => void;
  onCancel?: () => void;
  initialValues?: Partial<OrderRequest>;
}

interface OrderFormData extends OrderRequest {
  estimatedValue?: number;
  riskCheck?: boolean;
}

const OrderCreateForm: React.FC<OrderCreateFormProps> = ({
  onSuccess,
  onCancel,
  initialValues,
}) => {
  const [form] = Form.useForm<OrderFormData>();
  const [estimatedValue, setEstimatedValue] = useState<number>(0);
  const [riskWarnings, setRiskWarnings] = useState<string[]>([]);
  const [confirmModalVisible, setConfirmModalVisible] = useState(false);
  const [orderData, setOrderData] = useState<OrderRequest | null>(null);

  // Get available symbols (fallback to predefined symbols if API not available)
  const { data: apiSymbols = [] } = useQuery({
    queryKey: ['symbols'],
    queryFn: () => api.data.getSymbols(),
  });
  
  // Use API symbols if available, otherwise use predefined symbols
  const symbols = apiSymbols.length > 0 ? apiSymbols : [];

  // Get current portfolio for risk checking
  const { data: portfolio } = useQuery({
    queryKey: ['portfolioSummary'],
    queryFn: () => api.portfolio.getSummary(),
  });

  // Get risk metrics for validation
  const { data: riskMetrics } = useQuery({
    queryKey: ['riskMetrics'],
    queryFn: () => api.risk.getMetrics(),
  });

  // Create order mutation
  const createOrderMutation = useMutation({
    mutationFn: (orderRequest: OrderRequest) => api.order.createOrder(orderRequest),
    onSuccess: (response) => {
      message.success(`订单创建成功！订单ID: ${response.orderId}`);
      form.resetFields();
      setEstimatedValue(0);
      setRiskWarnings([]);
      onSuccess?.(response.orderId);
    },
    onError: (error: any) => {
      message.error(`订单创建失败: ${error.message || '未知错误'}`);
    },
  });

  // Calculate estimated value when form values change
  const handleFormValuesChange = (changedValues: any, allValues: OrderFormData) => {
    const { quantity, price, orderType } = allValues;
    
    if (quantity && price && orderType !== 'MARKET') {
      const estimated = quantity * price;
      setEstimatedValue(estimated);
      
      // Perform risk checks
      performRiskCheck(allValues, estimated);
    } else if (quantity && orderType === 'MARKET') {
      // For market orders, we can't calculate exact value without current market price
      setEstimatedValue(0);
      performRiskCheck(allValues, 0);
    }
  };

  // Perform risk validation
  const performRiskCheck = (values: OrderFormData, estimated: number) => {
    const warnings: string[] = [];
    
    if (!portfolio || !riskMetrics) return;

    const { symbol, side, quantity, price } = values;
    
    // Check available cash for buy orders
    if (side === 'BUY' && estimated > 0) {
      if (estimated > portfolio.cashBalance) {
        warnings.push(`资金不足：需要 ¥${estimated.toLocaleString()}，可用 ¥${portfolio.cashBalance.toLocaleString()}`);
      }
    }

    // Check position size limits
    if (estimated > 0) {
      const positionRatio = estimated / portfolio.totalValue;
      if (positionRatio > riskMetrics.riskLimits.maxPositionSize) {
        warnings.push(`单笔订单超过最大仓位限制 ${(riskMetrics.riskLimits.maxPositionSize * 100).toFixed(1)}%`);
      }
    }

    // Check if selling more than current position
    if (side === 'SELL' && symbol && portfolio.positions[symbol]) {
      const currentPosition = portfolio.positions[symbol].quantity;
      if (quantity && quantity > currentPosition) {
        warnings.push(`卖出数量超过当前持仓：当前持有 ${currentPosition} 股`);
      }
    }

    // Check daily loss limits
    if (side === 'SELL' && symbol && portfolio.positions[symbol] && price) {
      const position = portfolio.positions[symbol];
      const potentialLoss = (position.averagePrice - price) * (quantity || 0);
      if (potentialLoss > 0) {
        const totalDailyLoss = riskMetrics.dailyPnl + potentialLoss;
        if (Math.abs(totalDailyLoss) > riskMetrics.riskLimits.maxDailyLoss) {
          warnings.push(`可能触发日损失限制：预计损失 ¥${potentialLoss.toLocaleString()}`);
        }
      }
    }

    setRiskWarnings(warnings);
  };

  // Handle form submission
  const handleSubmit = (values: OrderFormData) => {
    const orderRequest: OrderRequest = {
      symbol: values.symbol,
      side: values.side,
      quantity: values.quantity,
      price: values.price,
      orderType: values.orderType || 'MARKET',
      stopPrice: values.stopPrice,
      timeInForce: values.timeInForce,
    };

    // If there are risk warnings, show confirmation modal
    if (riskWarnings.length > 0) {
      setOrderData(orderRequest);
      setConfirmModalVisible(true);
    } else {
      createOrderMutation.mutate(orderRequest);
    }
  };

  // Handle confirmed order submission
  const handleConfirmedSubmit = () => {
    if (orderData) {
      createOrderMutation.mutate(orderData);
      setConfirmModalVisible(false);
      setOrderData(null);
    }
  };

  // Set initial form values
  useEffect(() => {
    if (initialValues) {
      form.setFieldsValue(initialValues);
    }
  }, [initialValues, form]);

  return (
    <>
      <Card
        title={
          <Space>
            <StockOutlined />
            <Title level={4} style={{ margin: 0 }}>创建订单</Title>
          </Space>
        }
        extra={
          onCancel && (
            <Button onClick={onCancel}>取消</Button>
          )
        }
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          onValuesChange={handleFormValuesChange}
          initialValues={{
            orderType: 'MARKET',
            side: 'BUY',
            timeInForce: 'DAY',
          }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="symbol"
                label="股票"
                rules={[
                  { required: true, message: '请选择股票' },
                ]}
              >
                <Select
                  placeholder="请选择股票代码或名称"
                  showSearch
                  filterOption={(input, option) =>
                    (option?.children as string)?.toLowerCase().includes(input.toLowerCase()) ||
                    (option?.value as string)?.toLowerCase().includes(input.toLowerCase())
                  }
                >
                  {symbols.map((symbol) => (
                    <Option key={symbol} value={symbol}>
                      {symbol}
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="side"
                label="买卖方向"
                rules={[{ required: true, message: '请选择买卖方向' }]}
              >
                <Select placeholder="请选择买卖方向">
                  <Option value="BUY">买入</Option>
                  <Option value="SELL">卖出</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="orderType"
                label="订单类型"
                rules={[{ required: true, message: '请选择订单类型' }]}
              >
                <Select placeholder="请选择订单类型">
                  <Option value="MARKET">市价单</Option>
                  <Option value="LIMIT">限价单</Option>
                  <Option value="STOP">止损单</Option>
                  <Option value="STOP_LIMIT">止损限价单</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="quantity"
                label="数量"
                rules={[
                  { required: true, message: '请输入数量' },
                  { type: 'number', min: 1, message: '数量必须大于0' },
                ]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="请输入数量"
                  min={1}
                  step={100}
                  formatter={(value) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                  parser={(value) => value!.replace(/\$\s?|(,*)/g, '')}
                />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item noStyle shouldUpdate={(prevValues, currentValues) => 
            prevValues.orderType !== currentValues.orderType
          }>
            {({ getFieldValue }) => {
              const orderType = getFieldValue('orderType');
              return orderType !== 'MARKET' ? (
                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item
                      name="price"
                      label="价格"
                      rules={[
                        { required: true, message: '请输入价格' },
                        { type: 'number', min: 0.01, message: '价格必须大于0' },
                      ]}
                    >
                      <InputNumber
                        style={{ width: '100%' }}
                        placeholder="请输入价格"
                        min={0.01}
                        step={0.01}
                        precision={2}
                        formatter={(value) => `¥ ${value}`}
                        parser={(value) => value!.replace(/¥\s?|(,*)/g, '')}
                      />
                    </Form.Item>
                  </Col>
                  {(orderType === 'STOP' || orderType === 'STOP_LIMIT') && (
                    <Col span={12}>
                      <Form.Item
                        name="stopPrice"
                        label="止损价格"
                        rules={[
                          { required: true, message: '请输入止损价格' },
                          { type: 'number', min: 0.01, message: '止损价格必须大于0' },
                        ]}
                      >
                        <InputNumber
                          style={{ width: '100%' }}
                          placeholder="请输入止损价格"
                          min={0.01}
                          step={0.01}
                          precision={2}
                          formatter={(value) => `¥ ${value}`}
                          parser={(value) => value!.replace(/¥\s?|(,*)/g, '')}
                        />
                      </Form.Item>
                    </Col>
                  )}
                </Row>
              ) : null;
            }}
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="timeInForce"
                label="有效期"
                tooltip="订单的有效时间"
              >
                <Select placeholder="请选择有效期">
                  <Option value="DAY">当日有效</Option>
                  <Option value="GTC">撤销前有效</Option>
                  <Option value="IOC">立即成交或取消</Option>
                  <Option value="FOK">全部成交或取消</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              {estimatedValue > 0 && (
                <div style={{ marginTop: 30 }}>
                  <Text strong>
                    <DollarOutlined /> 预估金额: ¥{estimatedValue.toLocaleString()}
                  </Text>
                </div>
              )}
            </Col>
          </Row>

          {/* Risk Warnings */}
          {riskWarnings.length > 0 && (
            <Alert
              message="风险提示"
              description={
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {riskWarnings.map((warning, index) => (
                    <li key={index}>{warning}</li>
                  ))}
                </ul>
              }
              type="warning"
              showIcon
              style={{ marginBottom: 16 }}
            />
          )}

          {/* Portfolio Info */}
          {portfolio && (
            <Card size="small" style={{ marginBottom: 16, backgroundColor: '#fafafa' }}>
              <Row gutter={16}>
                <Col span={8}>
                  <Text type="secondary">可用资金:</Text>
                  <br />
                  <Text strong>¥{portfolio.cashBalance.toLocaleString()}</Text>
                </Col>
                <Col span={8}>
                  <Text type="secondary">总资产:</Text>
                  <br />
                  <Text strong>¥{portfolio.totalValue.toLocaleString()}</Text>
                </Col>
                <Col span={8}>
                  <Text type="secondary">持仓数量:</Text>
                  <br />
                  <Text strong>{portfolio.positionsCount} 只</Text>
                </Col>
              </Row>
            </Card>
          )}

          <Divider />

          <Form.Item>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                loading={createOrderMutation.isPending}
                size="large"
              >
                创建订单
              </Button>
              {onCancel && (
                <Button size="large" onClick={onCancel}>
                  取消
                </Button>
              )}
            </Space>
          </Form.Item>
        </Form>
      </Card>

      {/* Risk Confirmation Modal */}
      <Modal
        title={
          <Space>
            <ExclamationCircleOutlined style={{ color: '#faad14' }} />
            风险确认
          </Space>
        }
        open={confirmModalVisible}
        onOk={handleConfirmedSubmit}
        onCancel={() => {
          setConfirmModalVisible(false);
          setOrderData(null);
        }}
        okText="确认提交"
        cancelText="取消"
        okButtonProps={{ 
          loading: createOrderMutation.isPending,
          danger: true,
        }}
      >
        <Alert
          message="检测到以下风险，请确认是否继续："
          description={
            <ul style={{ margin: '8px 0 0 0', paddingLeft: 20 }}>
              {riskWarnings.map((warning, index) => (
                <li key={index}>{warning}</li>
              ))}
            </ul>
          }
          type="warning"
          showIcon
        />
        <div style={{ marginTop: 16 }}>
          <Text type="secondary">
            <InfoCircleOutlined /> 继续提交订单可能会增加投资风险，请谨慎操作。
          </Text>
        </div>
      </Modal>
    </>
  );
};

export default OrderCreateForm;