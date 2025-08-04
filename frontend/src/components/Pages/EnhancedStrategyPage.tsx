import React, { useState } from 'react';
import { Card, Row, Col, Table, Button, Modal, Form, Input, InputNumber, Select, message, Alert, Spin, Tag } from 'antd';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { PlusOutlined, PlayCircleOutlined, PauseCircleOutlined, SettingOutlined, DeleteOutlined } from '@ant-design/icons';
import { api } from '../../services/api';
import { StrategyConfig, StrategyPerformance } from '../../services/types';

const { Option } = Select;

interface StrategyData {
  name: string;
  config: StrategyConfig;
  performance?: any;
  status: 'running' | 'stopped';
}

const EnhancedStrategyPage: React.FC = () => {
  const queryClient = useQueryClient();
  const [form] = Form.useForm();
  const [selectedStrategy, setSelectedStrategy] = useState<StrategyData | null>(null);
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [strategies, setStrategies] = useState<StrategyData[]>([]);

  // Fetch strategy performance data
  const { 
    data: strategyPerformance, 
    isLoading: performanceLoading, 
    error: performanceError 
  } = useQuery<StrategyPerformance>({
    queryKey: ['strategy', 'performance'],
    queryFn: api.strategy.getAllPerformance,
    refetchInterval: 10000, // Refetch every 10 seconds
  });

  // Fetch available strategies
  const { 
    data: availableStrategies, 
    isLoading: strategiesLoading 
  } = useQuery<string[]>({
    queryKey: ['strategy', 'list'],
    queryFn: api.strategy.getStrategies,
  });

  // Fetch available symbols
  const { 
    data: availableSymbols, 
    isLoading: symbolsLoading 
  } = useQuery<string[]>({
    queryKey: ['data', 'symbols'],
    queryFn: api.data.getSymbols,
  });

  // Add strategy mutation
  const addStrategyMutation = useMutation({
    mutationFn: (config: StrategyConfig) => api.strategy.addStrategy(config),
    onSuccess: () => {
      message.success('策略添加成功');
      setShowConfigModal(false);
      form.resetFields();
      queryClient.invalidateQueries({ queryKey: ['strategy'] });
    },
    onError: (error: any) => {
      message.error(`策略添加失败: ${error.message}`);
    },
  });

  // Update strategy mutation
  const updateStrategyMutation = useMutation({
    mutationFn: ({ name, config }: { name: string; config: Partial<StrategyConfig> }) => 
      api.strategy.updateStrategy(name, config),
    onSuccess: () => {
      message.success('策略更新成功');
      setShowConfigModal(false);
      form.resetFields();
      queryClient.invalidateQueries({ queryKey: ['strategy'] });
    },
    onError: (error: any) => {
      message.error(`策略更新失败: ${error.message}`);
    },
  });

  // Start strategy mutation
  const startStrategyMutation = useMutation({
    mutationFn: (strategyName: string) => api.strategy.startStrategy(strategyName),
    onSuccess: (_, strategyName) => {
      message.success(`策略 ${strategyName} 启动成功`);
      queryClient.invalidateQueries({ queryKey: ['strategy'] });
    },
    onError: (error: any) => {
      message.error(`策略启动失败: ${error.message}`);
    },
  });

  // Stop strategy mutation
  const stopStrategyMutation = useMutation({
    mutationFn: (strategyName: string) => api.strategy.stopStrategy(strategyName),
    onSuccess: (_, strategyName) => {
      message.success(`策略 ${strategyName} 停止成功`);
      queryClient.invalidateQueries({ queryKey: ['strategy'] });
    },
    onError: (error: any) => {
      message.error(`策略停止失败: ${error.message}`);
    },
  });

  // Delete strategy mutation
  const deleteStrategyMutation = useMutation({
    mutationFn: (strategyName: string) => api.strategy.deleteStrategy(strategyName),
    onSuccess: (_, strategyName) => {
      message.success(`策略 ${strategyName} 删除成功`);
      queryClient.invalidateQueries({ queryKey: ['strategy'] });
    },
    onError: (error: any) => {
      message.error(`策略删除失败: ${error.message}`);
    },
  });

  // Handle form submission
  const handleSubmit = async (values: any) => {
    const config: StrategyConfig = {
      name: values.name,
      symbol: values.symbol,
      initial_capital: values.initial_capital,
      risk_tolerance: values.risk_tolerance / 100, // Convert percentage to decimal
      max_position_ratio: values.max_position_ratio / 100,
      stop_loss: values.stop_loss / 100,
      take_profit: values.take_profit / 100,
      technical_indicators: {
        ma_period: values.ma_period || 20,
        rsi_period: values.rsi_period || 14,
        macd_fast: values.macd_fast || 12,
        macd_slow: values.macd_slow || 26,
        macd_signal: values.macd_signal || 9,
      },
    };

    if (selectedStrategy) {
      updateStrategyMutation.mutate({ name: selectedStrategy.name, config });
    } else {
      addStrategyMutation.mutate(config);
    }
  };

  // Handle strategy control
  const handleStrategyControl = (strategyName: string, action: 'start' | 'stop') => {
    if (action === 'start') {
      startStrategyMutation.mutate(strategyName);
    } else {
      stopStrategyMutation.mutate(strategyName);
    }
  };

  // Handle strategy deletion
  const handleDeleteStrategy = (strategyName: string) => {
    Modal.confirm({
      title: '确认删除',
      content: `确定要删除策略 "${strategyName}" 吗？此操作不可撤销。`,
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: () => deleteStrategyMutation.mutate(strategyName),
    });
  };

  // Open configuration modal
  const openConfigModal = (strategy?: StrategyData) => {
    setSelectedStrategy(strategy || null);
    if (strategy) {
      form.setFieldsValue({
        name: strategy.config.name,
        symbol: strategy.config.symbol,
        initial_capital: strategy.config.initial_capital,
        risk_tolerance: (strategy.config.risk_tolerance || 0) * 100,
        max_position_ratio: (strategy.config.max_position_ratio || 0) * 100,
        stop_loss: (strategy.config.stop_loss || 0) * 100,
        take_profit: (strategy.config.take_profit || 0) * 100,
        ma_period: strategy.config.technical_indicators?.ma_period,
        rsi_period: strategy.config.technical_indicators?.rsi_period,
        macd_fast: strategy.config.technical_indicators?.macd_fast,
        macd_slow: strategy.config.technical_indicators?.macd_slow,
        macd_signal: strategy.config.technical_indicators?.macd_signal,
      });
    } else {
      form.resetFields();
    }
    setShowConfigModal(true);
  };

  // Table columns
  const columns = [
    {
      title: '策略名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: StrategyData) => (
        <div>
          <div style={{ fontWeight: 'bold' }}>{text}</div>
          <div style={{ fontSize: '12px', color: '#666' }}>{record.config.symbol}</div>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'running' ? 'green' : 'red'}>
          {status === 'running' ? '运行中' : '已停止'}
        </Tag>
      ),
    },
    {
      title: '初始资金',
      dataIndex: ['config', 'initial_capital'],
      key: 'initial_capital',
      render: (value: number) => `¥${value.toLocaleString()}`,
    },
    {
      title: '风险容忍度',
      dataIndex: ['config', 'risk_tolerance'],
      key: 'risk_tolerance',
      render: (value: number) => `${(value * 100).toFixed(1)}%`,
    },
    {
      title: '最大仓位',
      dataIndex: ['config', 'max_position_ratio'],
      key: 'max_position_ratio',
      render: (value: number) => `${(value * 100).toFixed(1)}%`,
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: StrategyData) => (
        <div style={{ display: 'flex', gap: '8px' }}>
          <Button
            size="small"
            type={record.status === 'running' ? 'default' : 'primary'}
            icon={record.status === 'running' ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
            onClick={() => handleStrategyControl(record.name, record.status === 'running' ? 'stop' : 'start')}
            loading={startStrategyMutation.isLoading || stopStrategyMutation.isLoading}
          >
            {record.status === 'running' ? '停止' : '启动'}
          </Button>
          <Button
            size="small"
            icon={<SettingOutlined />}
            onClick={() => openConfigModal(record)}
          >
            配置
          </Button>
          <Button
            size="small"
            danger
            icon={<DeleteOutlined />}
            onClick={() => handleDeleteStrategy(record.name)}
            loading={deleteStrategyMutation.isLoading}
          >
            删除
          </Button>
        </div>
      ),
    },
  ];

  // Mock strategy data (in real app, this would come from API)
  React.useEffect(() => {
    if (availableStrategies) {
      const mockStrategies: StrategyData[] = availableStrategies.map((name, index) => ({
        name,
        status: index === 0 ? 'running' : 'stopped',
        config: {
          name,
          symbol: '000001.SZ',
          initial_capital: 1000000,
          risk_tolerance: 0.02,
          max_position_ratio: 0.3,
          stop_loss: 0.05,
          take_profit: 0.1,
          technical_indicators: {
            ma_period: 20,
            rsi_period: 14,
            macd_fast: 12,
            macd_slow: 26,
            macd_signal: 9,
          },
        },
      }));
      setStrategies(mockStrategies);
    }
  }, [availableStrategies]);

  const runningStrategies = strategies.filter(s => s.status === 'running').length;
  const totalSignals = strategyPerformance?.data?.total_signals || 0;
  const successfulTrades = strategyPerformance?.data?.successful_trades || 0;
  const totalReturn = strategyPerformance?.data?.total_return || 0;

  return (
    <div style={{ padding: '24px' }}>
      {/* Header */}
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0 }}>策略管理</h2>
        <Button 
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => openConfigModal()}
          loading={addStrategyMutation.isLoading}
        >
          添加策略
        </Button>
      </div>

      {/* Error Alert */}
      {performanceError && (
        <Alert
          message="策略性能数据加载失败"
          description={performanceError.message}
          type="error"
          showIcon
          closable
          style={{ marginBottom: '24px' }}
        />
      )}

      {/* Strategy Overview */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#1890ff' }}>
                {strategies.length}
              </div>
              <div style={{ color: '#666' }}>总策略数</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#52c41a' }}>
                {runningStrategies}
              </div>
              <div style={{ color: '#666' }}>运行中</div>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Spin spinning={performanceLoading}>
            <Card>
              <div style={{ textAlign: 'center' }}>
                <div style={{ 
                  fontSize: '24px', 
                  fontWeight: 'bold', 
                  color: totalReturn >= 0 ? '#52c41a' : '#ff4d4f' 
                }}>
                  {totalReturn >= 0 ? '+' : ''}{(totalReturn * 100).toFixed(1)}%
                </div>
                <div style={{ color: '#666' }}>总收益率</div>
              </div>
            </Card>
          </Spin>
        </Col>
        <Col span={6}>
          <Spin spinning={performanceLoading}>
            <Card>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#1890ff' }}>
                  {totalSignals}
                </div>
                <div style={{ color: '#666' }}>总信号数</div>
              </div>
            </Card>
          </Spin>
        </Col>
      </Row>

      {/* Strategy List */}
      <Card title="策略列表">
        <Table
          columns={columns}
          dataSource={strategies}
          rowKey="name"
          loading={strategiesLoading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条，共 ${total} 条`,
          }}
        />
      </Card>

      {/* Configuration Modal */}
      <Modal
        title={selectedStrategy ? '编辑策略' : '添加策略'}
        open={showConfigModal}
        onCancel={() => setShowConfigModal(false)}
        footer={null}
        width={800}
        destroyOnClose
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          initialValues={{
            initial_capital: 1000000,
            risk_tolerance: 2,
            max_position_ratio: 30,
            stop_loss: 5,
            take_profit: 10,
            ma_period: 20,
            rsi_period: 14,
            macd_fast: 12,
            macd_slow: 26,
            macd_signal: 9,
          }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="name"
                label="策略名称"
                rules={[{ required: true, message: '请输入策略名称' }]}
              >
                <Input placeholder="输入策略名称" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="symbol"
                label="股票代码"
                rules={[{ required: true, message: '请选择股票代码' }]}
              >
                <Select placeholder="选择股票代码" loading={symbolsLoading}>
                  {availableSymbols?.map(symbol => (
                    <Option key={symbol} value={symbol}>{symbol}</Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="initial_capital"
                label="初始资金 (¥)"
                rules={[{ required: true, message: '请输入初始资金' }]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  min={10000}
                  max={100000000}
                  formatter={value => `¥ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                  parser={value => value!.replace(/¥\s?|(,*)/g, '')}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="risk_tolerance"
                label="风险容忍度 (%)"
                rules={[{ required: true, message: '请输入风险容忍度' }]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  min={0.1}
                  max={10}
                  step={0.1}
                  formatter={value => `${value}%`}
                  parser={value => value!.replace('%', '')}
                />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item
                name="max_position_ratio"
                label="最大仓位比例 (%)"
                rules={[{ required: true, message: '请输入最大仓位比例' }]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  min={1}
                  max={100}
                  formatter={value => `${value}%`}
                  parser={value => value!.replace('%', '')}
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="stop_loss"
                label="止损比例 (%)"
                rules={[{ required: true, message: '请输入止损比例' }]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  min={0.1}
                  max={50}
                  step={0.1}
                  formatter={value => `${value}%`}
                  parser={value => value!.replace('%', '')}
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="take_profit"
                label="止盈比例 (%)"
                rules={[{ required: true, message: '请输入止盈比例' }]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  min={0.1}
                  max={100}
                  step={0.1}
                  formatter={value => `${value}%`}
                  parser={value => value!.replace('%', '')}
                />
              </Form.Item>
            </Col>
          </Row>

          <div style={{ marginBottom: '16px', fontWeight: 'bold', color: '#1890ff' }}>
            技术指标参数
          </div>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item name="ma_period" label="移动平均周期">
                <InputNumber style={{ width: '100%' }} min={5} max={200} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="rsi_period" label="RSI周期">
                <InputNumber style={{ width: '100%' }} min={5} max={50} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="macd_fast" label="MACD快线">
                <InputNumber style={{ width: '100%' }} min={5} max={50} />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="macd_slow" label="MACD慢线">
                <InputNumber style={{ width: '100%' }} min={10} max={100} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="macd_signal" label="MACD信号线">
                <InputNumber style={{ width: '100%' }} min={5} max={50} />
              </Form.Item>
            </Col>
          </Row>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '24px' }}>
            <Button onClick={() => setShowConfigModal(false)}>
              取消
            </Button>
            <Button 
              type="primary" 
              htmlType="submit"
              loading={addStrategyMutation.isLoading || updateStrategyMutation.isLoading}
            >
              {selectedStrategy ? '更新策略' : '添加策略'}
            </Button>
          </div>
        </Form>
      </Modal>
    </div>
  );
};

export default EnhancedStrategyPage;