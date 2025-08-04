import React, { useState, useEffect } from 'react';
import {
  Card,
  Form,
  Row,
  Col,
  Input,
  Select,
  DatePicker,
  InputNumber,
  Button,
  Space,
  Divider,
  Alert,
  Spin,
  Typography,
  Tooltip,
  Progress,
  message,
} from 'antd';
import {
  PlayCircleOutlined,
  SettingOutlined,
  InfoCircleOutlined,
  ReloadOutlined,
  SaveOutlined,
  HistoryOutlined,
} from '@ant-design/icons';
import { useQuery, useMutation } from '@tanstack/react-query';
import dayjs from 'dayjs';
import { api } from '../../services/api';
import type { BacktestConfig, BacktestResult, StrategyConfig } from '../../services/types';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;
const { TextArea } = Input;

interface BacktestConfigFormProps {
  onBacktestComplete?: (result: BacktestResult) => void;
  initialConfig?: Partial<BacktestConfig>;
  className?: string;
}

interface BacktestTask {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message?: string;
  result?: BacktestResult;
}

export const BacktestConfigForm: React.FC<BacktestConfigFormProps> = ({
  onBacktestComplete,
  initialConfig,
  className,
}) => {
  const [form] = Form.useForm();
  const [currentTask, setCurrentTask] = useState<BacktestTask | null>(null);
  const [savedConfigs, setSavedConfigs] = useState<BacktestConfig[]>([]);

  // Fetch available strategies
  const { data: strategies, isLoading: strategiesLoading } = useQuery({
    queryKey: ['strategies'],
    queryFn: () => api.strategy.getStrategies(),
  });

  // Fetch available symbols
  const { data: symbols, isLoading: symbolsLoading } = useQuery({
    queryKey: ['symbols'],
    queryFn: () => api.data.getSymbols(),
  });

  // Fetch strategy configuration when strategy is selected
  const [selectedStrategy, setSelectedStrategy] = useState<string>('');
  const { data: strategyConfig } = useQuery({
    queryKey: ['strategyConfig', selectedStrategy],
    queryFn: () => api.strategy.getStrategyConfig(selectedStrategy),
    enabled: !!selectedStrategy,
  });

  // Run backtest mutation
  const backtestMutation = useMutation({
    mutationFn: (config: BacktestConfig) => api.analytics.runBacktest(config),
    onSuccess: (result) => {
      setCurrentTask(prev => prev ? { ...prev, status: 'completed', progress: 100, result } : null);
      message.success('回测完成');
      onBacktestComplete?.(result);
    },
    onError: (error: any) => {
      setCurrentTask(prev => prev ? { ...prev, status: 'failed', message: error.message } : null);
      message.error(`回测失败: ${error.message}`);
    },
  });

  // Initialize form with default values
  useEffect(() => {
    const defaultValues = {
      startDate: dayjs().subtract(1, 'year'),
      endDate: dayjs(),
      initialCapital: 1000000,
      commission: 0.0003,
      slippage: 0.001,
      ...initialConfig,
    };
    form.setFieldsValue(defaultValues);
  }, [form, initialConfig]);

  // Update form when strategy config is loaded
  useEffect(() => {
    if (strategyConfig) {
      form.setFieldsValue({
        symbols: strategyConfig.symbols,
        parameters: strategyConfig.parameters,
      });
    }
  }, [strategyConfig, form]);

  const handleSubmit = async (values: any) => {
    try {
      const config: BacktestConfig = {
        strategyName: values.strategyName,
        symbols: values.symbols,
        startDate: values.dateRange[0].format('YYYY-MM-DD'),
        endDate: values.dateRange[1].format('YYYY-MM-DD'),
        initialCapital: values.initialCapital,
        commission: values.commission,
        slippage: values.slippage,
        parameters: values.parameters || {},
      };

      // Create task
      const taskId = `backtest_${Date.now()}`;
      setCurrentTask({
        id: taskId,
        status: 'running',
        progress: 0,
        message: '正在运行回测...',
      });

      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setCurrentTask(prev => {
          if (!prev || prev.status !== 'running') {
            clearInterval(progressInterval);
            return prev;
          }
          const newProgress = Math.min(prev.progress + Math.random() * 20, 90);
          return { ...prev, progress: newProgress };
        });
      }, 1000);

      // Run backtest
      await backtestMutation.mutateAsync(config);
      clearInterval(progressInterval);
    } catch (error) {
      console.error('Backtest submission error:', error);
    }
  };

  const handleSaveConfig = () => {
    form.validateFields().then(values => {
      const config: BacktestConfig = {
        strategyName: values.strategyName,
        symbols: values.symbols,
        startDate: values.dateRange[0].format('YYYY-MM-DD'),
        endDate: values.dateRange[1].format('YYYY-MM-DD'),
        initialCapital: values.initialCapital,
        commission: values.commission,
        slippage: values.slippage,
        parameters: values.parameters || {},
      };
      
      setSavedConfigs(prev => [...prev, config]);
      message.success('配置已保存');
    });
  };

  const handleLoadConfig = (config: BacktestConfig) => {
    form.setFieldsValue({
      strategyName: config.strategyName,
      symbols: config.symbols,
      dateRange: [dayjs(config.startDate), dayjs(config.endDate)],
      initialCapital: config.initialCapital,
      commission: config.commission,
      slippage: config.slippage,
      parameters: config.parameters,
    });
    setSelectedStrategy(config.strategyName);
    message.success('配置已加载');
  };

  const isLoading = strategiesLoading || symbolsLoading;
  const isRunning = currentTask?.status === 'running';

  return (
    <div className={className}>
      <Row gutter={[24, 24]}>
        {/* Configuration Form */}
        <Col xs={24} lg={18} xl={16}>
          <Card
            title={
              <Space>
                <SettingOutlined />
                回测配置
              </Space>
            }
            extra={
              <Space>
                <Button
                  icon={<SaveOutlined />}
                  onClick={handleSaveConfig}
                  disabled={isRunning}
                >
                  保存配置
                </Button>
                <Button
                  icon={<ReloadOutlined />}
                  onClick={() => form.resetFields()}
                  disabled={isRunning}
                >
                  重置
                </Button>
              </Space>
            }
          >
            <Spin spinning={isLoading}>
              <Form
                form={form}
                layout="vertical"
                onFinish={handleSubmit}
                disabled={isRunning}
              >
                <Divider orientation="left">策略选择</Divider>
                <Row gutter={16}>
                  <Col xs={24} md={12}>
                    <Form.Item
                      name="strategyName"
                      label="策略名称"
                      rules={[{ required: true, message: '请选择策略' }]}
                    >
                      <Select
                        placeholder="选择要回测的策略"
                        loading={strategiesLoading}
                        onChange={setSelectedStrategy}
                        showSearch
                        filterOption={(input, option) =>
                          (option?.children as string)?.toLowerCase().includes(input.toLowerCase())
                        }
                      >
                        {strategies?.map(strategy => (
                          <Option key={strategy} value={strategy}>
                            {strategy}
                          </Option>
                        ))}
                      </Select>
                    </Form.Item>
                  </Col>
                  <Col xs={24} md={12}>
                    <Form.Item
                      name="symbols"
                      label="股票代码"
                      rules={[{ required: true, message: '请选择股票代码' }]}
                    >
                      <Select
                        mode="multiple"
                        placeholder="选择股票代码"
                        loading={symbolsLoading}
                        showSearch
                        filterOption={(input, option) =>
                          (option?.value as string)?.toLowerCase().includes(input.toLowerCase())
                        }
                      >
                        {symbols?.map(symbol => (
                          <Option key={symbol} value={symbol}>
                            {symbol}
                          </Option>
                        ))}
                      </Select>
                    </Form.Item>
                  </Col>
                </Row>

                <Divider orientation="left">时间范围</Divider>
                <Row gutter={16}>
                  <Col span={24}>
                    <Form.Item
                      name="dateRange"
                      label="回测时间范围"
                      rules={[{ required: true, message: '请选择时间范围' }]}
                    >
                      <RangePicker
                        style={{ width: '100%' }}
                        format="YYYY-MM-DD"
                        disabledDate={(current) => current && current > dayjs().endOf('day')}
                      />
                    </Form.Item>
                  </Col>
                </Row>

                <Divider orientation="left">资金设置</Divider>
                <Row gutter={16}>
                  <Col xs={24} sm={12} lg={8}>
                    <Form.Item
                      name="initialCapital"
                      label={
                        <Space>
                          初始资金
                          <Tooltip title="回测开始时的初始资金量">
                            <InfoCircleOutlined />
                          </Tooltip>
                        </Space>
                      }
                      rules={[{ required: true, message: '请输入初始资金' }]}
                    >
                      <InputNumber
                        style={{ width: '100%' }}
                        min={10000}
                        max={100000000}
                        step={10000}
                        formatter={(value) => `¥ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                        parser={(value) => value!.replace(/¥\s?|(,*)/g, '')}
                      />
                    </Form.Item>
                  </Col>
                  <Col xs={24} sm={12} lg={8}>
                    <Form.Item
                      name="commission"
                      label={
                        <Space>
                          手续费率
                          <Tooltip title="每笔交易的手续费率，通常为0.0003 (万分之三)">
                            <InfoCircleOutlined />
                          </Tooltip>
                        </Space>
                      }
                      rules={[{ required: true, message: '请输入手续费率' }]}
                    >
                      <InputNumber
                        style={{ width: '100%' }}
                        min={0}
                        max={0.01}
                        step={0.0001}
                        precision={4}
                        formatter={(value) => `${(value! * 10000).toFixed(1)}‱`}
                        parser={(value) => parseFloat(value!.replace('‱', '')) / 10000}
                      />
                    </Form.Item>
                  </Col>
                  <Col xs={24} sm={12} lg={8}>
                    <Form.Item
                      name="slippage"
                      label={
                        <Space>
                          滑点率
                          <Tooltip title="实际成交价格与预期价格的偏差率">
                            <InfoCircleOutlined />
                          </Tooltip>
                        </Space>
                      }
                      rules={[{ required: true, message: '请输入滑点率' }]}
                    >
                      <InputNumber
                        style={{ width: '100%' }}
                        min={0}
                        max={0.01}
                        step={0.0001}
                        precision={4}
                        formatter={(value) => `${(value! * 100).toFixed(2)}%`}
                        parser={(value) => parseFloat(value!.replace('%', '')) / 100}
                      />
                    </Form.Item>
                  </Col>
                </Row>

                <Divider orientation="left">策略参数</Divider>
                <Form.Item
                  name="parameters"
                  label="自定义参数"
                >
                  <TextArea
                    rows={4}
                    placeholder="输入JSON格式的策略参数，例如: {&quot;ma_period&quot;: 20, &quot;rsi_period&quot;: 14}"
                  />
                </Form.Item>

                <Form.Item style={{ marginTop: 24 }}>
                  <Button
                    type="primary"
                    htmlType="submit"
                    icon={<PlayCircleOutlined />}
                    size="large"
                    loading={isRunning}
                    disabled={isRunning}
                  >
                    {isRunning ? '运行中...' : '开始回测'}
                  </Button>
                </Form.Item>
              </Form>
            </Spin>
          </Card>
        </Col>

        {/* Sidebar */}
        <Col xs={24} lg={6} xl={8}>
          {/* Running Task Status */}
          {currentTask && (
            <Card
              title="回测状态"
              style={{ marginBottom: 16 }}
            >
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Text strong>任务ID: </Text>
                  <Text code>{currentTask.id}</Text>
                </div>
                <div>
                  <Text strong>状态: </Text>
                  <Text type={
                    currentTask.status === 'completed' ? 'success' :
                    currentTask.status === 'failed' ? 'danger' :
                    currentTask.status === 'running' ? 'warning' : 'secondary'
                  }>
                    {currentTask.status === 'completed' ? '已完成' :
                     currentTask.status === 'failed' ? '失败' :
                     currentTask.status === 'running' ? '运行中' : '等待中'}
                  </Text>
                </div>
                {currentTask.status === 'running' && (
                  <Progress percent={Math.round(currentTask.progress)} />
                )}
                {currentTask.message && (
                  <Alert
                    message={currentTask.message}
                    type={currentTask.status === 'failed' ? 'error' : 'info'}
                    showIcon
                  />
                )}
              </Space>
            </Card>
          )}

          {/* Saved Configurations */}
          <Card
            title={
              <Space>
                <HistoryOutlined />
                保存的配置
              </Space>
            }
          >
            {savedConfigs.length === 0 ? (
              <Text type="secondary">暂无保存的配置</Text>
            ) : (
              <Space direction="vertical" style={{ width: '100%' }}>
                {savedConfigs.map((config, index) => (
                  <Card
                    key={index}
                    size="small"
                    hoverable
                    onClick={() => handleLoadConfig(config)}
                    style={{ cursor: 'pointer' }}
                  >
                    <Space direction="vertical" size="small">
                      <Text strong>{config.strategyName}</Text>
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {config.startDate} ~ {config.endDate}
                      </Text>
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        初始资金: ¥{config.initialCapital.toLocaleString()}
                      </Text>
                    </Space>
                  </Card>
                ))}
              </Space>
            )}
          </Card>

          {/* Quick Tips */}
          <Card title="使用提示" style={{ marginTop: 16 }}>
            <Space direction="vertical" size="small">
              <Text style={{ fontSize: '12px' }}>
                • 选择合适的时间范围，建议至少包含一个完整的市场周期
              </Text>
              <Text style={{ fontSize: '12px' }}>
                • 手续费率通常设置为万分之三 (0.0003)
              </Text>
              <Text style={{ fontSize: '12px' }}>
                • 滑点率建议设置为千分之一 (0.001)
              </Text>
              <Text style={{ fontSize: '12px' }}>
                • 可以保存常用配置以便快速重用
              </Text>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default BacktestConfigForm;