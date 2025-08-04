import React, { useState, useEffect } from 'react';
import {
  Card,
  Form,
  Input,
  InputNumber,
  Select,
  Button,
  Row,
  Col,
  Divider,
  Space,
  message,
  Tooltip,
  Switch,
  Collapse,
  Alert,
} from 'antd';
import {
  PlusOutlined,
  InfoCircleOutlined,
  SettingOutlined,
  SaveOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import { useMutation, useQuery } from '@tanstack/react-query';
import { api } from '../../services/api';
import { StrategyConfig } from '../../services/types';

const { Option } = Select;
const { Panel } = Collapse;
const { TextArea } = Input;

interface StrategyConfigFormProps {
  onSuccess?: (strategy: StrategyConfig) => void;
  initialValues?: Partial<StrategyConfig>;
  mode?: 'create' | 'edit';
  strategyName?: string;
}

// 预定义的策略类型和参数模板
const STRATEGY_TEMPLATES = {
  'moving_average': {
    name: '移动平均策略',
    description: '基于移动平均线交叉的趋势跟踪策略',
    parameters: {
      short_period: 5,
      long_period: 20,
      signal_threshold: 0.02,
    },
    indicators: {
      sma_short: { period: 5 },
      sma_long: { period: 20 },
    },
  },
  'rsi_reversal': {
    name: 'RSI反转策略',
    description: '基于RSI指标的均值回归策略',
    parameters: {
      rsi_period: 14,
      oversold_threshold: 30,
      overbought_threshold: 70,
    },
    indicators: {
      rsi: { period: 14 },
    },
  },
  'bollinger_bands': {
    name: '布林带策略',
    description: '基于布林带的突破和回归策略',
    parameters: {
      period: 20,
      std_dev: 2,
      breakout_threshold: 0.01,
    },
    indicators: {
      bollinger: { period: 20, std_dev: 2 },
    },
  },
  'custom': {
    name: '自定义策略',
    description: '自定义参数配置',
    parameters: {},
    indicators: {},
  },
};

// 可用股票代码列表
const AVAILABLE_SYMBOLS = [
  '000001.SZ', '000002.SZ', '000858.SZ', '002415.SZ',
  '600000.SH', '600036.SH', '600519.SH', '600887.SH',
  '000858.SZ', '002594.SZ', '300059.SZ', '300750.SZ',
];

const StrategyConfigForm: React.FC<StrategyConfigFormProps> = ({
  onSuccess,
  initialValues,
  mode = 'create',
  strategyName,
}) => {
  const [form] = Form.useForm();
  const [selectedTemplate, setSelectedTemplate] = useState<string>('custom');
  const [previewData, setPreviewData] = useState<any>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // 获取可用股票代码
  const { data: availableSymbols } = useQuery({
    queryKey: ['availableSymbols'],
    queryFn: () => api.data.getSymbols(),
    initialData: AVAILABLE_SYMBOLS,
  });

  // 添加策略的mutation
  const addStrategyMutation = useMutation({
    mutationFn: (config: StrategyConfig) => api.strategy.addStrategy(config),
    onSuccess: (response) => {
      message.success('策略添加成功');
      form.resetFields();
      onSuccess?.(response.data);
    },
    onError: (error: any) => {
      message.error(`策略添加失败: ${error.message}`);
    },
  });

  // 更新策略的mutation
  const updateStrategyMutation = useMutation({
    mutationFn: ({ name, config }: { name: string; config: Partial<StrategyConfig> }) =>
      api.strategy.updateStrategy(name, config),
    onSuccess: () => {
      message.success('策略更新成功');
      onSuccess?.(form.getFieldsValue());
    },
    onError: (error: any) => {
      message.error(`策略更新失败: ${error.message}`);
    },
  });

  // 初始化表单值
  useEffect(() => {
    if (initialValues) {
      form.setFieldsValue(initialValues);
      // 根据初始值设置模板
      const template = Object.keys(STRATEGY_TEMPLATES).find(key => 
        key !== 'custom' && 
        JSON.stringify(STRATEGY_TEMPLATES[key as keyof typeof STRATEGY_TEMPLATES].parameters) === 
        JSON.stringify(initialValues.parameters)
      );
      if (template) {
        setSelectedTemplate(template);
      }
    }
  }, [initialValues, form]);

  // 处理策略模板选择
  const handleTemplateChange = (template: string) => {
    setSelectedTemplate(template);
    const templateConfig = STRATEGY_TEMPLATES[template as keyof typeof STRATEGY_TEMPLATES];
    
    if (template !== 'custom') {
      form.setFieldsValue({
        parameters: templateConfig.parameters,
        indicators: templateConfig.indicators,
      });
    }
  };

  // 实时预览策略配置
  const handleFormChange = () => {
    const values = form.getFieldsValue();
    setPreviewData(values);
  };

  // 验证策略配置
  const validateStrategy = async (values: any) => {
    const errors: string[] = [];

    // 基本验证
    if (!values.name || values.name.trim().length === 0) {
      errors.push('策略名称不能为空');
    }

    if (!values.symbols || values.symbols.length === 0) {
      errors.push('至少选择一个股票代码');
    }

    if (values.initialCapital < 10000) {
      errors.push('初始资金不能少于10,000元');
    }

    if (values.riskTolerance < 0.01 || values.riskTolerance > 0.1) {
      errors.push('风险容忍度应在1%-10%之间');
    }

    if (values.maxPositionSize < 0.01 || values.maxPositionSize > 1) {
      errors.push('最大仓位比例应在1%-100%之间');
    }

    // 止损止盈验证
    if (values.stopLoss && (values.stopLoss < 0.01 || values.stopLoss > 0.5)) {
      errors.push('止损比例应在1%-50%之间');
    }

    if (values.takeProfit && (values.takeProfit < 0.01 || values.takeProfit > 2)) {
      errors.push('止盈比例应在1%-200%之间');
    }

    return errors;
  };

  // 提交表单
  const handleSubmit = async (values: any) => {
    try {
      // 验证配置
      const errors = await validateStrategy(values);
      if (errors.length > 0) {
        message.error(errors.join('; '));
        return;
      }

      const strategyConfig: StrategyConfig = {
        name: values.name,
        symbols: values.symbols,
        initialCapital: values.initialCapital,
        riskTolerance: values.riskTolerance,
        maxPositionSize: values.maxPositionSize,
        stopLoss: values.stopLoss,
        takeProfit: values.takeProfit,
        indicators: values.indicators || {},
        parameters: values.parameters || {},
      };

      if (mode === 'create') {
        addStrategyMutation.mutate(strategyConfig);
      } else if (mode === 'edit' && strategyName) {
        updateStrategyMutation.mutate({ name: strategyName, config: strategyConfig });
      }
    } catch (error) {
      console.error('Form submission error:', error);
      message.error('提交失败，请检查配置');
    }
  };

  // 重置表单
  const handleReset = () => {
    form.resetFields();
    setSelectedTemplate('custom');
    setPreviewData(null);
  };

  const isLoading = addStrategyMutation.isPending || updateStrategyMutation.isPending;

  return (
    <Card 
      title={
        <Space>
          <SettingOutlined />
          {mode === 'create' ? '添加策略' : '编辑策略'}
        </Space>
      }
      extra={
        <Space>
          <Button icon={<ReloadOutlined />} onClick={handleReset}>
            重置
          </Button>
          <Switch
            checkedChildren="高级"
            unCheckedChildren="基础"
            checked={showAdvanced}
            onChange={setShowAdvanced}
          />
        </Space>
      }
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        onValuesChange={handleFormChange}
        initialValues={{
          initialCapital: 100000,
          riskTolerance: 0.02,
          maxPositionSize: 0.1,
          stopLoss: 0.05,
          takeProfit: 0.1,
        }}
      >
        {/* 策略模板选择 */}
        <Card size="small" title="策略模板" style={{ marginBottom: 16 }}>
          <Select
            value={selectedTemplate}
            onChange={handleTemplateChange}
            style={{ width: '100%' }}
            placeholder="选择策略模板"
          >
            {Object.entries(STRATEGY_TEMPLATES).map(([key, template]) => (
              <Option key={key} value={key}>
                <div>
                  <div style={{ fontWeight: 'bold' }}>{template.name}</div>
                  <div style={{ fontSize: '12px', color: '#666' }}>
                    {template.description}
                  </div>
                </div>
              </Option>
            ))}
          </Select>
        </Card>

        {/* 基础配置 */}
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item
              name="name"
              label={
                <Space>
                  策略名称
                  <Tooltip title="策略的唯一标识符，不能与现有策略重复">
                    <InfoCircleOutlined />
                  </Tooltip>
                </Space>
              }
              rules={[
                { required: true, message: '请输入策略名称' },
                { min: 2, max: 50, message: '策略名称长度应在2-50字符之间' },
                { pattern: /^[a-zA-Z0-9_\u4e00-\u9fa5]+$/, message: '只能包含字母、数字、下划线和中文' },
              ]}
            >
              <Input placeholder="请输入策略名称" />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              name="symbols"
              label={
                <Space>
                  股票代码
                  <Tooltip title="选择策略要交易的股票代码">
                    <InfoCircleOutlined />
                  </Tooltip>
                </Space>
              }
              rules={[{ required: true, message: '请选择至少一个股票代码' }]}
            >
              <Select
                mode="multiple"
                placeholder="请选择股票代码"
                showSearch
                filterOption={(input, option) =>
                  (option?.children as string)?.toLowerCase().includes(input.toLowerCase())
                }
              >
                {availableSymbols?.map((symbol) => (
                  <Option key={symbol} value={symbol}>
                    {symbol}
                  </Option>
                ))}
              </Select>
            </Form.Item>
          </Col>
        </Row>

        <Row gutter={16}>
          <Col span={8}>
            <Form.Item
              name="initialCapital"
              label={
                <Space>
                  初始资金
                  <Tooltip title="策略的初始资金，单位：元">
                    <InfoCircleOutlined />
                  </Tooltip>
                </Space>
              }
              rules={[
                { required: true, message: '请输入初始资金' },
                { type: 'number', min: 10000, message: '初始资金不能少于10,000元' },
              ]}
            >
              <InputNumber
                style={{ width: '100%' }}
                min={10000}
                max={10000000}
                step={10000}
                formatter={(value) => `¥ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                parser={(value) => value!.replace(/¥\s?|(,*)/g, '') as any}
              />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item
              name="riskTolerance"
              label={
                <Space>
                  风险容忍度
                  <Tooltip title="单笔交易的最大风险比例">
                    <InfoCircleOutlined />
                  </Tooltip>
                </Space>
              }
              rules={[
                { required: true, message: '请输入风险容忍度' },
                { type: 'number', min: 0.01, max: 0.1, message: '风险容忍度应在1%-10%之间' },
              ]}
            >
              <InputNumber
                style={{ width: '100%' }}
                min={0.01}
                max={0.1}
                step={0.01}
                formatter={(value) => `${((value || 0) * 100).toFixed(1)}%`}
                parser={(value) => (parseFloat(value!.replace('%', '')) / 100) as any}
              />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item
              name="maxPositionSize"
              label={
                <Space>
                  最大仓位比例
                  <Tooltip title="单个股票的最大仓位比例">
                    <InfoCircleOutlined />
                  </Tooltip>
                </Space>
              }
              rules={[
                { required: true, message: '请输入最大仓位比例' },
                { type: 'number', min: 0.01, max: 1, message: '最大仓位比例应在1%-100%之间' },
              ]}
            >
              <InputNumber
                style={{ width: '100%' }}
                min={0.01}
                max={1}
                step={0.01}
                formatter={(value) => `${((value || 0) * 100).toFixed(1)}%`}
                parser={(value) => (parseFloat(value!.replace('%', '')) / 100) as any}
              />
            </Form.Item>
          </Col>
        </Row>

        {/* 风险控制 */}
        <Divider>风险控制</Divider>
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item
              name="stopLoss"
              label={
                <Space>
                  止损比例
                  <Tooltip title="触发止损的价格下跌比例（可选）">
                    <InfoCircleOutlined />
                  </Tooltip>
                </Space>
              }
              rules={[
                { type: 'number', min: 0.01, max: 0.5, message: '止损比例应在1%-50%之间' },
              ]}
            >
              <InputNumber
                style={{ width: '100%' }}
                min={0.01}
                max={0.5}
                step={0.01}
                placeholder="可选"
                formatter={(value) => value ? `${(value * 100).toFixed(1)}%` : ''}
                parser={(value) => value ? (parseFloat(value.replace('%', '')) / 100) as any : undefined}
              />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              name="takeProfit"
              label={
                <Space>
                  止盈比例
                  <Tooltip title="触发止盈的价格上涨比例（可选）">
                    <InfoCircleOutlined />
                  </Tooltip>
                </Space>
              }
              rules={[
                { type: 'number', min: 0.01, max: 2, message: '止盈比例应在1%-200%之间' },
              ]}
            >
              <InputNumber
                style={{ width: '100%' }}
                min={0.01}
                max={2}
                step={0.01}
                placeholder="可选"
                formatter={(value) => value ? `${(value * 100).toFixed(1)}%` : ''}
                parser={(value) => value ? (parseFloat(value.replace('%', '')) / 100) as any : undefined}
              />
            </Form.Item>
          </Col>
        </Row>

        {/* 高级配置 */}
        {showAdvanced && (
          <Collapse style={{ marginBottom: 16 }}>
            <Panel header="技术指标配置" key="indicators">
              <Form.Item
                name="indicators"
                label="指标参数"
              >
                <TextArea
                  rows={4}
                  placeholder='JSON格式，例如: {"sma": {"period": 20}, "rsi": {"period": 14}}'
                />
              </Form.Item>
            </Panel>
            <Panel header="策略参数配置" key="parameters">
              <Form.Item
                name="parameters"
                label="策略参数"
              >
                <TextArea
                  rows={4}
                  placeholder='JSON格式，例如: {"threshold": 0.02, "lookback": 5}'
                />
              </Form.Item>
            </Panel>
          </Collapse>
        )}

        {/* 配置预览 */}
        {previewData && showAdvanced && (
          <Alert
            message="配置预览"
            description={
              <pre style={{ fontSize: '12px', maxHeight: '200px', overflow: 'auto' }}>
                {JSON.stringify(previewData, null, 2)}
              </pre>
            }
            type="info"
            style={{ marginBottom: 16 }}
          />
        )}

        {/* 提交按钮 */}
        <Form.Item>
          <Space>
            <Button
              type="primary"
              htmlType="submit"
              loading={isLoading}
              icon={<SaveOutlined />}
              size="large"
            >
              {mode === 'create' ? '添加策略' : '更新策略'}
            </Button>
            <Button onClick={handleReset} disabled={isLoading}>
              重置
            </Button>
          </Space>
        </Form.Item>
      </Form>
    </Card>
  );
};

export default StrategyConfigForm;