import React, { useState } from 'react';
import {
  Card,
  Button,
  Space,
  Modal,
  Form,
  InputNumber,
  Select,
  Switch,
  Alert,
  Typography,
  Row,
  Col,
  Divider,
  Table,
  Tag,
  Popconfirm,
  message,
  Input,
  DatePicker,
  Tooltip,
  Progress
} from 'antd';
import {
  StopOutlined,
  PlayCircleOutlined,
  SettingOutlined,
  ExclamationCircleOutlined,
  HistoryOutlined,
  LockOutlined,
  UnlockOutlined,
  WarningOutlined,
  ReloadOutlined,
  AuditOutlined
} from '@ant-design/icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import dayjs from 'dayjs';
import { api } from '../../services/api';
import { RiskLimits } from '../../services/types';

const { Text, Title } = Typography;
const { Option } = Select;
const { TextArea } = Input;

interface RiskControlPanelProps {
  onEmergencyStop?: () => void;
  onSystemRestart?: () => void;
}

interface RiskControlAction {
  id: string;
  timestamp: string;
  action: string;
  operator: string;
  reason: string;
  parameters?: Record<string, any>;
  status: 'success' | 'failed' | 'pending';
  result?: string;
}

interface EmergencyStopConfig {
  stopAllStrategies: boolean;
  cancelAllOrders: boolean;
  closeAllPositions: boolean;
  reason: string;
  notifyUsers: boolean;
}

interface RiskLimitAdjustment {
  maxPositionSize?: number;
  maxDrawdownLimit?: number;
  maxDailyLoss?: number;
  reason: string;
  effectiveTime?: string;
  duration?: number; // in hours
}

const RiskControlPanel: React.FC<RiskControlPanelProps> = ({
  onEmergencyStop,
  onSystemRestart
}) => {
  const [isEmergencyModalVisible, setIsEmergencyModalVisible] = useState(false);
  const [isLimitAdjustModalVisible, setIsLimitAdjustModalVisible] = useState(false);
  const [isAuditModalVisible, setIsAuditModalVisible] = useState(false);
  const [emergencyForm] = Form.useForm();
  const [limitForm] = Form.useForm();
  const queryClient = useQueryClient();

  // Fetch current system status
  const { data: systemHealth } = useQuery({
    queryKey: ['systemHealth'],
    queryFn: () => api.system.getHealth(),
    refetchInterval: 5000,
  });

  // Fetch risk configuration
  const { data: riskConfig } = useQuery({
    queryKey: ['riskConfig'],
    queryFn: () => api.risk.getConfig(),
  });

  // Mock audit log data (in real implementation, this would come from API)
  const auditLogs: RiskControlAction[] = [
    {
      id: '1',
      timestamp: dayjs().subtract(1, 'hour').toISOString(),
      action: '紧急停止',
      operator: 'admin',
      reason: '风险指标超限',
      status: 'success',
      result: '所有策略已停止，订单已取消'
    },
    {
      id: '2',
      timestamp: dayjs().subtract(2, 'hours').toISOString(),
      action: '调整风险限制',
      operator: 'risk_manager',
      reason: '市场波动加剧',
      parameters: { maxDrawdownLimit: 0.05 },
      status: 'success',
      result: '风险限制已更新'
    },
    {
      id: '3',
      timestamp: dayjs().subtract(3, 'hours').toISOString(),
      action: '系统重启',
      operator: 'admin',
      reason: '系统维护',
      status: 'success',
      result: '系统重启完成'
    }
  ];

  // Emergency stop mutation
  const emergencyStopMutation = useMutation({
    mutationFn: async (config: EmergencyStopConfig) => {
      // First stop the system
      await api.system.stopSystem();
      

      
      return { success: true, message: '紧急停止执行成功' };
    },
    onSuccess: (data) => {
      message.success(data.message);
      setIsEmergencyModalVisible(false);
      emergencyForm.resetFields();
      queryClient.invalidateQueries({ queryKey: ['systemHealth'] });
      onEmergencyStop?.();
    },
    onError: (error: any) => {
      message.error(`紧急停止失败: ${error.message}`);
    },
  });

  // System restart mutation
  const systemRestartMutation = useMutation({
    mutationFn: async (reason: string) => {
      await api.system.restartSystem();

      return { success: true, message: '系统重启成功' };
    },
    onSuccess: (data) => {
      message.success(data.message);
      queryClient.invalidateQueries({ queryKey: ['systemHealth'] });
      onSystemRestart?.();
    },
    onError: (error: any) => {
      message.error(`系统重启失败: ${error.message}`);
    },
  });

  // Risk limit adjustment mutation
  const adjustLimitsMutation = useMutation({
    mutationFn: async (adjustment: RiskLimitAdjustment) => {
      const limits: Partial<RiskLimits> = {};
      
      if (adjustment.maxPositionSize !== undefined) {
        limits.maxPositionSize = adjustment.maxPositionSize / 100;
      }
      if (adjustment.maxDrawdownLimit !== undefined) {
        limits.maxDrawdownLimit = adjustment.maxDrawdownLimit / 100;
      }
      if (adjustment.maxDailyLoss !== undefined) {
        limits.maxDailyLoss = adjustment.maxDailyLoss;
      }

      await api.risk.updateLimits(limits);

      
      return { success: true, message: '风险限制调整成功' };
    },
    onSuccess: (data) => {
      message.success(data.message);
      setIsLimitAdjustModalVisible(false);
      limitForm.resetFields();
      queryClient.invalidateQueries({ queryKey: ['riskConfig'] });
      queryClient.invalidateQueries({ queryKey: ['riskMetrics'] });
    },
    onError: (error: any) => {
      message.error(`风险限制调整失败: ${error.message}`);
    },
  });

  const handleEmergencyStop = async (values: EmergencyStopConfig) => {
    emergencyStopMutation.mutate(values);
  };

  const handleSystemRestart = (reason: string) => {
    systemRestartMutation.mutate(reason);
  };

  const handleLimitAdjustment = async (values: RiskLimitAdjustment) => {
    adjustLimitsMutation.mutate(values);
  };

  const auditColumns = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: string) => dayjs(timestamp).format('YYYY-MM-DD HH:mm:ss'),
      sorter: (a: RiskControlAction, b: RiskControlAction) => 
        dayjs(a.timestamp).unix() - dayjs(b.timestamp).unix(),
    },
    {
      title: '操作',
      dataIndex: 'action',
      key: 'action',
      render: (action: string) => <Tag color="blue">{action}</Tag>,
    },
    {
      title: '操作员',
      dataIndex: 'operator',
      key: 'operator',
    },
    {
      title: '原因',
      dataIndex: 'reason',
      key: 'reason',
      ellipsis: true,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'success' ? 'green' : status === 'failed' ? 'red' : 'orange'}>
          {status === 'success' ? '成功' : status === 'failed' ? '失败' : '进行中'}
        </Tag>
      ),
    },
    {
      title: '结果',
      dataIndex: 'result',
      key: 'result',
      ellipsis: true,
    },
  ];

  const isSystemRunning = systemHealth?.systemRunning ?? false;

  return (
    <>
      <Card
        title={
          <Space>
            <LockOutlined />
            <span>风险控制操作</span>
          </Space>
        }
        extra={
          <Button
            size="small"
            icon={<AuditOutlined />}
            onClick={() => setIsAuditModalVisible(true)}
          >
            操作审计
          </Button>
        }
      >
        {/* System Status Alert */}
        <Alert
          message={`系统状态: ${isSystemRunning ? '运行中' : '已停止'}`}
          type={isSystemRunning ? 'success' : 'warning'}
          showIcon
          style={{ marginBottom: '16px' }}
        />

        {/* Emergency Controls */}
        <Card title="紧急控制" size="small" style={{ marginBottom: '16px' }}>
          <Row gutter={16}>
            <Col xs={24} sm={12} md={8}>
              <Button
                type="primary"
                danger
                size="large"
                icon={<StopOutlined />}
                onClick={() => setIsEmergencyModalVisible(true)}
                disabled={!isSystemRunning}
                block
              >
                紧急停止
              </Button>
            </Col>
            <Col xs={24} sm={12} md={8}>
              <Popconfirm
                title="确认重启系统？"
                description="系统重启将中断所有正在进行的操作"
                onConfirm={() => handleSystemRestart('手动重启')}
                okText="确认"
                cancelText="取消"
              >
                <Button
                  type="default"
                  size="large"
                  icon={<ReloadOutlined />}
                  loading={systemRestartMutation.isPending}
                  block
                >
                  系统重启
                </Button>
              </Popconfirm>
            </Col>
            <Col xs={24} sm={12} md={8}>
              <Button
                type="default"
                size="large"
                icon={<PlayCircleOutlined />}
                onClick={() => api.system.startSystem()}
                disabled={isSystemRunning}
                block
              >
                启动系统
              </Button>
            </Col>
          </Row>
        </Card>

        {/* Risk Limit Controls */}
        <Card title="风险限制控制" size="small" style={{ marginBottom: '16px' }}>
          <Row gutter={16} style={{ marginBottom: '16px' }}>
            <Col xs={24} sm={8}>
              <Button
                type="primary"
                icon={<SettingOutlined />}
                onClick={() => setIsLimitAdjustModalVisible(true)}
                block
              >
                调整风险限制
              </Button>
            </Col>
            <Col xs={24} sm={8}>
              <Button
                type="default"
                icon={<LockOutlined />}
                block
              >
                锁定交易
              </Button>
            </Col>
            <Col xs={24} sm={8}>
              <Button
                type="default"
                icon={<UnlockOutlined />}
                block
              >
                解锁交易
              </Button>
            </Col>
          </Row>

          {/* Current Risk Limits Display */}
          {riskConfig && (
            <Row gutter={16}>
              <Col xs={24} sm={8}>
                <div style={{ textAlign: 'center', padding: '12px', background: '#f5f5f5', borderRadius: '4px' }}>
                  <Text type="secondary">最大仓位比例</Text>
                  <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#1890ff' }}>
                    {((riskConfig.maxPositionSize || 0) * 100).toFixed(1)}%
                  </div>
                </div>
              </Col>
              <Col xs={24} sm={8}>
                <div style={{ textAlign: 'center', padding: '12px', background: '#f5f5f5', borderRadius: '4px' }}>
                  <Text type="secondary">最大回撤限制</Text>
                  <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#ff4d4f' }}>
                    {((riskConfig.maxDrawdownLimit || 0) * 100).toFixed(1)}%
                  </div>
                </div>
              </Col>
              <Col xs={24} sm={8}>
                <div style={{ textAlign: 'center', padding: '12px', background: '#f5f5f5', borderRadius: '4px' }}>
                  <Text type="secondary">最大日损失</Text>
                  <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#faad14' }}>
                    ¥{(riskConfig.maxDailyLoss || 0).toLocaleString()}
                  </div>
                </div>
              </Col>
            </Row>
          )}
        </Card>

        {/* Manual Intervention */}
        <Card title="手动干预" size="small">
          <Row gutter={16}>
            <Col xs={24} sm={12} md={6}>
              <Button type="default" block>
                强制平仓
              </Button>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Button type="default" block>
                取消所有订单
              </Button>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Button type="default" block>
                暂停策略
              </Button>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Button type="default" block>
                恢复策略
              </Button>
            </Col>
          </Row>
        </Card>
      </Card>

      {/* Emergency Stop Modal */}
      <Modal
        title={
          <Space>
            <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />
            <span>紧急停止确认</span>
          </Space>
        }
        open={isEmergencyModalVisible}
        onCancel={() => setIsEmergencyModalVisible(false)}
        onOk={() => emergencyForm.submit()}
        confirmLoading={emergencyStopMutation.isPending}
        okText="执行紧急停止"
        okButtonProps={{ danger: true }}
        cancelText="取消"
      >
        <Alert
          message="警告"
          description="紧急停止将立即中断所有交易活动，请谨慎操作！"
          type="warning"
          showIcon
          style={{ marginBottom: '16px' }}
        />

        <Form
          form={emergencyForm}
          layout="vertical"
          onFinish={handleEmergencyStop}
        >
          <Form.Item
            name="reason"
            label="停止原因"
            rules={[{ required: true, message: '请输入停止原因' }]}
          >
            <TextArea
              rows={3}
              placeholder="请详细说明紧急停止的原因..."
            />
          </Form.Item>

          <Form.Item name="stopAllStrategies" valuePropName="checked" initialValue={true}>
            <Switch /> 停止所有策略
          </Form.Item>

          <Form.Item name="cancelAllOrders" valuePropName="checked" initialValue={true}>
            <Switch /> 取消所有订单
          </Form.Item>

          <Form.Item name="closeAllPositions" valuePropName="checked" initialValue={false}>
            <Switch /> 强制平仓所有持仓
          </Form.Item>

          <Form.Item name="notifyUsers" valuePropName="checked" initialValue={true}>
            <Switch /> 通知相关用户
          </Form.Item>
        </Form>
      </Modal>

      {/* Risk Limit Adjustment Modal */}
      <Modal
        title="调整风险限制"
        open={isLimitAdjustModalVisible}
        onCancel={() => setIsLimitAdjustModalVisible(false)}
        onOk={() => limitForm.submit()}
        confirmLoading={adjustLimitsMutation.isPending}
        okText="确认调整"
        cancelText="取消"
      >
        <Form
          form={limitForm}
          layout="vertical"
          onFinish={handleLimitAdjustment}
        >
          <Form.Item
            name="reason"
            label="调整原因"
            rules={[{ required: true, message: '请输入调整原因' }]}
          >
            <TextArea
              rows={2}
              placeholder="请说明调整风险限制的原因..."
            />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="maxPositionSize"
                label="最大仓位比例 (%)"
                rules={[
                  { type: 'number', min: 1, max: 100, message: '仓位比例必须在1-100%之间' }
                ]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  min={1}
                  max={100}
                  step={1}
                  formatter={value => `${value}%`}
                  parser={value => parseFloat(value!.replace('%', '')) || 0}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="maxDrawdownLimit"
                label="最大回撤限制 (%)"
                rules={[
                  { type: 'number', min: 1, max: 50, message: '回撤限制必须在1-50%之间' }
                ]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  min={1}
                  max={50}
                  step={0.1}
                  formatter={value => `${value}%`}
                  parser={value => parseFloat(value!.replace('%', '')) || 0}
                />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="maxDailyLoss"
            label="最大日损失 (¥)"
            rules={[
              { type: 'number', min: 100, message: '最大日损失不能小于100元' }
            ]}
          >
            <InputNumber
              style={{ width: '100%' }}
              min={100}
              step={100}
              formatter={value => `¥ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
              parser={value => parseFloat(value!.replace(/¥\s?|(,*)/g, '')) || 0}
            />
          </Form.Item>

          <Form.Item
            name="effectiveTime"
            label="生效时间"
          >
            <DatePicker
              showTime
              style={{ width: '100%' }}
              placeholder="立即生效"
            />
          </Form.Item>

          <Form.Item
            name="duration"
            label="持续时间 (小时)"
            help="留空表示永久生效"
          >
            <InputNumber
              style={{ width: '100%' }}
              min={1}
              max={168}
              step={1}
              placeholder="永久生效"
            />
          </Form.Item>
        </Form>
      </Modal>

      {/* Audit Log Modal */}
      <Modal
        title={
          <Space>
            <HistoryOutlined />
            <span>风险控制操作审计</span>
          </Space>
        }
        open={isAuditModalVisible}
        onCancel={() => setIsAuditModalVisible(false)}
        width={1000}
        footer={null}
      >
        <Table
          columns={auditColumns}
          dataSource={auditLogs}
          rowKey="id"
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条，共 ${total} 条`,
          }}
          scroll={{ x: 800 }}
        />
      </Modal>
    </>
  );
};

export default RiskControlPanel;