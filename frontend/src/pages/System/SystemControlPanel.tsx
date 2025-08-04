import React, { useState } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Space,
  Modal,
  Form,
  Input,
  Select,
  Checkbox,
  Table,
  Tag,
  Alert,
  Typography,
  Divider,
  Progress,
  Timeline,
  message,
  Popconfirm,
  Badge,
  Tooltip,
} from 'antd';
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  SettingOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  HistoryOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../../services/api';
import { SystemHealth, SystemControlRequest, SystemControlResponse } from '../../services/types';
import { formatDateTime, formatUptime } from '../../utils/format';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

interface SystemOperation {
  id: string;
  timestamp: string;
  operation: string;
  modules: string[];
  status: 'success' | 'failed' | 'in_progress';
  duration: number;
  operator: string;
  description: string;
  errorMessage?: string;
}

interface MaintenanceMode {
  enabled: boolean;
  startTime?: string;
  estimatedDuration?: number;
  reason?: string;
  allowedOperations?: string[];
}

const SystemControlCard: React.FC<{
  title: string;
  description: string;
  icon: React.ReactNode;
  action: () => void;
  loading?: boolean;
  disabled?: boolean;
  danger?: boolean;
}> = ({ title, description, icon, action, loading, disabled, danger }) => {
  return (
    <Card
      hoverable={!disabled}
      style={{
        textAlign: 'center',
        opacity: disabled ? 0.6 : 1,
      }}
    >
      <Space direction="vertical" size="middle">
        <div style={{ fontSize: 48, color: danger ? '#ff4d4f' : '#1890ff' }}>
          {icon}
        </div>
        <div>
          <Title level={4} style={{ margin: 0 }}>
            {title}
          </Title>
          <Paragraph type="secondary" style={{ margin: 0 }}>
            {description}
          </Paragraph>
        </div>
        <Button
          type={danger ? 'primary' : 'default'}
          danger={danger}
          size="large"
          onClick={action}
          loading={loading}
          disabled={disabled}
          style={{ width: '100%' }}
        >
          {title}
        </Button>
      </Space>
    </Card>
  );
};

const ModuleControlModal: React.FC<{
  visible: boolean;
  operation: 'start' | 'stop' | 'restart';
  modules: string[];
  onConfirm: (selectedModules: string[]) => void;
  onCancel: () => void;
}> = ({ visible, operation, modules, onConfirm, onCancel }) => {
  const [form] = Form.useForm();
  const [selectedModules, setSelectedModules] = useState<string[]>([]);

  const handleConfirm = () => {
    onConfirm(selectedModules);
  };

  const operationText = {
    start: '启动',
    stop: '停止',
    restart: '重启',
  };

  return (
    <Modal
      title={`${operationText[operation]}模块`}
      open={visible}
      onOk={handleConfirm}
      onCancel={onCancel}
      okText="确认"
      cancelText="取消"
      width={500}
    >
      <Alert
        message={`确认要${operationText[operation]}以下模块吗？`}
        description={`此操作将${operationText[operation]}选中的模块，可能会影响系统功能。`}
        type="warning"
        showIcon
        style={{ marginBottom: 16 }}
      />
      
      <Form form={form} layout="vertical">
        <Form.Item label="选择模块">
          <Checkbox.Group
            options={modules.map(module => ({ label: module, value: module }))}
            value={selectedModules}
            onChange={setSelectedModules}
          />
        </Form.Item>
        
        <Form.Item label="操作原因">
          <Input.TextArea
            placeholder={`请输入${operationText[operation]}原因...`}
            rows={3}
          />
        </Form.Item>
      </Form>
    </Modal>
  );
};

const MaintenanceModeModal: React.FC<{
  visible: boolean;
  currentMode: MaintenanceMode;
  onConfirm: (mode: MaintenanceMode) => void;
  onCancel: () => void;
}> = ({ visible, currentMode, onConfirm, onCancel }) => {
  const [form] = Form.useForm();

  const handleConfirm = async () => {
    try {
      const values = await form.validateFields();
      onConfirm({
        enabled: values.enabled,
        estimatedDuration: values.estimatedDuration,
        reason: values.reason,
        allowedOperations: values.allowedOperations || [],
      });
    } catch (error) {
      console.error('Form validation failed:', error);
    }
  };

  return (
    <Modal
      title="维护模式设置"
      open={visible}
      onOk={handleConfirm}
      onCancel={onCancel}
      okText="确认"
      cancelText="取消"
      width={600}
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={{
          enabled: currentMode.enabled,
          estimatedDuration: currentMode.estimatedDuration,
          reason: currentMode.reason,
          allowedOperations: currentMode.allowedOperations,
        }}
      >
        <Form.Item
          name="enabled"
          label="启用维护模式"
          valuePropName="checked"
        >
          <Checkbox>启用维护模式</Checkbox>
        </Form.Item>

        <Form.Item
          name="estimatedDuration"
          label="预计维护时长（分钟）"
          rules={[{ required: true, message: '请输入预计维护时长' }]}
        >
          <Select placeholder="选择维护时长">
            <Option value={30}>30分钟</Option>
            <Option value={60}>1小时</Option>
            <Option value={120}>2小时</Option>
            <Option value={240}>4小时</Option>
            <Option value={480}>8小时</Option>
          </Select>
        </Form.Item>

        <Form.Item
          name="reason"
          label="维护原因"
          rules={[{ required: true, message: '请输入维护原因' }]}
        >
          <Input.TextArea
            placeholder="请输入维护原因..."
            rows={3}
          />
        </Form.Item>

        <Form.Item
          name="allowedOperations"
          label="维护期间允许的操作"
        >
          <Checkbox.Group>
            <Row>
              <Col span={12}>
                <Checkbox value="read">只读访问</Checkbox>
              </Col>
              <Col span={12}>
                <Checkbox value="config">配置管理</Checkbox>
              </Col>
              <Col span={12}>
                <Checkbox value="monitor">系统监控</Checkbox>
              </Col>
              <Col span={12}>
                <Checkbox value="backup">数据备份</Checkbox>
              </Col>
            </Row>
          </Checkbox.Group>
        </Form.Item>
      </Form>
    </Modal>
  );
};

const OperationLogModal: React.FC<{
  visible: boolean;
  onClose: () => void;
}> = ({ visible, onClose }) => {
  // Mock operation log data
  const operationLogs: SystemOperation[] = [
    {
      id: '1',
      timestamp: '2024-01-15 14:30:00',
      operation: '系统重启',
      modules: ['data', 'strategy', 'execution'],
      status: 'success',
      duration: 45,
      operator: 'admin',
      description: '定期维护重启',
    },
    {
      id: '2',
      timestamp: '2024-01-15 10:15:00',
      operation: '模块停止',
      modules: ['risk'],
      status: 'success',
      duration: 5,
      operator: 'admin',
      description: '风险模块配置更新',
    },
    {
      id: '3',
      timestamp: '2024-01-14 16:45:00',
      operation: '系统启动',
      modules: ['all'],
      status: 'failed',
      duration: 30,
      operator: 'admin',
      description: '系统启动失败',
      errorMessage: '数据库连接超时',
    },
  ];

  const columns = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: string) => formatDateTime(timestamp),
    },
    {
      title: '操作',
      dataIndex: 'operation',
      key: 'operation',
      render: (operation: string) => <Text strong>{operation}</Text>,
    },
    {
      title: '模块',
      dataIndex: 'modules',
      key: 'modules',
      render: (modules: string[]) => (
        <Space wrap>
          {modules.map(module => (
            <Tag key={module} color="blue">
              {module}
            </Tag>
          ))}
        </Space>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const statusConfig = {
          success: { color: 'success', text: '成功' },
          failed: { color: 'error', text: '失败' },
          in_progress: { color: 'processing', text: '进行中' },
        };
        const config = statusConfig[status as keyof typeof statusConfig];
        return <Badge status={config.color as any} text={config.text} />;
      },
    },
    {
      title: '耗时',
      dataIndex: 'duration',
      key: 'duration',
      render: (duration: number) => `${duration}秒`,
    },
    {
      title: '操作者',
      dataIndex: 'operator',
      key: 'operator',
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
    },
  ];

  return (
    <Modal
      title="操作日志"
      open={visible}
      onCancel={onClose}
      footer={null}
      width={1000}
    >
      <Table
        columns={columns}
        dataSource={operationLogs}
        rowKey="id"
        pagination={{ pageSize: 10 }}
        size="small"
        expandable={{
          expandedRowRender: (record) => (
            <div style={{ padding: 16 }}>
              <Paragraph>
                <Text strong>详细描述：</Text> {record.description}
              </Paragraph>
              {record.errorMessage && (
                <Alert
                  message="错误信息"
                  description={record.errorMessage}
                  type="error"
                  showIcon
                />
              )}
            </div>
          ),
          rowExpandable: (record) => !!record.errorMessage,
        }}
      />
    </Modal>
  );
};

const SystemControlPanel: React.FC = () => {
  const [moduleControlVisible, setModuleControlVisible] = useState(false);
  const [maintenanceModeVisible, setMaintenanceModeVisible] = useState(false);
  const [operationLogVisible, setOperationLogVisible] = useState(false);
  const [currentOperation, setCurrentOperation] = useState<'start' | 'stop' | 'restart'>('start');

  const queryClient = useQueryClient();

  // Query system health
  const {
    data: systemHealth,
    isLoading: healthLoading,
    error: healthError,
  } = useQuery({
    queryKey: ['systemHealth'],
    queryFn: () => api.system.getHealth(),
    refetchInterval: 5000,
  });

  // Mock maintenance mode state
  const [maintenanceMode, setMaintenanceMode] = useState<MaintenanceMode>({
    enabled: false,
  });

  // System control mutations
  const startSystemMutation = useMutation({
    mutationFn: (modules?: string[]) => api.system.startSystem(modules),
    onSuccess: (response: SystemControlResponse) => {
      // message.success(response.message || '系统启动成功'); // 移除成功弹窗
      queryClient.invalidateQueries({ queryKey: ['systemHealth'] });
    },
    onError: (error: any) => {
      message.error(`系统启动失败: ${error.message}`);
    },
  });

  const stopSystemMutation = useMutation({
    mutationFn: (modules?: string[]) => api.system.stopSystem(modules),
    onSuccess: (response: SystemControlResponse) => {
      // message.success(response.message || '系统停止成功'); // 移除成功弹窗
      queryClient.invalidateQueries({ queryKey: ['systemHealth'] });
    },
    onError: (error: any) => {
      message.error(`系统停止失败: ${error.message}`);
    },
  });

  const restartSystemMutation = useMutation({
    mutationFn: (modules?: string[]) => api.system.restartSystem(modules),
    onSuccess: (response: SystemControlResponse) => {
      // message.success(response.message || '系统重启成功'); // 移除成功弹窗
      queryClient.invalidateQueries({ queryKey: ['systemHealth'] });
    },
    onError: (error: any) => {
      message.error(`系统重启失败: ${error.message}`);
    },
  });

  const handleSystemStart = () => {
    if (systemHealth?.systemRunning) {
      // message.warning('系统已在运行中'); // 移除警告弹窗
      return;
    }
    startSystemMutation.mutate(undefined);
  };

  const handleSystemStop = () => {
    if (!systemHealth?.systemRunning) {
      // message.warning('系统已停止'); // 移除警告弹窗
      return;
    }
    stopSystemMutation.mutate(undefined);
  };

  const handleSystemRestart = () => {
    restartSystemMutation.mutate(undefined);
  };

  const handleModuleOperation = (operation: 'start' | 'stop' | 'restart') => {
    setCurrentOperation(operation);
    setModuleControlVisible(true);
  };

  const handleModuleConfirm = (selectedModules: string[]) => {
    if (selectedModules.length === 0) {
      // message.warning('请选择要操作的模块'); // 移除警告弹窗
      return;
    }

    switch (currentOperation) {
      case 'start':
        startSystemMutation.mutate(selectedModules);
        break;
      case 'stop':
        stopSystemMutation.mutate(selectedModules);
        break;
      case 'restart':
        restartSystemMutation.mutate(selectedModules);
        break;
    }
    setModuleControlVisible(false);
  };

  const handleMaintenanceMode = (mode: MaintenanceMode) => {
    setMaintenanceMode(mode);
    message.success(mode.enabled ? '维护模式已启用' : '维护模式已禁用');
    setMaintenanceModeVisible(false);
  };

  const availableModules = systemHealth?.modules ? Object.keys(systemHealth.modules) : [];
  const isSystemRunning = systemHealth?.systemRunning || false;
  const isAnyOperationLoading = 
    startSystemMutation.isPending || 
    stopSystemMutation.isPending || 
    restartSystemMutation.isPending;

  if (healthError) {
    return (
      <div style={{ padding: 24 }}>
        <Alert
          message="系统状态获取失败"
          description="无法获取系统状态信息，请检查网络连接或联系管理员"
          type="error"
          showIcon
        />
      </div>
    );
  }

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>系统控制面板</Title>
        <Space>
          <Button icon={<HistoryOutlined />} onClick={() => setOperationLogVisible(true)}>
            操作日志
          </Button>
          <Button icon={<SettingOutlined />} onClick={() => setMaintenanceModeVisible(true)}>
            维护模式
          </Button>
        </Space>
      </div>

      {/* Maintenance Mode Alert */}
      {maintenanceMode.enabled && (
        <Alert
          message="系统维护模式"
          description={`系统当前处于维护模式，预计维护时长：${maintenanceMode.estimatedDuration}分钟。原因：${maintenanceMode.reason}`}
          type="warning"
          icon={<WarningOutlined />}
          showIcon
          closable
          style={{ marginBottom: 24 }}
        />
      )}

      {/* System Status Overview */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 48, marginBottom: 16 }}>
                {isSystemRunning ? (
                  <CheckCircleOutlined style={{ color: '#52c41a' }} />
                ) : (
                  <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
                )}
              </div>
              <Title level={4}>系统状态</Title>
              <Text type={isSystemRunning ? 'success' : 'danger'}>
                {isSystemRunning ? '运行中' : '已停止'}
              </Text>
            </div>
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 48, marginBottom: 16 }}>
                <InfoCircleOutlined style={{ color: '#1890ff' }} />
              </div>
              <Title level={4}>运行时间</Title>
              <Text>
                {systemHealth?.uptimeSeconds ? formatUptime(systemHealth.uptimeSeconds) : '0秒'}
              </Text>
            </div>
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 48, marginBottom: 16 }}>
                <SyncOutlined style={{ color: '#1890ff' }} />
              </div>
              <Title level={4}>活跃模块</Title>
              <Text>
                {systemHealth?.modules
                  ? `${Object.values(systemHealth.modules).filter(m => m.initialized).length} / ${Object.keys(systemHealth.modules).length}`
                  : '0 / 0'}
              </Text>
            </div>
          </Card>
        </Col>
      </Row>

      {/* System Control Actions */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <SystemControlCard
            title="启动系统"
            description="启动所有系统模块"
            icon={<PlayCircleOutlined />}
            action={handleSystemStart}
            loading={startSystemMutation.isPending}
            disabled={isSystemRunning || maintenanceMode.enabled}
          />
        </Col>
        <Col span={8}>
          <SystemControlCard
            title="停止系统"
            description="安全停止所有系统模块"
            icon={<PauseCircleOutlined />}
            action={handleSystemStop}
            loading={stopSystemMutation.isPending}
            disabled={!isSystemRunning}
            danger
          />
        </Col>
        <Col span={8}>
          <SystemControlCard
            title="重启系统"
            description="重启所有系统模块"
            icon={<ReloadOutlined />}
            action={handleSystemRestart}
            loading={restartSystemMutation.isPending}
            disabled={maintenanceMode.enabled}
            danger
          />
        </Col>
      </Row>

      {/* Module Control */}
      <Card title="模块控制" style={{ marginBottom: 24 }}>
        <Paragraph type="secondary">
          选择性启动、停止或重启特定模块。这对于维护和调试特定功能非常有用。
        </Paragraph>
        <Space>
          <Button
            icon={<PlayCircleOutlined />}
            onClick={() => handleModuleOperation('start')}
            disabled={isAnyOperationLoading || maintenanceMode.enabled}
          >
            启动模块
          </Button>
          <Button
            icon={<PauseCircleOutlined />}
            onClick={() => handleModuleOperation('stop')}
            disabled={isAnyOperationLoading}
          >
            停止模块
          </Button>
          <Button
            icon={<ReloadOutlined />}
            onClick={() => handleModuleOperation('restart')}
            disabled={isAnyOperationLoading || maintenanceMode.enabled}
          >
            重启模块
          </Button>
        </Space>
      </Card>

      {/* System Information */}
      <Card title="系统信息">
        <Row gutter={[16, 16]}>
          <Col span={12}>
            <div>
              <Text strong>系统版本：</Text>
              <Text>myQuant v1.0.0</Text>
            </div>
          </Col>
          <Col span={12}>
            <div>
              <Text strong>最后更新：</Text>
              <Text>
                {systemHealth?.timestamp ? formatDateTime(systemHealth.timestamp) : '--'}
              </Text>
            </div>
          </Col>
          <Col span={12}>
            <div>
              <Text strong>配置文件：</Text>
              <Text code>config/monolith_config.yaml</Text>
            </div>
          </Col>
          <Col span={12}>
            <div>
              <Text strong>日志级别：</Text>
              <Tag color="blue">INFO</Tag>
            </div>
          </Col>
        </Row>
      </Card>

      {/* Modals */}
      <ModuleControlModal
        visible={moduleControlVisible}
        operation={currentOperation}
        modules={availableModules}
        onConfirm={handleModuleConfirm}
        onCancel={() => setModuleControlVisible(false)}
      />

      <MaintenanceModeModal
        visible={maintenanceModeVisible}
        currentMode={maintenanceMode}
        onConfirm={handleMaintenanceMode}
        onCancel={() => setMaintenanceModeVisible(false)}
      />

      <OperationLogModal
        visible={operationLogVisible}
        onClose={() => setOperationLogVisible(false)}
      />
    </div>
  );
};

export default SystemControlPanel;