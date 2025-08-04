import React, { useState } from 'react';
import {
  Card,
  Row,
  Col,
  Descriptions,
  Button,
  Space,
  Modal,
  Form,
  Input,
  InputNumber,
  Switch,
  Select,
  Table,
  Tag,
  Alert,
  Typography,
  Divider,
  Tooltip,
  message,
  Popconfirm,
} from 'antd';
import {
  SettingOutlined,
  EditOutlined,
  SaveOutlined,
  ReloadOutlined,
  HistoryOutlined,
  DownloadOutlined,
  UploadOutlined,
  InfoCircleOutlined,
  WarningOutlined,
} from '@ant-design/icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../../services/api';
import { formatDateTime, formatCurrency, formatPercent } from '../../utils/format';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

interface SystemConfig {
  trading: {
    initialCapital: number;
    commissionRate: number;
    slippageRate: number;
    maxPositionSize: number;
    maxDailyLoss: number;
    maxDrawdownLimit: number;
  };
  data: {
    dataSource: string;
    updateInterval: number;
    cacheSize: number;
    retentionDays: number;
  };
  strategy: {
    maxStrategies: number;
    defaultRiskTolerance: number;
    signalTimeout: number;
    backtestPeriod: number;
  };
  system: {
    logLevel: string;
    maxLogSize: number;
    backupInterval: number;
    healthCheckInterval: number;
  };
  performance: {
    enableOptimization: boolean;
    cacheEnabled: boolean;
    batchSize: number;
    threadPoolSize: number;
  };
}

interface ConfigHistory {
  id: string;
  timestamp: string;
  version: string;
  changes: string[];
  author: string;
  description: string;
}

interface ConfigBackup {
  id: string;
  timestamp: string;
  filename: string;
  size: number;
  description: string;
}

const ConfigSection: React.FC<{
  title: string;
  config: Record<string, any>;
  onEdit: (section: string, config: Record<string, any>) => void;
  section: string;
}> = ({ title, config, onEdit, section }) => {
  const getConfigDescription = (key: string, value: any) => {
    const descriptions: Record<string, string> = {
      initialCapital: '系统初始资金，用于策略交易',
      commissionRate: '交易佣金费率，影响交易成本计算',
      slippageRate: '滑点率，模拟实际交易中的价格偏差',
      maxPositionSize: '单个持仓的最大比例限制',
      maxDailyLoss: '单日最大损失限制，触发后停止交易',
      maxDrawdownLimit: '最大回撤限制，用于风险控制',
      dataSource: '市场数据来源配置',
      updateInterval: '数据更新间隔（毫秒）',
      cacheSize: '数据缓存大小（MB）',
      retentionDays: '数据保留天数',
      maxStrategies: '系统支持的最大策略数量',
      defaultRiskTolerance: '默认风险容忍度',
      signalTimeout: '交易信号超时时间（秒）',
      backtestPeriod: '默认回测周期（天）',
      logLevel: '系统日志级别',
      maxLogSize: '单个日志文件最大大小（MB）',
      backupInterval: '自动备份间隔（小时）',
      healthCheckInterval: '健康检查间隔（秒）',
      enableOptimization: '启用性能优化',
      cacheEnabled: '启用缓存机制',
      batchSize: '批处理大小',
      threadPoolSize: '线程池大小',
    };
    return descriptions[key] || '配置参数';
  };

  const formatConfigValue = (key: string, value: any) => {
    if (typeof value === 'boolean') {
      return value ? '启用' : '禁用';
    }
    if (key.includes('Rate') || key.includes('Tolerance')) {
      return formatPercent(value);
    }
    if (key.includes('Capital') || key.includes('Loss')) {
      return formatCurrency(value);
    }
    if (key.includes('Interval') || key.includes('Timeout') || key.includes('Period')) {
      return `${value} ${key.includes('Interval') || key.includes('Timeout') ? '秒' : '天'}`;
    }
    if (key.includes('Size') && typeof value === 'number') {
      return `${value} MB`;
    }
    return String(value);
  };

  return (
    <Card
      title={title}
      extra={
        <Button
          icon={<EditOutlined />}
          size="small"
          onClick={() => onEdit(section, config)}
        >
          编辑
        </Button>
      }
    >
      <Descriptions column={1} size="small">
        {Object.entries(config).map(([key, value]) => (
          <Descriptions.Item
            key={key}
            label={
              <Tooltip title={getConfigDescription(key, value)}>
                <span>
                  {key} <InfoCircleOutlined style={{ color: '#1890ff' }} />
                </span>
              </Tooltip>
            }
          >
            <Text strong>{formatConfigValue(key, value)}</Text>
          </Descriptions.Item>
        ))}
      </Descriptions>
    </Card>
  );
};

const ConfigEditModal: React.FC<{
  visible: boolean;
  section: string;
  config: Record<string, any>;
  onSave: (section: string, config: Record<string, any>) => void;
  onCancel: () => void;
}> = ({ visible, section, config, onSave, onCancel }) => {
  const [form] = Form.useForm();

  const handleSave = async () => {
    try {
      const values = await form.validateFields();
      onSave(section, values);
    } catch (error) {
      console.error('Form validation failed:', error);
    }
  };

  const getFormItem = (key: string, value: any) => {
    if (typeof value === 'boolean') {
      return (
        <Form.Item
          name={key}
          label={key}
          valuePropName="checked"
          initialValue={value}
        >
          <Switch />
        </Form.Item>
      );
    }

    if (typeof value === 'number') {
      const isPercentage = key.includes('Rate') || key.includes('Tolerance');
      const isCurrency = key.includes('Capital') || key.includes('Loss');
      
      return (
        <Form.Item
          name={key}
          label={key}
          initialValue={value}
          rules={[{ required: true, message: `请输入${key}` }]}
        >
          <InputNumber
            style={{ width: '100%' }}
            min={0}
            step={isPercentage ? 0.001 : isCurrency ? 1000 : 1}
            formatter={
              isPercentage
                ? (value) => `${((value || 0) * 100).toFixed(1)}%`
                : isCurrency
                ? (value) => `¥ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')
                : undefined
            }
            parser={
              isPercentage
                ? (value) => (value!.replace('%', '') as any) / 100
                : isCurrency
                ? (value) => value!.replace(/¥\s?|(,*)/g, '') as any
                : undefined
            }
          />
        </Form.Item>
      );
    }

    if (key === 'logLevel') {
      return (
        <Form.Item
          name={key}
          label={key}
          initialValue={value}
          rules={[{ required: true, message: `请选择${key}` }]}
        >
          <Select>
            <Option value="DEBUG">DEBUG</Option>
            <Option value="INFO">INFO</Option>
            <Option value="WARNING">WARNING</Option>
            <Option value="ERROR">ERROR</Option>
            <Option value="CRITICAL">CRITICAL</Option>
          </Select>
        </Form.Item>
      );
    }

    if (key === 'dataSource') {
      return (
        <Form.Item
          name={key}
          label={key}
          initialValue={value}
          rules={[{ required: true, message: `请选择${key}` }]}
        >
          <Select>
            <Option value="tushare">Tushare</Option>
            <Option value="akshare">AKShare</Option>
            <Option value="yahoo">Yahoo Finance</Option>
            <Option value="local">本地数据</Option>
          </Select>
        </Form.Item>
      );
    }

    return (
      <Form.Item
        name={key}
        label={key}
        initialValue={value}
        rules={[{ required: true, message: `请输入${key}` }]}
      >
        <Input />
      </Form.Item>
    );
  };

  return (
    <Modal
      title={`编辑 ${section} 配置`}
      open={visible}
      onOk={handleSave}
      onCancel={onCancel}
      width={600}
      okText="保存"
      cancelText="取消"
    >
      <Form form={form} layout="vertical">
        {Object.entries(config).map(([key, value]) => (
          <div key={key}>{getFormItem(key, value)}</div>
        ))}
      </Form>
    </Modal>
  );
};

const ConfigHistoryModal: React.FC<{
  visible: boolean;
  onClose: () => void;
  onRestore: (configId: string) => void;
}> = ({ visible, onClose, onRestore }) => {
  // Mock history data - in real implementation, this would come from API
  const historyData: ConfigHistory[] = [
    {
      id: '1',
      timestamp: '2024-01-15 14:30:00',
      version: '1.2.3',
      changes: ['更新交易佣金费率', '调整最大仓位限制'],
      author: 'admin',
      description: '优化交易参数配置',
    },
    {
      id: '2',
      timestamp: '2024-01-14 09:15:00',
      version: '1.2.2',
      changes: ['增加数据缓存大小', '调整健康检查间隔'],
      author: 'admin',
      description: '性能优化配置更新',
    },
    {
      id: '3',
      timestamp: '2024-01-13 16:45:00',
      version: '1.2.1',
      changes: ['修改日志级别', '更新备份间隔'],
      author: 'admin',
      description: '系统维护配置调整',
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
      title: '版本',
      dataIndex: 'version',
      key: 'version',
      render: (version: string) => <Tag color="blue">{version}</Tag>,
    },
    {
      title: '变更内容',
      dataIndex: 'changes',
      key: 'changes',
      render: (changes: string[]) => (
        <div>
          {changes.map((change, index) => (
            <Tag key={index} color="green" style={{ marginBottom: 4 }}>
              {change}
            </Tag>
          ))}
        </div>
      ),
    },
    {
      title: '操作者',
      dataIndex: 'author',
      key: 'author',
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: ConfigHistory) => (
        <Popconfirm
          title="确定要恢复到此配置版本吗？"
          onConfirm={() => onRestore(record.id)}
          okText="确定"
          cancelText="取消"
        >
          <Button size="small" type="link">
            恢复
          </Button>
        </Popconfirm>
      ),
    },
  ];

  return (
    <Modal
      title="配置历史"
      open={visible}
      onCancel={onClose}
      footer={null}
      width={800}
    >
      <Table
        columns={columns}
        dataSource={historyData}
        rowKey="id"
        pagination={{ pageSize: 10 }}
        size="small"
      />
    </Modal>
  );
};

const ConfigBackupModal: React.FC<{
  visible: boolean;
  onClose: () => void;
  onBackup: () => void;
  onRestore: (backupId: string) => void;
}> = ({ visible, onClose, onBackup, onRestore }) => {
  // Mock backup data - in real implementation, this would come from API
  const backupData: ConfigBackup[] = [
    {
      id: '1',
      timestamp: '2024-01-15 14:30:00',
      filename: 'config_backup_20240115_143000.json',
      size: 2048,
      description: '定期自动备份',
    },
    {
      id: '2',
      timestamp: '2024-01-14 14:30:00',
      filename: 'config_backup_20240114_143000.json',
      size: 2045,
      description: '定期自动备份',
    },
    {
      id: '3',
      timestamp: '2024-01-13 14:30:00',
      filename: 'config_backup_20240113_143000.json',
      size: 2040,
      description: '手动备份',
    },
  ];

  const columns = [
    {
      title: '备份时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: string) => formatDateTime(timestamp),
    },
    {
      title: '文件名',
      dataIndex: 'filename',
      key: 'filename',
      render: (filename: string) => <Text code>{filename}</Text>,
    },
    {
      title: '大小',
      dataIndex: 'size',
      key: 'size',
      render: (size: number) => `${(size / 1024).toFixed(2)} KB`,
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: ConfigBackup) => (
        <Space>
          <Button size="small" type="link" icon={<DownloadOutlined />}>
            下载
          </Button>
          <Popconfirm
            title="确定要从此备份恢复配置吗？"
            onConfirm={() => onRestore(record.id)}
            okText="确定"
            cancelText="取消"
          >
            <Button size="small" type="link">
              恢复
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <Modal
      title="配置备份管理"
      open={visible}
      onCancel={onClose}
      footer={
        <Space>
          <Button onClick={onClose}>关闭</Button>
          <Button type="primary" icon={<SaveOutlined />} onClick={onBackup}>
            创建备份
          </Button>
        </Space>
      }
      width={800}
    >
      <Table
        columns={columns}
        dataSource={backupData}
        rowKey="id"
        pagination={{ pageSize: 10 }}
        size="small"
      />
    </Modal>
  );
};

const SystemConfigManager: React.FC = () => {
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [historyModalVisible, setHistoryModalVisible] = useState(false);
  const [backupModalVisible, setBackupModalVisible] = useState(false);
  const [currentSection, setCurrentSection] = useState<string>('');
  const [currentConfig, setCurrentConfig] = useState<Record<string, any>>({});

  const queryClient = useQueryClient();

  // Mock system configuration - in real implementation, this would come from API
  const systemConfig: SystemConfig = {
    trading: {
      initialCapital: 1000000,
      commissionRate: 0.0003,
      slippageRate: 0.0001,
      maxPositionSize: 0.2,
      maxDailyLoss: 50000,
      maxDrawdownLimit: 0.1,
    },
    data: {
      dataSource: 'tushare',
      updateInterval: 1000,
      cacheSize: 512,
      retentionDays: 365,
    },
    strategy: {
      maxStrategies: 10,
      defaultRiskTolerance: 0.05,
      signalTimeout: 300,
      backtestPeriod: 252,
    },
    system: {
      logLevel: 'INFO',
      maxLogSize: 100,
      backupInterval: 24,
      healthCheckInterval: 30,
    },
    performance: {
      enableOptimization: true,
      cacheEnabled: true,
      batchSize: 1000,
      threadPoolSize: 4,
    },
  };

  const updateConfigMutation = useMutation({
    mutationFn: async ({ section, config }: { section: string; config: Record<string, any> }) => {
      // In real implementation, this would call the API
      await new Promise(resolve => setTimeout(resolve, 1000));
      return { success: true };
    },
    onSuccess: () => {
      message.success('配置更新成功');
      setEditModalVisible(false);
      queryClient.invalidateQueries({ queryKey: ['systemConfig'] });
    },
    onError: (error: any) => {
      message.error(`配置更新失败: ${error.message}`);
    },
  });

  const handleEdit = (section: string, config: Record<string, any>) => {
    setCurrentSection(section);
    setCurrentConfig(config);
    setEditModalVisible(true);
  };

  const handleSave = (section: string, config: Record<string, any>) => {
    updateConfigMutation.mutate({ section, config });
  };

  const handleRestore = (configId: string) => {
    message.success(`配置已恢复到版本 ${configId}`);
    setHistoryModalVisible(false);
  };

  const handleBackup = () => {
    message.success('配置备份创建成功');
  };

  const handleRestoreFromBackup = (backupId: string) => {
    message.success(`配置已从备份 ${backupId} 恢复`);
    setBackupModalVisible(false);
  };

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>系统配置管理</Title>
        <Space>
          <Button icon={<HistoryOutlined />} onClick={() => setHistoryModalVisible(true)}>
            配置历史
          </Button>
          <Button icon={<SaveOutlined />} onClick={() => setBackupModalVisible(true)}>
            备份管理
          </Button>
          <Button icon={<ReloadOutlined />} type="primary">
            重新加载配置
          </Button>
        </Space>
      </div>

      <Alert
        message="配置修改提醒"
        description="修改系统配置可能会影响系统运行，请谨慎操作。建议在修改前创建配置备份。"
        type="warning"
        icon={<WarningOutlined />}
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Row gutter={[16, 16]}>
        <Col span={12}>
          <ConfigSection
            title="交易配置"
            config={systemConfig.trading}
            onEdit={handleEdit}
            section="trading"
          />
        </Col>
        <Col span={12}>
          <ConfigSection
            title="数据配置"
            config={systemConfig.data}
            onEdit={handleEdit}
            section="data"
          />
        </Col>
        <Col span={12}>
          <ConfigSection
            title="策略配置"
            config={systemConfig.strategy}
            onEdit={handleEdit}
            section="strategy"
          />
        </Col>
        <Col span={12}>
          <ConfigSection
            title="系统配置"
            config={systemConfig.system}
            onEdit={handleEdit}
            section="system"
          />
        </Col>
        <Col span={24}>
          <ConfigSection
            title="性能配置"
            config={systemConfig.performance}
            onEdit={handleEdit}
            section="performance"
          />
        </Col>
      </Row>

      <ConfigEditModal
        visible={editModalVisible}
        section={currentSection}
        config={currentConfig}
        onSave={handleSave}
        onCancel={() => setEditModalVisible(false)}
      />

      <ConfigHistoryModal
        visible={historyModalVisible}
        onClose={() => setHistoryModalVisible(false)}
        onRestore={handleRestore}
      />

      <ConfigBackupModal
        visible={backupModalVisible}
        onClose={() => setBackupModalVisible(false)}
        onBackup={handleBackup}
        onRestore={handleRestoreFromBackup}
      />
    </div>
  );
};

export default SystemConfigManager;