import React, { useState } from 'react';
import {
  Card,
  Table,
  Button,
  Space,
  Modal,
  message,
  Popconfirm,
  Tag,
  Drawer,
  Descriptions,
  Alert,
  Timeline,
  Typography,
  Row,
  Col,
  Switch,
  Tooltip,
  Input,
  Select,
} from 'antd';
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  EditOutlined,
  DeleteOutlined,
  DownloadOutlined,
  UploadOutlined,
  SettingOutlined,
  FileTextOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
} from '@ant-design/icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '../../services/api';
import { StrategyConfig } from '../../services/types';
import StrategyConfigForm from './StrategyConfigForm';

const { Text, Paragraph } = Typography;
const { TextArea } = Input;
const { Option } = Select;

interface StrategyOperationsProps {
  onEditStrategy?: (strategyName: string) => void;
}

interface StrategyWithStatus extends StrategyConfig {
  status: 'running' | 'stopped' | 'error' | 'starting' | 'stopping';
  lastUpdate: string;
  errorMessage?: string;
  runningTime?: number;
}

// 模拟策略日志数据
const generateMockLogs = (strategyName: string) => [
  {
    timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
    level: 'INFO',
    message: `策略 ${strategyName} 生成买入信号: 000001.SZ @ ¥12.45`,
  },
  {
    timestamp: new Date(Date.now() - 10 * 60 * 1000).toISOString(),
    level: 'INFO',
    message: `策略 ${strategyName} 技术指标计算完成`,
  },
  {
    timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
    level: 'WARN',
    message: `策略 ${strategyName} 风险检查: 仓位接近上限`,
  },
  {
    timestamp: new Date(Date.now() - 20 * 60 * 1000).toISOString(),
    level: 'INFO',
    message: `策略 ${strategyName} 启动成功`,
  },
];

const StrategyOperations: React.FC<StrategyOperationsProps> = ({ onEditStrategy }) => {
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [logDrawerVisible, setLogDrawerVisible] = useState(false);
  const [backupModalVisible, setBackupModalVisible] = useState(false);
  const [selectedLogs, setSelectedLogs] = useState<any[]>([]);
  const [logLevel, setLogLevel] = useState<string>('ALL');

  const queryClient = useQueryClient();

  // 获取策略列表
  const { data: strategyList, isLoading } = useQuery({
    queryKey: ['strategyList'],
    queryFn: () => api.strategy.getStrategies(),
    refetchInterval: 5000,
  });

  // 获取策略性能数据
  const { data: strategyPerformance } = useQuery({
    queryKey: ['strategyPerformance'],
    queryFn: () => api.strategy.getPerformance(),
    refetchInterval: 5000,
  });

  // 启动策略
  const startStrategyMutation = useMutation({
    mutationFn: (strategyName: string) => api.strategy.startStrategy(strategyName),
    onSuccess: (_, strategyName) => {
      message.success(`策略 ${strategyName} 启动成功`);
      queryClient.invalidateQueries({ queryKey: ['strategyList'] });
      queryClient.invalidateQueries({ queryKey: ['strategyPerformance'] });
    },
    onError: (error: any, strategyName) => {
      message.error(`策略 ${strategyName} 启动失败: ${error.message}`);
    },
  });

  // 停止策略
  const stopStrategyMutation = useMutation({
    mutationFn: (strategyName: string) => api.strategy.stopStrategy(strategyName),
    onSuccess: (_, strategyName) => {
      message.success(`策略 ${strategyName} 已停止`);
      queryClient.invalidateQueries({ queryKey: ['strategyList'] });
      queryClient.invalidateQueries({ queryKey: ['strategyPerformance'] });
    },
    onError: (error: any, strategyName) => {
      message.error(`策略 ${strategyName} 停止失败: ${error.message}`);
    },
  });

  // 删除策略
  const deleteStrategyMutation = useMutation({
    mutationFn: (strategyName: string) => api.strategy.deleteStrategy(strategyName),
    onSuccess: (_, strategyName) => {
      message.success(`策略 ${strategyName} 已删除`);
      queryClient.invalidateQueries({ queryKey: ['strategyList'] });
      queryClient.invalidateQueries({ queryKey: ['strategyPerformance'] });
    },
    onError: (error: any, strategyName) => {
      message.error(`策略 ${strategyName} 删除失败: ${error.message}`);
    },
  });

  // 获取策略配置
  const { data: strategyConfig } = useQuery({
    queryKey: ['strategyConfig', selectedStrategy],
    queryFn: () => selectedStrategy ? api.strategy.getStrategyConfig(selectedStrategy) : null,
    enabled: !!selectedStrategy,
  });

  // 生成策略状态数据
  const strategiesWithStatus: StrategyWithStatus[] = (strategyList || []).map(name => {
    const performance = strategyPerformance?.[name];
    const isActive = performance && performance.signalsGenerated > 0;
    
    return {
      name,
      symbols: ['000001.SZ', '600000.SH'], // 模拟数据
      initialCapital: 100000,
      riskTolerance: 0.02,
      maxPositionSize: 0.1,
      indicators: {},
      parameters: {},
      status: isActive ? 'running' : 'stopped',
      lastUpdate: new Date().toISOString(),
      runningTime: isActive ? Math.floor(Math.random() * 86400) : 0,
    };
  });

  // 处理策略启动/停止
  const handleToggleStrategy = (strategyName: string, currentStatus: string) => {
    if (currentStatus === 'running') {
      stopStrategyMutation.mutate(strategyName);
    } else {
      startStrategyMutation.mutate(strategyName);
    }
  };

  // 处理策略编辑
  const handleEditStrategy = (strategyName: string) => {
    setSelectedStrategy(strategyName);
    setEditModalVisible(true);
    onEditStrategy?.(strategyName);
  };

  // 处理查看日志
  const handleViewLogs = (strategyName: string) => {
    setSelectedStrategy(strategyName);
    setSelectedLogs(generateMockLogs(strategyName));
    setLogDrawerVisible(true);
  };

  // 处理策略备份
  const handleBackupStrategy = (strategyName: string) => {
    const strategy = strategiesWithStatus.find(s => s.name === strategyName);
    if (strategy) {
      const backupData = {
        name: strategy.name,
        config: strategy,
        performance: strategyPerformance?.[strategyName],
        timestamp: new Date().toISOString(),
      };
      
      const blob = new Blob([JSON.stringify(backupData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `strategy_${strategyName}_backup_${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      message.success(`策略 ${strategyName} 备份已下载`);
    }
  };

  // 格式化运行时间
  const formatRunningTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // 表格列定义
  const columns = [
    {
      title: '策略名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: StrategyWithStatus) => (
        <Space direction="vertical" size="small">
          <Text strong>{name}</Text>
          <Space>
            <Tag color={
              record.status === 'running' ? 'green' :
              record.status === 'error' ? 'red' :
              record.status === 'starting' || record.status === 'stopping' ? 'orange' :
              'default'
            }>
              {record.status === 'running' ? '运行中' :
               record.status === 'error' ? '异常' :
               record.status === 'starting' ? '启动中' :
               record.status === 'stopping' ? '停止中' :
               '已停止'}
            </Tag>
            {record.status === 'running' && (
              <Text type="secondary" style={{ fontSize: '12px' }}>
                运行时间: {formatRunningTime(record.runningTime || 0)}
              </Text>
            )}
          </Space>
        </Space>
      ),
    },
    {
      title: '交易标的',
      dataIndex: 'symbols',
      key: 'symbols',
      render: (symbols: string[]) => (
        <Space wrap>
          {symbols.slice(0, 3).map(symbol => (
            <Tag key={symbol}>{symbol}</Tag>
          ))}
          {symbols.length > 3 && <Tag>+{symbols.length - 3}</Tag>}
        </Space>
      ),
    },
    {
      title: '初始资金',
      dataIndex: 'initialCapital',
      key: 'initialCapital',
      render: (value: number) => `¥${value.toLocaleString()}`,
    },
    {
      title: '风险容忍度',
      dataIndex: 'riskTolerance',
      key: 'riskTolerance',
      render: (value: number) => `${(value * 100).toFixed(1)}%`,
    },
    {
      title: '最大仓位',
      dataIndex: 'maxPositionSize',
      key: 'maxPositionSize',
      render: (value: number) => `${(value * 100).toFixed(1)}%`,
    },
    {
      title: '最后更新',
      dataIndex: 'lastUpdate',
      key: 'lastUpdate',
      render: (value: string) => new Date(value).toLocaleString(),
    },
    {
      title: '操作',
      key: 'actions',
      width: 300,
      render: (_, record: StrategyWithStatus) => (
        <Space wrap>
          <Tooltip title={record.status === 'running' ? '停止策略' : '启动策略'}>
            <Button
              type={record.status === 'running' ? 'default' : 'primary'}
              size="small"
              icon={record.status === 'running' ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
              loading={
                (record.status === 'starting' || record.status === 'stopping') ||
                startStrategyMutation.isPending || stopStrategyMutation.isPending
              }
              onClick={() => handleToggleStrategy(record.name, record.status)}
            >
              {record.status === 'running' ? '停止' : '启动'}
            </Button>
          </Tooltip>
          
          <Tooltip title="编辑策略">
            <Button
              size="small"
              icon={<EditOutlined />}
              onClick={() => handleEditStrategy(record.name)}
            >
              编辑
            </Button>
          </Tooltip>
          
          <Tooltip title="查看日志">
            <Button
              size="small"
              icon={<FileTextOutlined />}
              onClick={() => handleViewLogs(record.name)}
            >
              日志
            </Button>
          </Tooltip>
          
          <Tooltip title="备份策略">
            <Button
              size="small"
              icon={<DownloadOutlined />}
              onClick={() => handleBackupStrategy(record.name)}
            >
              备份
            </Button>
          </Tooltip>
          
          <Popconfirm
            title="确定要删除这个策略吗？"
            description="删除后将无法恢复，请确认操作。"
            onConfirm={() => deleteStrategyMutation.mutate(record.name)}
            okText="确定"
            cancelText="取消"
            icon={<ExclamationCircleOutlined style={{ color: 'red' }} />}
          >
            <Tooltip title="删除策略">
              <Button
                size="small"
                danger
                icon={<DeleteOutlined />}
                loading={deleteStrategyMutation.isPending}
              >
                删除
              </Button>
            </Tooltip>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <Card
        title={
          <Space>
            <SettingOutlined />
            策略操作管理
          </Space>
        }
        extra={
          <Space>
            <Button icon={<UploadOutlined />}>
              导入策略
            </Button>
            <Button icon={<SyncOutlined />} onClick={() => queryClient.invalidateQueries()}>
              刷新
            </Button>
          </Space>
        }
      >
        <Table
          columns={columns}
          dataSource={strategiesWithStatus}
          rowKey="name"
          loading={isLoading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 个策略`,
          }}
          expandable={{
            expandedRowRender: (record) => (
              <Descriptions column={2} size="small">
                <Descriptions.Item label="策略状态">
                  <Space>
                    {record.status === 'running' ? (
                      <CheckCircleOutlined style={{ color: '#52c41a' }} />
                    ) : record.status === 'error' ? (
                      <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
                    ) : (
                      <PauseCircleOutlined style={{ color: '#d9d9d9' }} />
                    )}
                    {record.status === 'running' ? '正常运行' :
                     record.status === 'error' ? '运行异常' : '已停止'}
                  </Space>
                </Descriptions.Item>
                <Descriptions.Item label="错误信息">
                  {record.errorMessage || '无'}
                </Descriptions.Item>
                <Descriptions.Item label="技术指标">
                  {Object.keys(record.indicators).length > 0 ? 
                    Object.keys(record.indicators).join(', ') : '无'}
                </Descriptions.Item>
                <Descriptions.Item label="策略参数">
                  {Object.keys(record.parameters || {}).length > 0 ? 
                    Object.keys(record.parameters || {}).join(', ') : '无'}
                </Descriptions.Item>
              </Descriptions>
            ),
            rowExpandable: () => true,
          }}
        />
      </Card>

      {/* 编辑策略模态框 */}
      <Modal
        title="编辑策略"
        open={editModalVisible}
        onCancel={() => {
          setEditModalVisible(false);
          setSelectedStrategy(null);
        }}
        footer={null}
        width={800}
        destroyOnClose
      >
        {selectedStrategy && strategyConfig && (
          <StrategyConfigForm
            mode="edit"
            strategyName={selectedStrategy}
            initialValues={strategyConfig}
            onSuccess={() => {
              setEditModalVisible(false);
              setSelectedStrategy(null);
              queryClient.invalidateQueries({ queryKey: ['strategyList'] });
            }}
          />
        )}
      </Modal>

      {/* 策略日志抽屉 */}
      <Drawer
        title={
          <Space>
            <FileTextOutlined />
            策略运行日志 - {selectedStrategy}
          </Space>
        }
        placement="right"
        width={600}
        open={logDrawerVisible}
        onClose={() => {
          setLogDrawerVisible(false);
          setSelectedStrategy(null);
          setSelectedLogs([]);
        }}
        extra={
          <Space>
            <Select
              value={logLevel}
              onChange={setLogLevel}
              style={{ width: 100 }}
            >
              <Option value="ALL">全部</Option>
              <Option value="INFO">信息</Option>
              <Option value="WARN">警告</Option>
              <Option value="ERROR">错误</Option>
            </Select>
            <Button size="small">清空日志</Button>
            <Button size="small" icon={<DownloadOutlined />}>
              导出日志
            </Button>
          </Space>
        }
      >
        <Timeline
          items={selectedLogs
            .filter(log => logLevel === 'ALL' || log.level === logLevel)
            .map(log => ({
              color: log.level === 'ERROR' ? 'red' : log.level === 'WARN' ? 'orange' : 'blue',
              children: (
                <div>
                  <div style={{ fontSize: '12px', color: '#666' }}>
                    {new Date(log.timestamp).toLocaleString()}
                  </div>
                  <div>
                    <Tag color={
                      log.level === 'ERROR' ? 'red' :
                      log.level === 'WARN' ? 'orange' : 'blue'
                    }>
                      {log.level}
                    </Tag>
                    {log.message}
                  </div>
                </div>
              ),
            }))}
        />
        
        {selectedLogs.length === 0 && (
          <Alert
            message="暂无日志数据"
            description="策略还没有生成日志记录，或者日志级别过滤后没有匹配的记录。"
            type="info"
            showIcon
          />
        )}
      </Drawer>

      {/* 备份恢复模态框 */}
      <Modal
        title="策略备份与恢复"
        open={backupModalVisible}
        onCancel={() => setBackupModalVisible(false)}
        footer={null}
        width={600}
      >
        <Row gutter={16}>
          <Col span={12}>
            <Card title="备份策略" size="small">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Button block icon={<DownloadOutlined />}>
                  备份所有策略
                </Button>
                <Button block icon={<DownloadOutlined />}>
                  备份选中策略
                </Button>
              </Space>
            </Card>
          </Col>
          <Col span={12}>
            <Card title="恢复策略" size="small">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Button block icon={<UploadOutlined />}>
                  从文件恢复
                </Button>
                <Button block icon={<UploadOutlined />}>
                  从云端恢复
                </Button>
              </Space>
            </Card>
          </Col>
        </Row>
      </Modal>
    </div>
  );
};

export default StrategyOperations;