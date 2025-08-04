import React, { useState, useCallback, useRef } from 'react';
import { Card, Button, Space, Modal, Form, Select, InputNumber, message, Drawer, List, Badge } from 'antd';
import { PlusOutlined, SaveOutlined, PlayCircleOutlined, DeleteOutlined, SettingOutlined } from '@ant-design/icons';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Handle,
  Position,
  NodeProps,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { strategyApi } from '@/api/strategy';

const { Option } = Select;

// 块类型定义
enum BlockType {
  DATA = 'data',
  INDICATOR = 'indicator',
  CONDITION = 'condition',
  ACTION = 'action',
}

// 块类型配置
const BLOCK_TYPES = {
  [BlockType.DATA]: {
    label: '数据源',
    color: '#1890ff',
    icon: '📊',
  },
  [BlockType.INDICATOR]: {
    label: '技术指标',
    color: '#52c41a',
    icon: '📈',
  },
  [BlockType.CONDITION]: {
    label: '条件判断',
    color: '#faad14',
    icon: '🔀',
  },
  [BlockType.ACTION]: {
    label: '交易动作',
    color: '#f5222d',
    icon: '🎯',
  },
};

// 支持的指标
const INDICATORS = [
  { value: 'SMA', label: '简单移动平均线' },
  { value: 'EMA', label: '指数移动平均线' },
  { value: 'RSI', label: 'RSI相对强弱指标' },
  { value: 'MACD', label: 'MACD' },
  { value: 'BOLL', label: '布林带' },
  { value: 'KDJ', label: 'KDJ指标' },
  { value: 'ATR', label: '平均真实波幅' },
  { value: 'VOL', label: '成交量' },
];

// 操作符
const OPERATORS = [
  { value: '>', label: '大于' },
  { value: '<', label: '小于' },
  { value: '>=', label: '大于等于' },
  { value: '<=', label: '小于等于' },
  { value: '==', label: '等于' },
  { value: '!=', label: '不等于' },
  { value: 'cross_above', label: '上穿' },
  { value: 'cross_below', label: '下穿' },
];

// 自定义节点组件
const CustomNode: React.FC<NodeProps> = ({ data, selected }) => {
  const blockConfig = BLOCK_TYPES[data.blockType as BlockType];

  return (
    <div
      style={{
        background: selected ? blockConfig.color : '#fff',
        border: `2px solid ${blockConfig.color}`,
        borderRadius: 8,
        padding: '10px 15px',
        minWidth: 150,
        color: selected ? '#fff' : '#000',
      }}
    >
      <div style={{ fontWeight: 'bold', marginBottom: 5 }}>
        <span style={{ marginRight: 5 }}>{blockConfig.icon}</span>
        {data.label}
      </div>
      {data.subLabel && (
        <div style={{ fontSize: 12, opacity: 0.8 }}>{data.subLabel}</div>
      )}
      
      {/* 输入句柄 */}
      {data.blockType !== BlockType.DATA && (
        <Handle
          type="target"
          position={Position.Left}
          style={{ background: '#555' }}
        />
      )}
      
      {/* 输出句柄 */}
      {data.blockType !== BlockType.ACTION && (
        <Handle
          type="source"
          position={Position.Right}
          style={{ background: '#555' }}
        />
      )}
      
      {/* 条件块需要两个输入 */}
      {data.blockType === BlockType.CONDITION && (
        <Handle
          type="target"
          position={Position.Left}
          id="input2"
          style={{ background: '#555', top: '70%' }}
        />
      )}
    </div>
  );
};

const nodeTypes = {
  custom: CustomNode,
};

interface VisualStrategyBuilderProps {
  onSave?: (strategy: any) => void;
}

const VisualStrategyBuilder: React.FC<VisualStrategyBuilderProps> = ({ onSave }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [blockDrawerVisible, setBlockDrawerVisible] = useState(false);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [configModalVisible, setConfigModalVisible] = useState(false);
  const [form] = Form.useForm();
  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  // 连接验证
  const isValidConnection = useCallback((connection: Connection) => {
    const sourceNode = nodes.find(n => n.id === connection.source);
    const targetNode = nodes.find(n => n.id === connection.target);
    
    if (!sourceNode || !targetNode) return false;
    
    // 动作块只能作为终点
    if (sourceNode.data.blockType === BlockType.ACTION) return false;
    
    // 数据块只能作为起点
    if (targetNode.data.blockType === BlockType.DATA) return false;
    
    return true;
  }, [nodes]);

  // 添加连接
  const onConnect = useCallback(
    (params: Connection) => {
      if (isValidConnection(params)) {
        setEdges((eds) => addEdge(params, eds));
      } else {
        message.warning('无效的连接');
      }
    },
    [setEdges, isValidConnection]
  );

  // 添加新块
  const addBlock = (type: BlockType, config: any = {}) => {
    const id = `${type}_${Date.now()}`;
    const blockTypeConfig = BLOCK_TYPES[type];
    
    const newNode: Node = {
      id,
      type: 'custom',
      position: { x: 250, y: nodes.length * 100 + 50 },
      data: {
        label: config.label || blockTypeConfig.label,
        subLabel: config.subLabel || '',
        blockType: type,
        params: config.params || {},
      },
    };
    
    setNodes((nds) => [...nds, newNode]);
    setBlockDrawerVisible(false);
  };

  // 配置节点
  const configureNode = (values: any) => {
    if (!selectedNode) return;
    
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === selectedNode.id) {
          return {
            ...node,
            data: {
              ...node.data,
              params: values,
              subLabel: getNodeSubLabel(node.data.blockType, values),
            },
          };
        }
        return node;
      })
    );
    
    setConfigModalVisible(false);
    message.success('配置已更新');
  };

  // 获取节点副标签
  const getNodeSubLabel = (blockType: BlockType, params: any): string => {
    switch (blockType) {
      case BlockType.DATA:
        return params.field || 'close';
      case BlockType.INDICATOR:
        const indicator = INDICATORS.find(i => i.value === params.type);
        return indicator ? indicator.label : params.type;
      case BlockType.CONDITION:
        const operator = OPERATORS.find(o => o.value === params.operator);
        return operator ? operator.label : params.operator;
      case BlockType.ACTION:
        return params.type === 'buy' ? '买入' : params.type === 'sell' ? '卖出' : '平仓';
      default:
        return '';
    }
  };

  // 删除选中的节点
  const deleteSelectedNode = () => {
    if (!selectedNode) return;
    
    setNodes((nds) => nds.filter((n) => n.id !== selectedNode.id));
    setEdges((eds) => eds.filter((e) => e.source !== selectedNode.id && e.target !== selectedNode.id));
    setSelectedNode(null);
  };

  // 保存策略
  const saveStrategy = async () => {
    if (nodes.length === 0) {
      message.warning('请先添加策略块');
      return;
    }
    
    // 验证是否有动作块
    const hasAction = nodes.some(n => n.data.blockType === BlockType.ACTION);
    if (!hasAction) {
      message.warning('策略必须包含至少一个交易动作');
      return;
    }
    
    const strategyData = {
      nodes: nodes.map(n => ({
        id: n.id,
        type: n.data.blockType,
        params: n.data.params,
        position: n.position,
      })),
      edges: edges.map(e => ({
        source: e.source,
        target: e.target,
        targetHandle: e.targetHandle,
      })),
    };
    
    try {
      await strategyApi.saveVisualStrategy(strategyData);
      message.success('策略保存成功');
      onSave?.(strategyData);
    } catch (error) {
      message.error('策略保存失败');
    }
  };

  // 测试运行策略
  const testStrategy = async () => {
    message.info('策略测试功能开发中...');
  };

  return (
    <Card
      title="可视化策略构建器"
      extra={
        <Space>
          <Button icon={<PlusOutlined />} onClick={() => setBlockDrawerVisible(true)}>
            添加模块
          </Button>
          <Button icon={<SaveOutlined />} type="primary" onClick={saveStrategy}>
            保存策略
          </Button>
          <Button icon={<PlayCircleOutlined />} onClick={testStrategy}>
            测试运行
          </Button>
        </Space>
      }
    >
      <div ref={reactFlowWrapper} style={{ height: 600 }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={(_, node) => setSelectedNode(node)}
          nodeTypes={nodeTypes}
          fitView
        >
          <Controls />
          <Background variant="dots" gap={12} size={1} />
        </ReactFlow>
      </div>

      {/* 选中节点的操作栏 */}
      {selectedNode && (
        <div style={{ marginTop: 16, padding: 16, background: '#f5f5f5', borderRadius: 8 }}>
          <Space>
            <span>已选中: {selectedNode.data.label}</span>
            <Button
              size="small"
              icon={<SettingOutlined />}
              onClick={() => {
                form.setFieldsValue(selectedNode.data.params);
                setConfigModalVisible(true);
              }}
            >
              配置
            </Button>
            <Button
              size="small"
              danger
              icon={<DeleteOutlined />}
              onClick={deleteSelectedNode}
            >
              删除
            </Button>
          </Space>
        </div>
      )}

      {/* 添加模块抽屉 */}
      <Drawer
        title="选择模块类型"
        placement="right"
        visible={blockDrawerVisible}
        onClose={() => setBlockDrawerVisible(false)}
        width={300}
      >
        <List
          dataSource={Object.entries(BLOCK_TYPES)}
          renderItem={([type, config]) => (
            <List.Item
              style={{ cursor: 'pointer' }}
              onClick={() => {
                if (type === BlockType.INDICATOR) {
                  // 指标需要选择具体类型
                  Modal.confirm({
                    title: '选择指标类型',
                    content: (
                      <Select
                        style={{ width: '100%', marginTop: 16 }}
                        placeholder="请选择指标"
                        onChange={(value) => {
                          addBlock(type as BlockType, {
                            params: { type: value },
                          });
                          Modal.destroyAll();
                        }}
                      >
                        {INDICATORS.map(ind => (
                          <Option key={ind.value} value={ind.value}>
                            {ind.label}
                          </Option>
                        ))}
                      </Select>
                    ),
                    okButtonProps: { style: { display: 'none' } },
                  });
                } else {
                  addBlock(type as BlockType);
                }
              }}
            >
              <Badge color={config.color} text={
                <span>
                  <span style={{ marginRight: 8 }}>{config.icon}</span>
                  {config.label}
                </span>
              } />
            </List.Item>
          )}
        />
      </Drawer>

      {/* 配置模态框 */}
      <Modal
        title="模块配置"
        visible={configModalVisible}
        onOk={() => form.submit()}
        onCancel={() => setConfigModalVisible(false)}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={configureNode}
        >
          {selectedNode?.data.blockType === BlockType.DATA && (
            <Form.Item name="field" label="数据字段" initialValue="close">
              <Select>
                <Option value="open">开盘价</Option>
                <Option value="high">最高价</Option>
                <Option value="low">最低价</Option>
                <Option value="close">收盘价</Option>
                <Option value="volume">成交量</Option>
              </Select>
            </Form.Item>
          )}
          
          {selectedNode?.data.blockType === BlockType.INDICATOR && (
            <>
              <Form.Item name="period" label="周期" initialValue={20}>
                <InputNumber min={1} max={200} style={{ width: '100%' }} />
              </Form.Item>
              {selectedNode.data.params.type === 'BOLL' && (
                <Form.Item name="std" label="标准差" initialValue={2}>
                  <InputNumber min={0.5} max={3} step={0.5} style={{ width: '100%' }} />
                </Form.Item>
              )}
            </>
          )}
          
          {selectedNode?.data.blockType === BlockType.CONDITION && (
            <Form.Item name="operator" label="比较操作" initialValue=">">
              <Select>
                {OPERATORS.map(op => (
                  <Option key={op.value} value={op.value}>{op.label}</Option>
                ))}
              </Select>
            </Form.Item>
          )}
          
          {selectedNode?.data.blockType === BlockType.ACTION && (
            <>
              <Form.Item name="type" label="动作类型" initialValue="buy">
                <Select>
                  <Option value="buy">买入</Option>
                  <Option value="sell">卖出</Option>
                  <Option value="close">平仓</Option>
                </Select>
              </Form.Item>
              <Form.Item name="position_size" label="仓位大小" initialValue={1.0}>
                <InputNumber
                  min={0.1}
                  max={1.0}
                  step={0.1}
                  style={{ width: '100%' }}
                  formatter={value => `${value * 100}%`}
                  parser={value => Number(value?.replace('%', '')) / 100}
                />
              </Form.Item>
            </>
          )}
        </Form>
      </Modal>
    </Card>
  );
};

export default VisualStrategyBuilder;