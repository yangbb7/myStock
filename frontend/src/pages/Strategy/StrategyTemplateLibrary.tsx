import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Badge, Button, Tag, Modal, Form, Input, Select, message, Tooltip, Space, Statistic } from 'antd';
import {
  TrophyOutlined,
  RiseOutlined,
  FallOutlined,
  LineChartOutlined,
  PlayCircleOutlined,
  CopyOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons';
import { strategyApi, StrategyTemplate } from '@/api/strategy';

const { Option } = Select;

// 策略难度配置
const DIFFICULTY_CONFIG = {
  beginner: { color: 'green', label: '初级', icon: '🌱' },
  intermediate: { color: 'orange', label: '中级', icon: '🌟' },
  advanced: { color: 'red', label: '高级', icon: '🔥' },
};

// 策略分类配置
const CATEGORY_CONFIG = {
  '趋势跟踪': { color: 'blue', icon: '📈' },
  '均值回归': { color: 'purple', icon: '🔄' },
  '突破策略': { color: 'cyan', icon: '🚀' },
  '套利策略': { color: 'gold', icon: '💰' },
  '高频交易': { color: 'magenta', icon: '⚡' },
  '基本面策略': { color: 'green', icon: '📊' },
};

interface StrategyTemplateLibraryProps {
  onSelectTemplate?: (template: StrategyTemplate) => void;
  onCreateFromTemplate?: (templateId: string, name: string, symbols: string[]) => void;
}

const StrategyTemplateLibrary: React.FC<StrategyTemplateLibraryProps> = ({
  onSelectTemplate,
  onCreateFromTemplate,
}) => {
  const [templates, setTemplates] = useState<StrategyTemplate[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<StrategyTemplate | null>(null);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [filterCategory, setFilterCategory] = useState<string>('all');
  const [filterDifficulty, setFilterDifficulty] = useState<string>('all');
  const [form] = Form.useForm();

  // 加载策略模板
  useEffect(() => {
    loadTemplates();
  }, []);

  const loadTemplates = async () => {
    setLoading(true);
    try {
      const response = await strategyApi.getStrategyTemplates();
      setTemplates(response.data.templates);
    } catch (error) {
      message.error('加载策略模板失败');
    } finally {
      setLoading(false);
    }
  };

  // 筛选模板
  const filteredTemplates = templates.filter(template => {
    if (filterCategory !== 'all' && template.category !== filterCategory) {
      return false;
    }
    if (filterDifficulty !== 'all' && template.difficulty !== filterDifficulty) {
      return false;
    }
    return true;
  });

  // 创建策略
  const handleCreateStrategy = async (values: any) => {
    if (!selectedTemplate) return;

    try {
      await strategyApi.createFromTemplate(
        selectedTemplate.id,
        values.name,
        values.symbols.split(',').map((s: string) => s.trim())
      );
      message.success('策略创建成功');
      setCreateModalVisible(false);
      form.resetFields();
      onCreateFromTemplate?.(selectedTemplate.id, values.name, values.symbols);
    } catch (error) {
      message.error('策略创建失败');
    }
  };

  // 渲染性能指标
  const renderPerformanceMetrics = (metrics?: any) => {
    if (!metrics) return null;

    return (
      <Row gutter={[8, 8]}>
        <Col span={12}>
          <Statistic
            title="年化收益"
            value={metrics.annual_return * 100}
            precision={2}
            suffix="%"
            valueStyle={{ color: metrics.annual_return > 0 ? '#3f8600' : '#cf1322', fontSize: 14 }}
          />
        </Col>
        <Col span={12}>
          <Statistic
            title="夏普比率"
            value={metrics.sharpe_ratio}
            precision={2}
            valueStyle={{ fontSize: 14 }}
          />
        </Col>
        <Col span={12}>
          <Statistic
            title="最大回撤"
            value={metrics.max_drawdown * 100}
            precision={2}
            suffix="%"
            valueStyle={{ color: '#cf1322', fontSize: 14 }}
          />
        </Col>
        <Col span={12}>
          <Statistic
            title="胜率"
            value={metrics.win_rate * 100}
            precision={1}
            suffix="%"
            valueStyle={{ fontSize: 14 }}
          />
        </Col>
      </Row>
    );
  };

  return (
    <>
      <Card
        title="策略模板库"
        extra={
          <Space>
            <Select
              value={filterCategory}
              onChange={setFilterCategory}
              style={{ width: 120 }}
              placeholder="选择分类"
            >
              <Option value="all">全部分类</Option>
              {Object.keys(CATEGORY_CONFIG).map(category => (
                <Option key={category} value={category}>
                  {CATEGORY_CONFIG[category as keyof typeof CATEGORY_CONFIG].icon} {category}
                </Option>
              ))}
            </Select>
            <Select
              value={filterDifficulty}
              onChange={setFilterDifficulty}
              style={{ width: 100 }}
              placeholder="选择难度"
            >
              <Option value="all">全部难度</Option>
              {Object.entries(DIFFICULTY_CONFIG).map(([key, config]) => (
                <Option key={key} value={key}>
                  {config.icon} {config.label}
                </Option>
              ))}
            </Select>
          </Space>
        }
      >
        <Row gutter={[16, 16]}>
          {filteredTemplates.map(template => {
            const difficultyConfig = DIFFICULTY_CONFIG[template.difficulty as keyof typeof DIFFICULTY_CONFIG];
            const categoryConfig = CATEGORY_CONFIG[template.category as keyof typeof CATEGORY_CONFIG];

            return (
              <Col key={template.id} xs={24} sm={12} md={8} lg={6}>
                <Card
                  hoverable
                  className="template-card"
                  actions={[
                    <Tooltip title="查看详情">
                      <InfoCircleOutlined
                        onClick={() => {
                          setSelectedTemplate(template);
                          onSelectTemplate?.(template);
                        }}
                      />
                    </Tooltip>,
                    <Tooltip title="使用模板">
                      <PlayCircleOutlined
                        onClick={() => {
                          setSelectedTemplate(template);
                          setCreateModalVisible(true);
                        }}
                      />
                    </Tooltip>,
                    <Tooltip title="复制策略">
                      <CopyOutlined
                        onClick={() => {
                          message.info('复制功能开发中');
                        }}
                      />
                    </Tooltip>,
                  ]}
                >
                  <div className="template-header">
                    <h3 style={{ marginBottom: 8 }}>
                      {categoryConfig?.icon} {template.name}
                    </h3>
                    <div style={{ marginBottom: 12 }}>
                      <Tag color={categoryConfig?.color}>{template.category}</Tag>
                      <Tag color={difficultyConfig?.color}>
                        {difficultyConfig?.icon} {difficultyConfig?.label}
                      </Tag>
                    </div>
                  </div>
                  
                  <p style={{ marginBottom: 16, minHeight: 48 }}>{template.description}</p>
                  
                  {template.performance_metrics && (
                    <div style={{ borderTop: '1px solid #f0f0f0', paddingTop: 12 }}>
                      {renderPerformanceMetrics(template.performance_metrics)}
                    </div>
                  )}
                </Card>
              </Col>
            );
          })}
        </Row>

        {filteredTemplates.length === 0 && (
          <div style={{ textAlign: 'center', padding: 60 }}>
            <LineChartOutlined style={{ fontSize: 48, color: '#ccc' }} />
            <p style={{ marginTop: 16, color: '#999' }}>暂无符合条件的策略模板</p>
          </div>
        )}
      </Card>

      {/* 创建策略模态框 */}
      <Modal
        title={`使用模板创建策略: ${selectedTemplate?.name}`}
        visible={createModalVisible}
        onOk={() => form.submit()}
        onCancel={() => {
          setCreateModalVisible(false);
          form.resetFields();
        }}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateStrategy}
        >
          <Form.Item
            name="name"
            label="策略名称"
            rules={[{ required: true, message: '请输入策略名称' }]}
          >
            <Input placeholder="请输入策略名称" />
          </Form.Item>
          
          <Form.Item
            name="symbols"
            label="交易标的"
            rules={[{ required: true, message: '请输入交易标的' }]}
            extra="多个标的用逗号分隔，如: 000001.SZ,000002.SZ"
          >
            <Input.TextArea
              rows={2}
              placeholder="请输入交易标的代码"
              defaultValue="000001.SZ"
            />
          </Form.Item>
          
          {selectedTemplate && (
            <Card title="模板信息" size="small">
              <p><strong>策略类型：</strong>{selectedTemplate.category}</p>
              <p><strong>难度等级：</strong>{DIFFICULTY_CONFIG[selectedTemplate.difficulty as keyof typeof DIFFICULTY_CONFIG]?.label}</p>
              <p><strong>策略描述：</strong>{selectedTemplate.description}</p>
              {selectedTemplate.performance_metrics && (
                <div style={{ marginTop: 16 }}>
                  <strong>历史表现：</strong>
                  {renderPerformanceMetrics(selectedTemplate.performance_metrics)}
                </div>
              )}
            </Card>
          )}
        </Form>
      </Modal>

      <style jsx>{`
        .template-card {
          height: 100%;
          transition: all 0.3s;
        }
        .template-card:hover {
          transform: translateY(-4px);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        .template-header h3 {
          margin: 0;
          font-size: 16px;
        }
      `}</style>
    </>
  );
};

export default StrategyTemplateLibrary;