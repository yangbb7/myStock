import React, { useState } from 'react';
import {
  Card,
  Row,
  Col,
  Form,
  Select,
  DatePicker,
  Button,
  Space,
  Checkbox,
  Input,
  Divider,
  List,
  Tag,
  Modal,
  message,
  Upload,
  Typography,
  Alert,
  Progress,
} from 'antd';
import {
  DownloadOutlined,
  FileExcelOutlined,
  FilePdfOutlined,
  ScheduleOutlined,
  SettingOutlined,
  UploadOutlined,
  DeleteOutlined,
  EyeOutlined,
} from '@ant-design/icons';
import { useMutation, useQuery } from '@tanstack/react-query';
import dayjs from 'dayjs';
import { api } from '../../../services/api';
import { LoadingState } from '../../../components/Common/LoadingState';

const { RangePicker } = DatePicker;
const { TextArea } = Input;
const { Title, Text } = Typography;

interface ReportTemplate {
  id: string;
  name: string;
  description: string;
  format: 'pdf' | 'excel';
  sections: string[];
  isDefault: boolean;
  createdAt: string;
}

interface ReportHistory {
  id: string;
  name: string;
  format: 'pdf' | 'excel';
  status: 'generating' | 'completed' | 'failed';
  progress: number;
  createdAt: string;
  downloadUrl?: string;
  size?: string;
}

interface ScheduledReport {
  id: string;
  name: string;
  template: string;
  frequency: 'daily' | 'weekly' | 'monthly';
  nextRun: string;
  isActive: boolean;
  recipients: string[];
}

export const ReportExport: React.FC = () => {
  const [form] = Form.useForm();
  const [templateModalVisible, setTemplateModalVisible] = useState(false);
  const [scheduleModalVisible, setScheduleModalVisible] = useState(false);
  const [selectedSections, setSelectedSections] = useState<string[]>([
    'portfolio_summary',
    'performance_analysis',
    'risk_analysis',
    'position_details',
  ]);

  // Mock data for templates
  const reportTemplates: ReportTemplate[] = [
    {
      id: '1',
      name: '标准投资组合报告',
      description: '包含投资组合概览、收益分析、风险指标等标准内容',
      format: 'pdf',
      sections: ['portfolio_summary', 'performance_analysis', 'risk_analysis', 'position_details'],
      isDefault: true,
      createdAt: '2024-01-15',
    },
    {
      id: '2',
      name: '详细风险报告',
      description: '专注于风险分析的详细报告',
      format: 'pdf',
      sections: ['risk_analysis', 'var_analysis', 'correlation_matrix', 'stress_testing'],
      isDefault: false,
      createdAt: '2024-01-10',
    },
    {
      id: '3',
      name: 'Excel数据导出',
      description: '导出所有数据到Excel格式，便于进一步分析',
      format: 'excel',
      sections: ['raw_data', 'position_details', 'transaction_history', 'performance_metrics'],
      isDefault: false,
      createdAt: '2024-01-05',
    },
  ];

  // Mock data for report history
  const [reportHistory, setReportHistory] = useState<ReportHistory[]>([
    {
      id: '1',
      name: '投资组合月度报告_2024-01',
      format: 'pdf',
      status: 'completed',
      progress: 100,
      createdAt: '2024-01-31 15:30:00',
      downloadUrl: '#',
      size: '2.5MB',
    },
    {
      id: '2',
      name: '风险分析报告_2024-01-30',
      format: 'excel',
      status: 'completed',
      progress: 100,
      createdAt: '2024-01-30 10:15:00',
      downloadUrl: '#',
      size: '1.8MB',
    },
    {
      id: '3',
      name: '投资组合周报_2024-W04',
      format: 'pdf',
      status: 'generating',
      progress: 65,
      createdAt: '2024-01-29 14:20:00',
    },
  ]);

  // Mock data for scheduled reports
  const scheduledReports: ScheduledReport[] = [
    {
      id: '1',
      name: '每日风险监控报告',
      template: '详细风险报告',
      frequency: 'daily',
      nextRun: '2024-02-01 09:00:00',
      isActive: true,
      recipients: ['risk@company.com', 'manager@company.com'],
    },
    {
      id: '2',
      name: '周度投资组合报告',
      template: '标准投资组合报告',
      frequency: 'weekly',
      nextRun: '2024-02-05 08:00:00',
      isActive: true,
      recipients: ['investors@company.com'],
    },
  ];

  // Available report sections
  const reportSections = [
    { value: 'portfolio_summary', label: '投资组合概览' },
    { value: 'performance_analysis', label: '收益分析' },
    { value: 'risk_analysis', label: '风险分析' },
    { value: 'position_details', label: '持仓明细' },
    { value: 'transaction_history', label: '交易记录' },
    { value: 'var_analysis', label: 'VaR分析' },
    { value: 'correlation_matrix', label: '相关性矩阵' },
    { value: 'stress_testing', label: '压力测试' },
    { value: 'benchmark_comparison', label: '基准对比' },
    { value: 'attribution_analysis', label: '归因分析' },
    { value: 'raw_data', label: '原始数据' },
    { value: 'performance_metrics', label: '绩效指标' },
  ];

  // Generate report mutation
  const generateReportMutation = useMutation({
    mutationFn: async (params: any) => {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      return { success: true, reportId: Date.now().toString() };
    },
    onSuccess: (data) => {
      message.success('报告生成任务已提交，请在报告历史中查看进度');
      // Add to report history
      const newReport: ReportHistory = {
        id: data.reportId,
        name: `自定义报告_${dayjs().format('YYYY-MM-DD_HH-mm')}`,
        format: form.getFieldValue('format'),
        status: 'generating',
        progress: 0,
        createdAt: dayjs().format('YYYY-MM-DD HH:mm:ss'),
      };
      setReportHistory(prev => [newReport, ...prev]);
    },
    onError: () => {
      message.error('报告生成失败，请重试');
    },
  });

  // Handle form submission
  const handleGenerateReport = (values: any) => {
    generateReportMutation.mutate({
      ...values,
      sections: selectedSections,
    });
  };

  // Handle download
  const handleDownload = (report: ReportHistory) => {
    if (report.status === 'completed' && report.downloadUrl) {
      // In real implementation, this would trigger actual download
      message.success(`开始下载 ${report.name}`);
    }
  };

  // Handle delete report
  const handleDeleteReport = (reportId: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这个报告吗？',
      onOk: () => {
        setReportHistory(prev => prev.filter(r => r.id !== reportId));
        message.success('报告已删除');
      },
    });
  };

  return (
    <div style={{ padding: '0 0 24px 0' }}>
      <Row gutter={[16, 16]}>
        {/* Report Generation Form */}
        <Col xs={24} lg={12}>
          <Card title="生成报告" size="small">
            <Form
              form={form}
              layout="vertical"
              onFinish={handleGenerateReport}
              initialValues={{
                format: 'pdf',
                dateRange: [dayjs().subtract(1, 'month'), dayjs()],
                template: '1',
              }}
            >
              <Form.Item
                name="reportName"
                label="报告名称"
                rules={[{ required: true, message: '请输入报告名称' }]}
              >
                <Input placeholder="请输入报告名称" />
              </Form.Item>

              <Form.Item
                name="template"
                label="报告模板"
                rules={[{ required: true, message: '请选择报告模板' }]}
              >
                <Select placeholder="选择报告模板">
                  {reportTemplates.map(template => (
                    <Select.Option key={template.id} value={template.id}>
                      <Space>
                        {template.format === 'pdf' ? <FilePdfOutlined /> : <FileExcelOutlined />}
                        {template.name}
                        {template.isDefault && <Tag color="blue">默认</Tag>}
                      </Space>
                    </Select.Option>
                  ))}
                </Select>
              </Form.Item>

              <Form.Item
                name="format"
                label="导出格式"
                rules={[{ required: true, message: '请选择导出格式' }]}
              >
                <Select>
                  <Select.Option value="pdf">
                    <Space>
                      <FilePdfOutlined />
                      PDF格式
                    </Space>
                  </Select.Option>
                  <Select.Option value="excel">
                    <Space>
                      <FileExcelOutlined />
                      Excel格式
                    </Space>
                  </Select.Option>
                </Select>
              </Form.Item>

              <Form.Item
                name="dateRange"
                label="报告时间范围"
                rules={[{ required: true, message: '请选择时间范围' }]}
              >
                <RangePicker style={{ width: '100%' }} />
              </Form.Item>

              <Form.Item label="报告内容">
                <Checkbox.Group
                  value={selectedSections}
                  onChange={setSelectedSections}
                  style={{ width: '100%' }}
                >
                  <Row gutter={[8, 8]}>
                    {reportSections.map(section => (
                      <Col span={12} key={section.value}>
                        <Checkbox value={section.value}>
                          {section.label}
                        </Checkbox>
                      </Col>
                    ))}
                  </Row>
                </Checkbox.Group>
              </Form.Item>

              <Form.Item name="description" label="报告描述">
                <TextArea
                  rows={3}
                  placeholder="可选：添加报告描述或备注"
                />
              </Form.Item>

              <Form.Item>
                <Space>
                  <Button
                    type="primary"
                    htmlType="submit"
                    icon={<DownloadOutlined />}
                    loading={generateReportMutation.isPending}
                  >
                    生成报告
                  </Button>
                  <Button
                    icon={<SettingOutlined />}
                    onClick={() => setTemplateModalVisible(true)}
                  >
                    管理模板
                  </Button>
                  <Button
                    icon={<ScheduleOutlined />}
                    onClick={() => setScheduleModalVisible(true)}
                  >
                    定时报告
                  </Button>
                </Space>
              </Form.Item>
            </Form>
          </Card>
        </Col>

        {/* Report Templates */}
        <Col xs={24} lg={12}>
          <Card title="报告模板" size="small">
            <List
              dataSource={reportTemplates}
              renderItem={(template) => (
                <List.Item
                  actions={[
                    <Button type="text" size="small" icon={<EyeOutlined />}>
                      预览
                    </Button>,
                    <Button type="text" size="small" icon={<SettingOutlined />}>
                      编辑
                    </Button>,
                  ]}
                >
                  <List.Item.Meta
                    avatar={template.format === 'pdf' ? <FilePdfOutlined /> : <FileExcelOutlined />}
                    title={
                      <Space>
                        {template.name}
                        {template.isDefault && <Tag color="blue">默认</Tag>}
                      </Space>
                    }
                    description={template.description}
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>

      {/* Report History */}
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col span={24}>
          <Card title="报告历史" size="small">
            <List
              dataSource={reportHistory}
              renderItem={(report) => (
                <List.Item
                  actions={[
                    report.status === 'completed' && (
                      <Button
                        type="text"
                        size="small"
                        icon={<DownloadOutlined />}
                        onClick={() => handleDownload(report)}
                      >
                        下载
                      </Button>
                    ),
                    <Button
                      type="text"
                      size="small"
                      icon={<DeleteOutlined />}
                      danger
                      onClick={() => handleDeleteReport(report.id)}
                    >
                      删除
                    </Button>,
                  ].filter(Boolean)}
                >
                  <List.Item.Meta
                    avatar={report.format === 'pdf' ? <FilePdfOutlined /> : <FileExcelOutlined />}
                    title={
                      <Space>
                        {report.name}
                        <Tag color={
                          report.status === 'completed' ? 'green' :
                          report.status === 'generating' ? 'blue' : 'red'
                        }>
                          {report.status === 'completed' ? '已完成' :
                           report.status === 'generating' ? '生成中' : '失败'}
                        </Tag>
                        {report.size && <Text type="secondary">({report.size})</Text>}
                      </Space>
                    }
                    description={
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <Text type="secondary">创建时间: {report.createdAt}</Text>
                        {report.status === 'generating' && (
                          <Progress percent={report.progress} size="small" />
                        )}
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>

      {/* Scheduled Reports */}
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col span={24}>
          <Card
            title="定时报告"
            size="small"
            extra={
              <Button
                type="primary"
                size="small"
                onClick={() => setScheduleModalVisible(true)}
              >
                新建定时任务
              </Button>
            }
          >
            <List
              dataSource={scheduledReports}
              renderItem={(schedule) => (
                <List.Item
                  actions={[
                    <Button type="text" size="small">
                      编辑
                    </Button>,
                    <Button type="text" size="small" danger>
                      删除
                    </Button>,
                  ]}
                >
                  <List.Item.Meta
                    avatar={<ScheduleOutlined />}
                    title={
                      <Space>
                        {schedule.name}
                        <Tag color={schedule.isActive ? 'green' : 'default'}>
                          {schedule.isActive ? '启用' : '禁用'}
                        </Tag>
                        <Tag>{
                          schedule.frequency === 'daily' ? '每日' :
                          schedule.frequency === 'weekly' ? '每周' : '每月'
                        }</Tag>
                      </Space>
                    }
                    description={
                      <Space direction="vertical">
                        <Text type="secondary">模板: {schedule.template}</Text>
                        <Text type="secondary">下次运行: {schedule.nextRun}</Text>
                        <Text type="secondary">
                          收件人: {schedule.recipients.join(', ')}
                        </Text>
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>

      {/* Template Management Modal */}
      <Modal
        title="管理报告模板"
        open={templateModalVisible}
        onCancel={() => setTemplateModalVisible(false)}
        footer={null}
        width={800}
      >
        <Alert
          message="模板管理功能"
          description="在这里可以创建、编辑和删除报告模板。自定义模板可以包含不同的报告内容组合。"
          type="info"
          style={{ marginBottom: 16 }}
        />
        <Space direction="vertical" style={{ width: '100%' }}>
          <Button type="primary" icon={<UploadOutlined />}>
            创建新模板
          </Button>
          {/* Template management content would go here */}
        </Space>
      </Modal>

      {/* Schedule Management Modal */}
      <Modal
        title="定时报告设置"
        open={scheduleModalVisible}
        onCancel={() => setScheduleModalVisible(false)}
        footer={null}
        width={600}
      >
        <Alert
          message="定时报告功能"
          description="设置定时生成和发送报告。可以选择每日、每周或每月的频率，并指定收件人邮箱。"
          type="info"
          style={{ marginBottom: 16 }}
        />
        <Form layout="vertical">
          <Form.Item label="任务名称" required>
            <Input placeholder="请输入定时任务名称" />
          </Form.Item>
          <Form.Item label="报告模板" required>
            <Select placeholder="选择报告模板">
              {reportTemplates.map(template => (
                <Select.Option key={template.id} value={template.id}>
                  {template.name}
                </Select.Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item label="执行频率" required>
            <Select placeholder="选择执行频率">
              <Select.Option value="daily">每日</Select.Option>
              <Select.Option value="weekly">每周</Select.Option>
              <Select.Option value="monthly">每月</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item label="收件人邮箱">
            <TextArea
              rows={3}
              placeholder="请输入收件人邮箱，多个邮箱用逗号分隔"
            />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary">保存设置</Button>
              <Button onClick={() => setScheduleModalVisible(false)}>取消</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default ReportExport;