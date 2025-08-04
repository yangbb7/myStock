import React from 'react';
import { Modal, Form, Switch, Select, Slider, Divider, Space, Typography, Card } from 'antd';
import { useSystemStore } from '../../stores/systemStore';

const { Title, Text } = Typography;
const { Option } = Select;

interface SettingsModalProps {
  visible: boolean;
  onClose: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  visible,
  onClose,
}) => {
  const { theme, setTheme } = useSystemStore();
  const [form] = Form.useForm();

  const handleSave = () => {
    form.validateFields().then((values) => {
      // Apply settings
      if (values.theme !== theme) {
        setTheme(values.theme);
      }
      
      // Save other settings to localStorage or send to backend
      localStorage.setItem('userSettings', JSON.stringify(values));
      
      onClose();
    });
  };

  const initialValues = {
    theme,
    language: 'zh-CN',
    autoRefresh: true,
    refreshInterval: 5,
    notifications: true,
    soundEnabled: false,
    compactMode: false,
    showAnimations: true,
  };

  return (
    <Modal
      title="系统设置"
      open={visible}
      onOk={handleSave}
      onCancel={onClose}
      width={600}
      okText="保存"
      cancelText="取消"
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={initialValues}
      >
        <Card size="small" title="外观设置" style={{ marginBottom: 16 }}>
          <Form.Item
            name="theme"
            label="主题模式"
            tooltip="选择浅色或深色主题"
          >
            <Select>
              <Option value="light">浅色主题</Option>
              <Option value="dark">深色主题</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="language"
            label="语言设置"
            tooltip="选择界面显示语言"
          >
            <Select>
              <Option value="zh-CN">简体中文</Option>
              <Option value="en-US">English</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="compactMode"
            label="紧凑模式"
            valuePropName="checked"
            tooltip="启用紧凑模式以显示更多内容"
          >
            <Switch />
          </Form.Item>

          <Form.Item
            name="showAnimations"
            label="显示动画"
            valuePropName="checked"
            tooltip="启用界面动画效果"
          >
            <Switch />
          </Form.Item>
        </Card>

        <Card size="small" title="数据刷新设置" style={{ marginBottom: 16 }}>
          <Form.Item
            name="autoRefresh"
            label="自动刷新"
            valuePropName="checked"
            tooltip="启用数据自动刷新"
          >
            <Switch />
          </Form.Item>

          <Form.Item
            name="refreshInterval"
            label="刷新间隔（秒）"
            tooltip="设置数据自动刷新的时间间隔"
          >
            <Slider
              min={1}
              max={60}
              marks={{
                1: '1s',
                5: '5s',
                10: '10s',
                30: '30s',
                60: '1m',
              }}
            />
          </Form.Item>
        </Card>

        <Card size="small" title="通知设置" style={{ marginBottom: 16 }}>
          <Form.Item
            name="notifications"
            label="启用通知"
            valuePropName="checked"
            tooltip="接收系统通知和告警"
          >
            <Switch />
          </Form.Item>

          <Form.Item
            name="soundEnabled"
            label="声音提醒"
            valuePropName="checked"
            tooltip="启用声音提醒"
          >
            <Switch />
          </Form.Item>
        </Card>

        <Divider />
        
        <Space direction="vertical" size="small" style={{ width: '100%' }}>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            * 设置将自动保存到本地存储
          </Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            * 部分设置需要刷新页面后生效
          </Text>
        </Space>
      </Form>
    </Modal>
  );
};

export default SettingsModal;