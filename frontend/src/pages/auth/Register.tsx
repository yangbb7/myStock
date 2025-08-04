import React from 'react';
import { Form, Input, Button, Card, Typography, Space, Divider } from 'antd';
import { UserOutlined, LockOutlined, MailOutlined } from '@ant-design/icons';
import { useNavigate, Link } from 'react-router-dom';
import { useUserStore } from '@/stores/userStore';
import type { RegisterData } from '@/types/auth';

const { Title, Text } = Typography;

const Register: React.FC = () => {
  const navigate = useNavigate();
  const { register, loading } = useUserStore();
  const [form] = Form.useForm();

  const handleRegister = async (values: RegisterData) => {
    try {
      await register(values);
      navigate('/');
    } catch (error) {
      // 错误已在store中处理
    }
  };

  const validatePassword = (_: any, value: string) => {
    if (!value) {
      return Promise.reject(new Error('请输入密码'));
    }
    if (value.length < 6) {
      return Promise.reject(new Error('密码长度至少6位'));
    }
    if (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/.test(value)) {
      return Promise.reject(new Error('密码必须包含大小写字母和数字'));
    }
    return Promise.resolve();
  };

  return (
    <div style={{ 
      minHeight: '100vh', 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center',
      background: '#f0f2f5'
    }}>
      <Card 
        style={{ 
          width: 400, 
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)' 
        }}
      >
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <div style={{ textAlign: 'center' }}>
            <Title level={2} style={{ marginBottom: 8 }}>
              创建账号
            </Title>
            <Text type="secondary">
              注册 MyQuant 量化交易系统账号
            </Text>
          </div>

          <Form
            form={form}
            name="register"
            onFinish={handleRegister}
            autoComplete="off"
            layout="vertical"
            size="large"
          >
            <Form.Item
              name="username"
              rules={[
                { required: true, message: '请输入用户名' },
                { min: 3, message: '用户名至少3个字符' },
                { max: 20, message: '用户名最多20个字符' },
                { pattern: /^[a-zA-Z0-9_]+$/, message: '用户名只能包含字母、数字和下划线' }
              ]}
            >
              <Input 
                prefix={<UserOutlined />} 
                placeholder="用户名" 
              />
            </Form.Item>

            <Form.Item
              name="email"
              rules={[
                { required: true, message: '请输入邮箱' },
                { type: 'email', message: '请输入有效的邮箱地址' }
              ]}
            >
              <Input 
                prefix={<MailOutlined />} 
                placeholder="邮箱地址" 
              />
            </Form.Item>

            <Form.Item
              name="password"
              rules={[
                { validator: validatePassword }
              ]}
            >
              <Input.Password
                prefix={<LockOutlined />}
                placeholder="密码"
              />
            </Form.Item>

            <Form.Item
              name="confirmPassword"
              dependencies={['password']}
              rules={[
                { required: true, message: '请确认密码' },
                ({ getFieldValue }) => ({
                  validator(_, value) {
                    if (!value || getFieldValue('password') === value) {
                      return Promise.resolve();
                    }
                    return Promise.reject(new Error('两次输入的密码不一致'));
                  },
                }),
              ]}
            >
              <Input.Password
                prefix={<LockOutlined />}
                placeholder="确认密码"
              />
            </Form.Item>

            <Form.Item>
              <Button 
                type="primary" 
                htmlType="submit" 
                loading={loading}
                block
              >
                注册
              </Button>
            </Form.Item>
          </Form>

          <Divider>或</Divider>

          <div style={{ textAlign: 'center' }}>
            <Text>
              已有账号？{' '}
              <Link to="/login">立即登录</Link>
            </Text>
          </div>
        </Space>
      </Card>
    </div>
  );
};

export default Register;