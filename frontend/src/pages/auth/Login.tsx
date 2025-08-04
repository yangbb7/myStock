import React from 'react';
import { Form, Input, Button, Card, Typography, Space, Divider } from 'antd';
import { UserOutlined, LockOutlined } from '@ant-design/icons';
import { useNavigate, Link } from 'react-router-dom';
import { useUserStore } from '@/stores/userStore';
import type { LoginCredentials } from '@/types/auth';

const { Title, Text } = Typography;

const Login: React.FC = () => {
  const navigate = useNavigate();
  const { login, loading } = useUserStore();
  const [form] = Form.useForm();

  const handleLogin = async (values: LoginCredentials) => {
    try {
      await login(values);
      navigate('/');
    } catch (error) {
      // 错误已在store中处理
    }
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
              欢迎回来
            </Title>
            <Text type="secondary">
              登录到 MyQuant 量化交易系统
            </Text>
          </div>

          <Form
            form={form}
            name="login"
            onFinish={handleLogin}
            autoComplete="off"
            layout="vertical"
            size="large"
          >
            <Form.Item
              name="username"
              rules={[
                { required: true, message: '请输入用户名' }
              ]}
            >
              <Input 
                prefix={<UserOutlined />} 
                placeholder="用户名" 
              />
            </Form.Item>

            <Form.Item
              name="password"
              rules={[
                { required: true, message: '请输入密码' }
              ]}
            >
              <Input.Password
                prefix={<LockOutlined />}
                placeholder="密码"
              />
            </Form.Item>

            <Form.Item>
              <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                <Link to="/forgot-password">
                  忘记密码？
                </Link>
              </Space>
            </Form.Item>

            <Form.Item>
              <Button 
                type="primary" 
                htmlType="submit" 
                loading={loading}
                block
              >
                登录
              </Button>
            </Form.Item>
          </Form>

          <Divider>或</Divider>

          <div style={{ textAlign: 'center' }}>
            <Text>
              还没有账号？{' '}
              <Link to="/register">立即注册</Link>
            </Text>
          </div>
        </Space>
      </Card>
    </div>
  );
};

export default Login;