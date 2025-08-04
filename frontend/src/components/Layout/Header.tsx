import React, { useState } from 'react';
import { Layout, Button, Space, Dropdown, Avatar, Badge, Switch, Tooltip, Modal } from 'antd';
import type { MenuProps } from 'antd';
import {
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  BellOutlined,
  UserOutlined,
  SettingOutlined,
  LogoutOutlined,
  SunOutlined,
  MoonOutlined,
  QuestionCircleOutlined,
  GlobalOutlined,
} from '@ant-design/icons';
import { useSystemStore } from '../../stores/systemStore';
import { useUserStore } from '../../stores/userStore';
import { SettingsModal } from './SettingsModal';

const { Header: AntHeader } = Layout;

interface HeaderProps {
  collapsed: boolean;
  onCollapse: (collapsed: boolean) => void;
  isMobile: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  collapsed,
  onCollapse,
  isMobile,
}) => {
  const { systemStatus, theme, setTheme } = useSystemStore();
  const { user, logout } = useUserStore();
  const [settingsVisible, setSettingsVisible] = useState(false);

  const handleThemeToggle = (checked: boolean) => {
    setTheme(checked ? 'dark' : 'light');
  };

  const handleSettingsClick = () => {
    setSettingsVisible(true);
  };

  const handleLogout = async () => {
    Modal.confirm({
      title: '确认退出',
      content: '您确定要退出系统吗？',
      okText: '确定',
      cancelText: '取消',
      onOk: async () => {
        await logout();
        // logout会清除认证状态，ProtectedRoute会自动重定向到登录页
      },
    });
  };

  const userMenuItems: MenuProps['items'] = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: '个人资料',
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: '系统设置',
      onClick: handleSettingsClick,
    },
    {
      key: 'help',
      icon: <QuestionCircleOutlined />,
      label: '帮助文档',
    },
    {
      type: 'divider',
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: '退出登录',
      onClick: handleLogout,
    },
  ];

  const notificationMenuItems: MenuProps['items'] = [
    {
      key: 'system',
      label: (
        <div>
          <div style={{ fontWeight: 'bold' }}>系统通知</div>
          <div style={{ fontSize: '12px', color: '#666' }}>
            系统运行正常
          </div>
        </div>
      ),
    },
    {
      key: 'risk',
      label: (
        <div>
          <div style={{ fontWeight: 'bold' }}>风险告警</div>
          <div style={{ fontSize: '12px', color: '#666' }}>
            暂无风险告警
          </div>
        </div>
      ),
    },
    {
      type: 'divider',
    },
    {
      key: 'viewAll',
      label: '查看全部通知',
    },
  ];

  return (
    <AntHeader
      className="header-container"
      style={{
        padding: '0 16px',
        background: '#fff',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: '1px solid #f0f0f0',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center' }}>
        <Button
          type="text"
          icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
          onClick={() => onCollapse(!collapsed)}
          style={{
            fontSize: '16px',
            width: 64,
            height: 64,
          }}
        />
        
        {!isMobile && (
          <div style={{ marginLeft: 16 }}>
            <h1 style={{ margin: 0, fontSize: '18px', fontWeight: 'bold' }}>
              myQuant 量化交易系统
            </h1>
          </div>
        )}
      </div>

      <Space size="middle">
        {/* System Status Indicator */}
        <Tooltip title={`系统${systemStatus?.isRunning ? '运行中' : '已停止'}`}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div
              style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                backgroundColor: systemStatus?.isRunning ? '#52c41a' : '#ff4d4f',
                animation: systemStatus?.isRunning ? 'pulse 2s infinite' : 'none',
              }}
            />
            {!isMobile && (
              <span style={{ fontSize: '12px', color: '#666' }}>
                {systemStatus?.isRunning ? '运行中' : '已停止'}
              </span>
            )}
          </div>
        </Tooltip>

        {/* Language Selector (placeholder for future i18n) */}
        {!isMobile && (
          <Tooltip title="语言设置">
            <Button type="text" icon={<GlobalOutlined />} size="small">
              中文
            </Button>
          </Tooltip>
        )}

        {/* Theme Toggle */}
        <Tooltip title={`切换到${theme === 'light' ? '深色' : '浅色'}主题`}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <SunOutlined style={{ color: theme === 'light' ? '#1890ff' : '#666' }} />
            <Switch
              size="small"
              checked={theme === 'dark'}
              onChange={handleThemeToggle}
            />
            <MoonOutlined style={{ color: theme === 'dark' ? '#1890ff' : '#666' }} />
          </div>
        </Tooltip>

        {/* Notifications */}
        <Dropdown
          menu={{ items: notificationMenuItems }}
          placement="bottomRight"
          trigger={['click']}
        >
          <Tooltip title="通知">
            <Button type="text" style={{ padding: '4px 8px' }}>
              <Badge count={0} size="small">
                <BellOutlined style={{ fontSize: '16px' }} />
              </Badge>
            </Button>
          </Tooltip>
        </Dropdown>

        {/* User Menu */}
        <Dropdown
          menu={{ items: userMenuItems }}
          placement="bottomRight"
          trigger={['click']}
        >
          <div style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 8 }}>
            <Avatar size="small" icon={<UserOutlined />} />
            {!isMobile && (
              <span style={{ fontSize: '14px' }}>
                {user?.username || '用户'}
              </span>
            )}
          </div>
        </Dropdown>
      </Space>

      {/* Settings Modal */}
      <SettingsModal
        visible={settingsVisible}
        onClose={() => setSettingsVisible(false)}
      />
    </AntHeader>
  );
};

export default Header;