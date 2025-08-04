import React, { useState, useEffect } from 'react';
import { Layout, Menu } from 'antd';
import type { MenuProps } from 'antd';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  DashboardOutlined,
  BarChartOutlined,
  LineChartOutlined,
  ShoppingCartOutlined,
  PieChartOutlined,
  ExclamationCircleOutlined,
  ExperimentOutlined,
  SettingOutlined,
  ApiOutlined,
} from '@ant-design/icons';

const { Sider } = Layout;

interface SidebarProps {
  collapsed: boolean;
  onCollapse: (collapsed: boolean) => void;
  isMobile: boolean;
}

type MenuItem = Required<MenuProps>['items'][number];

function getItem(
  label: React.ReactNode,
  key: React.Key,
  icon?: React.ReactNode,
  children?: MenuItem[],
): MenuItem {
  return {
    key,
    icon,
    children,
    label,
  } as MenuItem;
}

export const Sidebar: React.FC<SidebarProps> = ({
  collapsed,
  onCollapse,
  isMobile,
}) => {
  const location = useLocation();
  const navigate = useNavigate();
  const [selectedKeys, setSelectedKeys] = useState<string[]>([]);
  const [openKeys, setOpenKeys] = useState<string[]>([]);

  const menuItems: MenuItem[] = [
    getItem('系统仪表板', '/dashboard', <DashboardOutlined />),
    getItem('策略管理', '/strategy', <BarChartOutlined />),
    getItem('实时数据', '/data', <LineChartOutlined />),
    getItem('订单管理', '/orders', <ShoppingCartOutlined />),
    getItem('投资组合', '/portfolio', <PieChartOutlined />),
    getItem('风险监控', '/risk', <ExclamationCircleOutlined />),
    getItem('回测分析', '/backtest', <ExperimentOutlined />),
    getItem('系统管理', '/system', <SettingOutlined />),
    getItem('WebSocket测试', '/websocket-test', <ApiOutlined />),
  ];

  // Update selected keys based on current route
  useEffect(() => {
    const currentPath = location.pathname;
    setSelectedKeys([currentPath]);
    
    // Auto-expand parent menu items
    const pathSegments = currentPath.split('/').filter(Boolean);
    if (pathSegments.length > 1) {
      setOpenKeys([`/${pathSegments[0]}`]);
    }
  }, [location.pathname]);

  const handleMenuClick: MenuProps['onClick'] = ({ key }) => {
    navigate(key);
    
    // Close sidebar on mobile after navigation
    if (isMobile && !collapsed) {
      onCollapse(true);
    }
  };

  const handleOpenChange = (keys: string[]) => {
    setOpenKeys(keys);
  };

  return (
    <>
      {/* Mobile overlay */}
      {isMobile && !collapsed && (
        <div
          className="mobile-overlay"
          onClick={() => onCollapse(true)}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              onCollapse(true);
            }
          }}
          aria-label="关闭侧边栏"
        />
      )}
      
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        width={256}
        collapsedWidth={isMobile ? 0 : 80}
        className="sidebar-container"
        style={{
          position: isMobile ? 'fixed' : 'relative',
          height: isMobile ? '100vh' : 'auto',
          left: isMobile && collapsed ? -256 : 0,
          top: isMobile ? 64 : 0, // Account for header height
          zIndex: isMobile ? 1000 : 'auto',
        }}
        theme="light"
      >
        <div
          style={{
            height: 32,
            margin: 16,
            background: 'rgba(255, 255, 255, 0.3)',
            borderRadius: 6,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: collapsed ? '12px' : '16px',
            fontWeight: 'bold',
            color: '#1890ff',
          }}
        >
          {collapsed ? 'mQ' : 'myQuant'}
        </div>
        
        <Menu
          mode="inline"
          selectedKeys={selectedKeys}
          openKeys={collapsed ? [] : openKeys}
          onOpenChange={handleOpenChange}
          onClick={handleMenuClick}
          items={menuItems}
          style={{
            borderRight: 0,
            height: 'calc(100vh - 112px)', // Account for header and logo
            overflowY: 'auto',
          }}
        />
      </Sider>
    </>
  );
};

export default Sidebar;