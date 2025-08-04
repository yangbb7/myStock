import React, { useMemo } from 'react';
import { Breadcrumb as AntBreadcrumb } from 'antd';
import type { BreadcrumbProps } from 'antd';
import { useLocation, Link } from 'react-router-dom';
import { HomeOutlined } from '@ant-design/icons';

interface BreadcrumbItem {
  path: string;
  title: string;
  icon?: React.ReactNode;
}

interface CustomBreadcrumbProps extends BreadcrumbProps {
  customItems?: BreadcrumbItem[];
  showHome?: boolean;
  separator?: string;
}

// Route configuration for breadcrumb generation
const routeConfig: Record<string, { title: string; icon?: React.ReactNode }> = {
  '/dashboard': { title: '系统仪表板', icon: <HomeOutlined /> },
  '/strategy': { title: '策略管理' },
  '/strategy/add': { title: '添加策略' },
  '/strategy/edit': { title: '编辑策略' },
  '/data': { title: '实时数据' },
  '/data/market': { title: '市场数据' },
  '/data/tick': { title: 'Tick数据' },
  '/orders': { title: '订单管理' },
  '/orders/create': { title: '创建订单' },
  '/orders/history': { title: '历史订单' },
  '/portfolio': { title: '投资组合' },
  '/portfolio/analysis': { title: '组合分析' },
  '/portfolio/positions': { title: '持仓详情' },
  '/risk': { title: '风险监控' },
  '/risk/alerts': { title: '风险告警' },
  '/risk/limits': { title: '风险限制' },
  '/backtest': { title: '回测分析' },
  '/backtest/create': { title: '创建回测' },
  '/backtest/results': { title: '回测结果' },
  '/system': { title: '系统管理' },
  '/system/config': { title: '系统配置' },
  '/system/logs': { title: '系统日志' },
};

export const Breadcrumb: React.FC<CustomBreadcrumbProps> = ({
  customItems,
  showHome = true,
  separator = '/',
  ...props
}) => {
  const location = useLocation();

  const breadcrumbItems = useMemo(() => {
    if (customItems) {
      return customItems.map((item, index) => ({
        key: item.path,
        title: index === customItems.length - 1 ? (
          <span>
            {item.icon} {item.title}
          </span>
        ) : (
          <Link to={item.path}>
            {item.icon} {item.title}
          </Link>
        ),
      }));
    }

    const pathSegments = location.pathname.split('/').filter(Boolean);
    const items: any[] = [];

    // Add home item if enabled
    if (showHome && location.pathname !== '/dashboard') {
      items.push({
        key: '/dashboard',
        title: (
          <Link to="/dashboard">
            <HomeOutlined /> 首页
          </Link>
        ),
      });
    }

    // Build breadcrumb items from path segments
    let currentPath = '';
    pathSegments.forEach((segment, index) => {
      currentPath += `/${segment}`;
      const config = routeConfig[currentPath];
      
      if (config) {
        const isLast = index === pathSegments.length - 1;
        items.push({
          key: currentPath,
          title: isLast ? (
            <span>
              {config.icon} {config.title}
            </span>
          ) : (
            <Link to={currentPath}>
              {config.icon} {config.title}
            </Link>
          ),
        });
      }
    });

    return items;
  }, [location.pathname, customItems, showHome]);

  // Don't render breadcrumb if there's only one item or we're on the dashboard
  if (breadcrumbItems.length <= 1 && location.pathname === '/dashboard') {
    return null;
  }

  return (
    <AntBreadcrumb
      separator={separator}
      items={breadcrumbItems}
      {...props}
    />
  );
};

// Hook for programmatically setting breadcrumb items
export const useBreadcrumb = () => {
  const setBreadcrumb = (items: BreadcrumbItem[]) => {
    // This could be implemented with a context or store
    // For now, it's a placeholder for future implementation
    console.log('Setting breadcrumb items:', items);
  };

  return { setBreadcrumb };
};

export default Breadcrumb;