import React, { useState, useEffect, useCallback } from 'react';
import { Layout, theme } from 'antd';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { Footer } from './Footer';
import { Breadcrumb } from './Breadcrumb';
import { ErrorBoundary } from '../Common/ErrorBoundary';
import './Layout.css';

const { Content } = Layout;

interface MainLayoutProps {
  children: React.ReactNode;
  showBreadcrumb?: boolean;
  showFooter?: boolean;
  sidebarCollapsed?: boolean;
  onSidebarCollapse?: (collapsed: boolean) => void;
}

export const MainLayout: React.FC<MainLayoutProps> = ({
  children,
  showBreadcrumb = true,
  showFooter = true,
  sidebarCollapsed,
  onSidebarCollapse,
}) => {
  const [collapsed, setCollapsed] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  
  const {
    token: { colorBgContainer, borderRadiusLG },
  } = theme.useToken();

  // Keyboard navigation support
  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    // Toggle sidebar with Ctrl/Cmd + B
    if ((event.ctrlKey || event.metaKey) && event.key === 'b') {
      event.preventDefault();
      setCollapsed(prev => !prev);
    }
  }, []);

  // Handle responsive design
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (mobile && !collapsed) {
        setCollapsed(true);
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [collapsed]);

  // Add keyboard navigation
  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  const handleCollapse = (value: boolean) => {
    setCollapsed(value);
    if (onSidebarCollapse) {
      onSidebarCollapse(value);
    }
  };

  const effectiveCollapsed = sidebarCollapsed !== undefined ? sidebarCollapsed : collapsed;

  return (
    <Layout style={{ minHeight: '100vh' }} className="layout-transition">
      <Header 
        collapsed={effectiveCollapsed}
        onCollapse={handleCollapse}
        isMobile={isMobile}
      />
      
      <Layout>
        <Sidebar 
          collapsed={effectiveCollapsed}
          onCollapse={handleCollapse}
          isMobile={isMobile}
        />
        
        <Layout style={{ padding: isMobile ? '0 8px 8px' : '0 24px 24px' }}>
          {showBreadcrumb && (
            <div className="breadcrumb-container">
              <Breadcrumb />
            </div>
          )}
          
          <Content
            style={{
              padding: isMobile ? 16 : 24,
              margin: 0,
              minHeight: 280,
              background: colorBgContainer,
              borderRadius: borderRadiusLG,
              overflow: 'auto',
            }}
            className="layout-transition"
          >
            <ErrorBoundary>
              {children}
            </ErrorBoundary>
          </Content>
        </Layout>
      </Layout>
      
      {showFooter && <Footer />}
    </Layout>
  );
};

export default MainLayout;