import React, { useEffect } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { Spin } from 'antd';
import { useUserStore } from '@/stores/userStore';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredRole?: string;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ 
  children, 
  requiredRole 
}) => {
  const location = useLocation();
  const { isAuthenticated, user, checkAuth } = useUserStore();
  const [checking, setChecking] = React.useState(true);

  useEffect(() => {
    // 在开发环境中跳过认证检查
    if (import.meta.env.DEV) {
      console.warn('[DEV] Skipping auth check in development mode');
      setChecking(false);
      return;
    }
    
    const verifyAuth = async () => {
      try {
        await checkAuth();
      } catch (error) {
        console.error('Auth check failed:', error);
      } finally {
        setChecking(false);
      }
    };
    
    // Add timeout to prevent infinite loading
    const timeout = setTimeout(() => {
      console.warn('Auth check timeout, setting checking to false');
      setChecking(false);
    }, 5000); // 5 second timeout
    
    verifyAuth().finally(() => {
      clearTimeout(timeout);
    });
  }, [checkAuth]);

  if (checking) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh' 
      }}>
        <Spin size="large" tip="验证身份中..." />
      </div>
    );
  }

  if (!isAuthenticated) {
    // 开发环境下跳过认证
    if (import.meta.env.DEV) {
      console.warn('[DEV] Bypassing authentication check for development');
      return <>{children}</>;
    }
    // 保存用户原本想访问的路径
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  // 如果需要特定角色
  if (requiredRole && user?.is_superuser !== true) {
    return <Navigate to="/403" replace />;
  }

  return <>{children}</>;
};

export default ProtectedRoute;