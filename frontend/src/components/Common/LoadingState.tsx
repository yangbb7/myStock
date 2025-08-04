import React from 'react';
import { Spin, Card, Skeleton, Result, Button, Typography } from 'antd';
import { LoadingOutlined, ReloadOutlined } from '@ant-design/icons';
import { SimpleProgressIndicator } from './ProgressIndicator';
import { useErrorHandler } from './ErrorHandler';

const { Text } = Typography;

interface LoadingStateProps {
  loading?: boolean;
  error?: Error | string | null;
  empty?: boolean;
  emptyDescription?: string;
  children: React.ReactNode;
  onRetry?: () => void;
  size?: 'small' | 'default' | 'large';
  skeleton?: boolean;
  skeletonRows?: number;
  tip?: string;
  className?: string;
  progress?: number;
  progressTitle?: string;
  progressDescription?: string;
  showProgress?: boolean;
  context?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  loading = false,
  error,
  empty = false,
  emptyDescription = '暂无数据',
  children,
  onRetry,
  size = 'default',
  skeleton = false,
  skeletonRows = 3,
  tip,
  className,
  progress,
  progressTitle,
  progressDescription,
  showProgress = false,
  context,
}) => {
  const { handleError } = useErrorHandler();
  const antIcon = <LoadingOutlined style={{ fontSize: 24 }} spin />;

  // Error state
  if (error) {
    const errorMessage = typeof error === 'string' ? error : error.message;
    
    // Log error using error handler
    React.useEffect(() => {
      if (typeof error !== 'string') {
        handleError(error, context);
      }
    }, [error, handleError, context]);
    
    return (
      <div className={className}>
        <Result
          status="error"
          title="加载失败"
          subTitle={errorMessage}
          extra={
            onRetry && (
              <Button type="primary" icon={<ReloadOutlined />} onClick={onRetry}>
                重试
              </Button>
            )
          }
        />
      </div>
    );
  }

  // Loading state
  if (loading) {
    if (skeleton) {
      return (
        <div className={className}>
          <Skeleton active paragraph={{ rows: skeletonRows }} />
        </div>
      );
    }
    
    if (showProgress && (progress !== undefined || progressTitle)) {
      return (
        <div className={className}>
          <SimpleProgressIndicator
            loading={true}
            progress={progress}
            title={progressTitle || tip || '加载中...'}
            description={progressDescription}
          />
        </div>
      );
    }
    
    return (
      <div className={className} style={{ textAlign: 'center', padding: '50px 0' }}>
        <Spin indicator={antIcon} size={size} tip={tip} />
      </div>
    );
  }

  // Empty state
  if (empty) {
    return (
      <div className={className}>
        <Result
          status="404"
          title="暂无数据"
          subTitle={emptyDescription}
          extra={
            onRetry && (
              <Button type="primary" icon={<ReloadOutlined />} onClick={onRetry}>
                刷新
              </Button>
            )
          }
        />
      </div>
    );
  }

  // Success state
  return <div className={className}>{children}</div>;
};

// Skeleton wrapper component for cards
export const SkeletonCard: React.FC<{
  loading: boolean;
  children: React.ReactNode;
  rows?: number;
}> = ({ loading, children, rows = 3 }) => {
  return (
    <Card>
      {loading ? <Skeleton active paragraph={{ rows }} /> : children}
    </Card>
  );
};

export default LoadingState;