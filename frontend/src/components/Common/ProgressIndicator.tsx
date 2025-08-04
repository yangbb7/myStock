import React, { useState, useEffect } from 'react';
import { Progress, Card, Space, Typography, Button, Spin } from 'antd';
import { 
  CheckCircleOutlined, 
  CloseCircleOutlined, 
  LoadingOutlined,
  ReloadOutlined,
  PauseCircleOutlined
} from '@ant-design/icons';

const { Text, Title } = Typography;

// Progress step interface
export interface ProgressStep {
  id: string;
  title: string;
  description?: string;
  status: 'waiting' | 'running' | 'completed' | 'error' | 'paused';
  progress?: number;
  error?: string;
  duration?: number;
  startTime?: number;
  endTime?: number;
}

// Progress indicator props
interface ProgressIndicatorProps {
  steps: ProgressStep[];
  currentStep?: string;
  title?: string;
  showOverallProgress?: boolean;
  showStepDetails?: boolean;
  onRetry?: (stepId: string) => void;
  onPause?: (stepId: string) => void;
  onResume?: (stepId: string) => void;
  onCancel?: () => void;
  className?: string;
}

// Individual step component
const StepItem: React.FC<{
  step: ProgressStep;
  isActive: boolean;
  onRetry?: (stepId: string) => void;
  onPause?: (stepId: string) => void;
  onResume?: (stepId: string) => void;
}> = ({ step, isActive, onRetry, onPause, onResume }) => {
  const getStatusIcon = () => {
    switch (step.status) {
      case 'completed':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'error':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'running':
        return <LoadingOutlined style={{ color: '#1890ff' }} />;
      case 'paused':
        return <PauseCircleOutlined style={{ color: '#faad14' }} />;
      default:
        return <div style={{ width: 14, height: 14, borderRadius: '50%', backgroundColor: '#d9d9d9' }} />;
    }
  };

  const getStatusColor = () => {
    switch (step.status) {
      case 'completed':
        return '#52c41a';
      case 'error':
        return '#ff4d4f';
      case 'running':
        return '#1890ff';
      case 'paused':
        return '#faad14';
      default:
        return '#d9d9d9';
    }
  };

  const formatDuration = (ms?: number) => {
    if (!ms) return '';
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    }
    return `${seconds}s`;
  };

  const getDuration = () => {
    if (step.endTime && step.startTime) {
      return step.endTime - step.startTime;
    }
    if (step.startTime && step.status === 'running') {
      return Date.now() - step.startTime;
    }
    return step.duration;
  };

  return (
    <div 
      className={`progress-step ${isActive ? 'active' : ''}`}
      style={{
        padding: '12px 16px',
        border: `1px solid ${isActive ? getStatusColor() : '#f0f0f0'}`,
        borderRadius: 6,
        marginBottom: 8,
        backgroundColor: isActive ? '#fafafa' : 'white',
      }}
    >
      <Space align="start" style={{ width: '100%' }}>
        <div style={{ marginTop: 2 }}>
          {getStatusIcon()}
        </div>
        
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Text strong={isActive}>{step.title}</Text>
            {getDuration() && (
              <Text type="secondary" style={{ fontSize: 12 }}>
                {formatDuration(getDuration())}
              </Text>
            )}
          </div>
          
          {step.description && (
            <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 4 }}>
              {step.description}
            </Text>
          )}
          
          {step.status === 'running' && typeof step.progress === 'number' && (
            <Progress 
              percent={step.progress} 
              size="small" 
              style={{ marginTop: 8 }}
              strokeColor={getStatusColor()}
            />
          )}
          
          {step.status === 'error' && step.error && (
            <div style={{ marginTop: 8 }}>
              <Text type="danger" style={{ fontSize: 12 }}>
                {step.error}
              </Text>
              {onRetry && (
                <Button 
                  type="link" 
                  size="small" 
                  icon={<ReloadOutlined />}
                  onClick={() => onRetry(step.id)}
                  style={{ padding: 0, marginLeft: 8 }}
                >
                  重试
                </Button>
              )}
            </div>
          )}
          
          {step.status === 'running' && (
            <div style={{ marginTop: 8 }}>
              <Space size="small">
                {onPause && (
                  <Button 
                    type="link" 
                    size="small"
                    onClick={() => onPause(step.id)}
                    style={{ padding: 0 }}
                  >
                    暂停
                  </Button>
                )}
              </Space>
            </div>
          )}
          
          {step.status === 'paused' && (
            <div style={{ marginTop: 8 }}>
              {onResume && (
                <Button 
                  type="link" 
                  size="small"
                  onClick={() => onResume(step.id)}
                  style={{ padding: 0 }}
                >
                  继续
                </Button>
              )}
            </div>
          )}
        </div>
      </Space>
    </div>
  );
};

// Main progress indicator component
export const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({
  steps,
  currentStep,
  title = '操作进度',
  showOverallProgress = true,
  showStepDetails = true,
  onRetry,
  onPause,
  onResume,
  onCancel,
  className,
}) => {
  const [elapsedTime, setElapsedTime] = useState(0);
  const [startTime] = useState(Date.now());

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedTime(Date.now() - startTime);
    }, 1000);

    return () => clearInterval(timer);
  }, [startTime]);

  // Calculate overall progress
  const completedSteps = steps.filter(step => step.status === 'completed').length;
  const totalSteps = steps.length;
  const overallProgress = totalSteps > 0 ? (completedSteps / totalSteps) * 100 : 0;

  // Get current step info
  const activeStep = currentStep ? steps.find(step => step.id === currentStep) : 
                   steps.find(step => step.status === 'running');

  // Check if all steps are completed
  const isCompleted = steps.every(step => step.status === 'completed');
  const hasErrors = steps.some(step => step.status === 'error');

  const formatElapsedTime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) {
      return `${hours}:${(minutes % 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;
    }
    return `${minutes}:${(seconds % 60).toString().padStart(2, '0')}`;
  };

  return (
    <Card className={className} style={{ width: '100%' }}>
      <div style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Title level={4} style={{ margin: 0 }}>
            {title}
          </Title>
          <Text type="secondary">
            {formatElapsedTime(elapsedTime)}
          </Text>
        </div>
        
        {showOverallProgress && (
          <div style={{ marginTop: 12 }}>
            <Progress 
              percent={Math.round(overallProgress)}
              status={hasErrors ? 'exception' : isCompleted ? 'success' : 'active'}
              strokeColor={hasErrors ? '#ff4d4f' : isCompleted ? '#52c41a' : '#1890ff'}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                {completedSteps} / {totalSteps} 步骤完成
              </Text>
              {activeStep && (
                <Text type="secondary" style={{ fontSize: 12 }}>
                  当前: {activeStep.title}
                </Text>
              )}
            </div>
          </div>
        )}
      </div>

      {showStepDetails && (
        <div className="progress-steps">
          {steps.map(step => (
            <StepItem
              key={step.id}
              step={step}
              isActive={step.id === currentStep || step.status === 'running'}
              onRetry={onRetry}
              onPause={onPause}
              onResume={onResume}
            />
          ))}
        </div>
      )}

      {(isCompleted || hasErrors) && (
        <div style={{ marginTop: 16, textAlign: 'center' }}>
          <Space>
            {hasErrors && onRetry && (
              <Button 
                type="primary" 
                icon={<ReloadOutlined />}
                onClick={() => {
                  const errorStep = steps.find(step => step.status === 'error');
                  if (errorStep) onRetry(errorStep.id);
                }}
              >
                重试失败步骤
              </Button>
            )}
            {onCancel && (
              <Button onClick={onCancel}>
                {isCompleted ? '关闭' : '取消'}
              </Button>
            )}
          </Space>
        </div>
      )}
    </Card>
  );
};

// Simple loading indicator with progress
export const SimpleProgressIndicator: React.FC<{
  loading: boolean;
  progress?: number;
  title?: string;
  description?: string;
}> = ({ loading, progress, title = '加载中...', description }) => {
  if (!loading) return null;

  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      padding: '40px 20px',
      textAlign: 'center'
    }}>
      <Spin size="large" />
      <Title level={4} style={{ marginTop: 16, marginBottom: 8 }}>
        {title}
      </Title>
      {description && (
        <Text type="secondary" style={{ marginBottom: 16 }}>
          {description}
        </Text>
      )}
      {typeof progress === 'number' && (
        <Progress 
          percent={progress} 
          style={{ width: 300 }}
          strokeColor="#1890ff"
        />
      )}
    </div>
  );
};

// Hook for managing progress steps
export const useProgressSteps = (initialSteps: Omit<ProgressStep, 'status'>[]) => {
  const [steps, setSteps] = useState<ProgressStep[]>(
    initialSteps.map(step => ({ ...step, status: 'waiting' as const }))
  );
  const [currentStepId, setCurrentStepId] = useState<string | undefined>();

  const updateStep = (stepId: string, updates: Partial<ProgressStep>) => {
    setSteps(prev => prev.map(step => 
      step.id === stepId ? { ...step, ...updates } : step
    ));
  };

  const startStep = (stepId: string) => {
    updateStep(stepId, { 
      status: 'running', 
      startTime: Date.now(),
      progress: 0 
    });
    setCurrentStepId(stepId);
  };

  const completeStep = (stepId: string) => {
    updateStep(stepId, { 
      status: 'completed', 
      endTime: Date.now(),
      progress: 100 
    });
  };

  const errorStep = (stepId: string, error: string) => {
    updateStep(stepId, { 
      status: 'error', 
      error,
      endTime: Date.now() 
    });
  };

  const updateProgress = (stepId: string, progress: number) => {
    updateStep(stepId, { progress });
  };

  const resetSteps = () => {
    setSteps(prev => prev.map(step => ({ 
      ...step, 
      status: 'waiting' as const,
      progress: undefined,
      error: undefined,
      startTime: undefined,
      endTime: undefined
    })));
    setCurrentStepId(undefined);
  };

  return {
    steps,
    currentStepId,
    updateStep,
    startStep,
    completeStep,
    errorStep,
    updateProgress,
    resetSteps,
  };
};

export default ProgressIndicator;