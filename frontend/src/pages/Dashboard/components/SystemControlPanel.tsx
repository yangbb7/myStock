import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Button, 
  Space, 
  Typography, 
  Modal, 
  Alert,
  Tooltip,

  message
} from 'antd';
import { 
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  SettingOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined
} from '@ant-design/icons';
import { useSystemHealth, useSystemControl } from '../../../hooks/useApi';

const { Text, Title } = Typography;

const SystemControlPanel: React.FC = () => {
  const [confirmModalVisible, setConfirmModalVisible] = useState(false);
  const [pendingAction, setPendingAction] = useState<'start' | 'stop' | 'restart' | null>(null);
  
  const systemHealthResult = useSystemHealth();
  const { data: systemHealth } = systemHealthResult;
  
  // Debug logging
  useEffect(() => {
    console.log('[SystemControl] systemHealthResult:', systemHealthResult);
    
    // 添加手动API测试
    const testAPI = async () => {
      try {
        const response = await fetch('http://localhost:8000/health');
        const data = await response.json();
        console.log('[SystemControl] Direct API test result:', data);
      } catch (error) {
        console.error('[SystemControl] Direct API test error:', error);
      }
    };
    
    testAPI();
  }, [systemHealthResult]);
  const { startSystem, stopSystem, restartSystem } = useSystemControl();

  const isSystemRunning = systemHealth?.systemRunning || false;
  const isLoading = startSystem.isPending || stopSystem.isPending || restartSystem.isPending;

  // 添加调试日志和重置机制
  useEffect(() => {
    console.log('[SystemControl] Component mounted, mutation states:', {
      startSystemPending: startSystem.isPending,
      stopSystemPending: stopSystem.isPending,
      restartSystemPending: restartSystem.isPending,
      isLoading,
      systemHealth: systemHealth,
      isSystemRunning: isSystemRunning
    });
    
    // 添加全局测试函数
    (window as any).testSystemControl = () => {
      console.log('🧪 Testing system control...');
      showConfirmModal('start');
    };
  }, [startSystem.isPending, stopSystem.isPending, restartSystem.isPending, isLoading, systemHealth, isSystemRunning]);

  const handleSystemAction = async (action: 'start' | 'stop' | 'restart') => {
    console.log(`[SystemControl] Attempting to ${action} system`);
    
    try {
      let result;
      switch (action) {
        case 'start':
          console.log('[SystemControl] Calling startSystem mutation');
          result = await startSystem.mutateAsync(undefined);
          console.log('[SystemControl] Start system result:', result);
          message.success('系统启动成功');
          break;
        case 'stop':
          console.log('[SystemControl] Calling stopSystem mutation');
          result = await stopSystem.mutateAsync(undefined);
          console.log('[SystemControl] Stop system result:', result);
          message.success('系统停止成功');
          break;
        case 'restart':
          console.log('[SystemControl] Calling restartSystem mutation');
          result = await restartSystem.mutateAsync(undefined);
          console.log('[SystemControl] Restart system result:', result);
          message.success('系统重启成功');
          break;
      }
    } catch (error: any) {
      console.error(`[SystemControl] System ${action} failed:`, error);
      const errorMessage = error?.message || error?.detail || error?.toString() || '未知错误';
      message.error(`系统${action === 'start' ? '启动' : action === 'stop' ? '停止' : '重启'}失败: ${errorMessage}`);
    }
  };

  const showConfirmModal = (action: 'start' | 'stop' | 'restart') => {
    console.log(`[SystemControl] Showing confirm modal for action: ${action}`);
    console.log(`[SystemControl] Current states:`, {
      isSystemRunning,
      isLoading,
      systemHealth: systemHealth,
      confirmModalVisible,
      pendingAction
    });
    setPendingAction(action);
    setConfirmModalVisible(true);
  };

  const handleConfirm = () => {
    console.log(`[SystemControl] Confirm clicked, pending action: ${pendingAction}`);
    if (pendingAction) {
      handleSystemAction(pendingAction);
    }
    setConfirmModalVisible(false);
    setPendingAction(null);
  };

  const handleCancel = () => {
    setConfirmModalVisible(false);
    setPendingAction(null);
  };

  const handleResetMutations = () => {
    console.log('[SystemControl] Resetting mutations manually');
    startSystem.reset();
    stopSystem.reset();
    restartSystem.reset();
  };

  const getActionText = (action: string) => {
    switch (action) {
      case 'start':
        return '启动';
      case 'stop':
        return '停止';
      case 'restart':
        return '重启';
      default:
        return '';
    }
  };

  const getActionDescription = (action: string) => {
    switch (action) {
      case 'start':
        return '启动系统将初始化所有模块并开始交易服务。';
      case 'stop':
        return '停止系统将安全关闭所有模块，停止交易活动。';
      case 'restart':
        return '重启系统将先停止再启动所有模块，可能需要几分钟时间。';
      default:
        return '';
    }
  };

  return (
    <>
      <Card 
        size="small"
        style={{ 
          background: isSystemRunning ? '#f6ffed' : '#fff2f0',
          borderColor: isSystemRunning ? '#b7eb8f' : '#ffccc7'
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          {/* System Status */}
          <Space size="large">
            <Space>
              {isSystemRunning ? (
                <CheckCircleOutlined style={{ color: '#52c41a', fontSize: '20px' }} />
              ) : (
                <ExclamationCircleOutlined style={{ color: '#ff4d4f', fontSize: '20px' }} />
              )}
              <div>
                <Title level={5} style={{ margin: 0, color: isSystemRunning ? '#52c41a' : '#ff4d4f' }}>
                  系统状态: {isSystemRunning ? '运行中' : '已停止'}
                </Title>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  {isSystemRunning ? '所有模块正常运行，交易服务可用' : '系统已停止，交易服务不可用'}
                </Text>
              </div>
            </Space>
          </Space>

          {/* Control Buttons */}
          <Space>
            {(() => {
              console.log('[SystemControl] Rendering buttons, isSystemRunning:', isSystemRunning, 'systemHealth:', systemHealth);
              return !isSystemRunning ? (
                <Tooltip title="启动系统并初始化所有模块">
                  <Button
                    type="primary"
                    icon={<PlayCircleOutlined />}
                    loading={isLoading}
                    onClick={() => {
                      console.log('[SystemControl] Start button clicked!');
                      showConfirmModal('start');
                    }}
                    size="large"
                  >
                    启动系统
                  </Button>
                </Tooltip>
              ) : (
                <Tooltip title="安全停止系统和所有模块">
                  <Button
                    danger
                    icon={<PauseCircleOutlined />}
                    loading={isLoading}
                    onClick={() => {
                      console.log('[SystemControl] Stop button clicked!');
                      showConfirmModal('stop');
                    }}
                    size="large"
                  >
                    停止系统
                  </Button>
                </Tooltip>
              );
            })()}

            <Tooltip title="重启系统以应用配置更改">
              <Button
                icon={<ReloadOutlined />}
                loading={isLoading}
                onClick={() => showConfirmModal('restart')}
                disabled={!isSystemRunning}
              >
                重启
              </Button>
            </Tooltip>

            <Tooltip title="系统配置和管理">
              <Button
                icon={<SettingOutlined />}
                onClick={() => {
                  // Navigate to system management page
                  window.location.href = '/system';
                }}
              >
                设置
              </Button>
            </Tooltip>
            
            {/* 调试模式下显示重置按钮 */}
            {(import.meta.env.DEV || isLoading) && (
              <Tooltip title="重置加载状态（调试用）">
                <Button
                  danger
                  size="small"
                  onClick={handleResetMutations}
                  disabled={!isLoading}
                >
                  重置状态
                </Button>
              </Tooltip>
            )}
          </Space>
        </div>
      </Card>

      {/* Confirmation Modal */}
      <Modal
        title={`确认${getActionText(pendingAction || '')}`}
        open={confirmModalVisible}
        onOk={handleConfirm}
        onCancel={handleCancel}
        okText="确认"
        cancelText="取消"
        okButtonProps={{ 
          loading: isLoading,
          danger: pendingAction === 'stop'
        }}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <Alert
            message={`即将${getActionText(pendingAction || '')}系统`}
            description={getActionDescription(pendingAction || '')}
            type={pendingAction === 'stop' ? 'warning' : 'info'}
            showIcon
          />
          
          {pendingAction === 'stop' && (
            <Alert
              message="注意"
              description="停止系统将中断所有正在进行的交易活动，请确保没有重要操作正在执行。"
              type="error"
              showIcon
            />
          )}
          
          {pendingAction === 'restart' && (
            <Alert
              message="提示"
              description="重启过程可能需要1-2分钟，期间系统将暂时不可用。"
              type="warning"
              showIcon
            />
          )}
          
          <Text>
            您确定要{getActionText(pendingAction || '')}系统吗？
          </Text>
        </Space>
      </Modal>
    </>
  );
};

export default SystemControlPanel;