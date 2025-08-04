import React, { useState, useEffect } from 'react';
import { notification, message, Badge, Tooltip, Button } from 'antd';
import { 
  WifiOutlined, 
  DisconnectOutlined, 
  SyncOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons';

// Network quality levels
type NetworkQuality = 'excellent' | 'good' | 'fair' | 'poor' | 'offline';

// Connection info interface
interface ConnectionInfo {
  isOnline: boolean;
  quality: NetworkQuality;
  latency: number;
  lastCheck: Date;
  downlink?: number;
  effectiveType?: string;
  rtt?: number;
}

// Network monitor hook
export const useNetworkMonitor = () => {
  const [connectionInfo, setConnectionInfo] = useState<ConnectionInfo>({
    isOnline: navigator.onLine,
    quality: 'good',
    latency: 0,
    lastCheck: new Date(),
  });
  const [isChecking, setIsChecking] = useState(false);

  // Check network quality
  const checkNetworkQuality = async (): Promise<{ latency: number; quality: NetworkQuality }> => {
    if (!navigator.onLine) {
      return { latency: Infinity, quality: 'offline' };
    }

    const startTime = performance.now();
    
    try {
      // Use a small image or API endpoint to test connectivity
      const response = await fetch('/health', {
        method: 'GET',
        cache: 'no-cache',
      });
      
      const endTime = performance.now();
      const latency = endTime - startTime;

      let quality: NetworkQuality;
      if (latency < 100) quality = 'excellent';
      else if (latency < 300) quality = 'good';
      else if (latency < 1000) quality = 'fair';
      else quality = 'poor';

      return { latency, quality };
    } catch (error) {
      return { latency: Infinity, quality: 'poor' };
    }
  };

  // Get connection info from Navigator API
  const getConnectionInfo = () => {
    const connection = (navigator as any).connection || 
                     (navigator as any).mozConnection || 
                     (navigator as any).webkitConnection;
    
    if (connection) {
      return {
        downlink: connection.downlink,
        effectiveType: connection.effectiveType,
        rtt: connection.rtt,
      };
    }
    
    return {};
  };

  // Update connection info
  const updateConnectionInfo = async () => {
    setIsChecking(true);
    
    const { latency, quality } = await checkNetworkQuality();
    const additionalInfo = getConnectionInfo();
    
    setConnectionInfo(prev => ({
      ...prev,
      isOnline: navigator.onLine,
      quality,
      latency,
      lastCheck: new Date(),
      ...additionalInfo,
    }));
    
    setIsChecking(false);
  };

  // Handle online/offline events
  useEffect(() => {
    const handleOnline = () => {
      updateConnectionInfo();
      // Removed auto-notification to reduce popup noise
      // Users can still see network status in the indicator
    };

    const handleOffline = () => {
      setConnectionInfo(prev => ({
        ...prev,
        isOnline: false,
        quality: 'offline',
        lastCheck: new Date(),
      }));
      
      // Removed auto-notification to reduce popup noise
      // Users can still see network status in the indicator
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Initial check
    updateConnectionInfo();

    // Periodic checks
    const interval = setInterval(updateConnectionInfo, 30000); // Check every 30 seconds

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
      clearInterval(interval);
    };
  }, []);

  return {
    connectionInfo,
    isChecking,
    checkNetwork: updateConnectionInfo,
  };
};

// Network status indicator component
export const NetworkStatusIndicator: React.FC<{
  showDetails?: boolean;
  position?: 'topRight' | 'bottomRight' | 'topLeft' | 'bottomLeft';
}> = ({ 
  showDetails = false, 
  position = 'topRight' 
}) => {
  const { connectionInfo, isChecking, checkNetwork } = useNetworkMonitor();

  const getStatusColor = () => {
    switch (connectionInfo.quality) {
      case 'excellent':
        return '#52c41a';
      case 'good':
        return '#1890ff';
      case 'fair':
        return '#faad14';
      case 'poor':
        return '#ff7a45';
      case 'offline':
        return '#ff4d4f';
      default:
        return '#d9d9d9';
    }
  };

  const getStatusIcon = () => {
    if (isChecking) {
      return <SyncOutlined spin />;
    }
    
    switch (connectionInfo.quality) {
      case 'offline':
        return <DisconnectOutlined />;
      case 'poor':
        return <ExclamationCircleOutlined />;
      default:
        return <WifiOutlined />;
    }
  };

  const getStatusText = () => {
    switch (connectionInfo.quality) {
      case 'excellent':
        return '网络优秀';
      case 'good':
        return '网络良好';
      case 'fair':
        return '网络一般';
      case 'poor':
        return '网络较差';
      case 'offline':
        return '离线';
      default:
        return '未知';
    }
  };

  const formatLatency = (latency: number) => {
    if (latency === Infinity) return '∞';
    return `${Math.round(latency)}ms`;
  };

  const getPositionStyle = () => {
    const baseStyle = {
      position: 'fixed' as const,
      zIndex: 1000,
      padding: '8px 12px',
      borderRadius: '6px',
      backgroundColor: 'rgba(255, 255, 255, 0.9)',
      backdropFilter: 'blur(4px)',
      border: '1px solid #f0f0f0',
      boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
    };

    switch (position) {
      case 'topRight':
        return { ...baseStyle, top: 16, right: 16 };
      case 'bottomRight':
        return { ...baseStyle, bottom: 16, right: 16 };
      case 'topLeft':
        return { ...baseStyle, top: 16, left: 16 };
      case 'bottomLeft':
        return { ...baseStyle, bottom: 16, left: 16 };
      default:
        return { ...baseStyle, top: 16, right: 16 };
    }
  };

  const tooltipContent = (
    <div>
      <div>状态: {getStatusText()}</div>
      <div>延迟: {formatLatency(connectionInfo.latency)}</div>
      {connectionInfo.effectiveType && (
        <div>连接类型: {connectionInfo.effectiveType}</div>
      )}
      {connectionInfo.downlink && (
        <div>下行速度: {connectionInfo.downlink} Mbps</div>
      )}
      <div>最后检查: {connectionInfo.lastCheck.toLocaleTimeString()}</div>
    </div>
  );

  if (!showDetails) {
    return (
      <Tooltip title={tooltipContent}>
        <Badge 
          status={connectionInfo.isOnline ? 'success' : 'error'}
          style={getPositionStyle()}
        >
          <span style={{ color: getStatusColor(), fontSize: 16 }}>
            {getStatusIcon()}
          </span>
        </Badge>
      </Tooltip>
    );
  }

  return (
    <div style={getPositionStyle()}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ color: getStatusColor(), fontSize: 16 }}>
          {getStatusIcon()}
        </span>
        <div>
          <div style={{ fontSize: 12, fontWeight: 500 }}>
            {getStatusText()}
          </div>
          <div style={{ fontSize: 11, color: '#666' }}>
            {formatLatency(connectionInfo.latency)}
          </div>
        </div>
        <Button 
          type="text" 
          size="small" 
          onClick={checkNetwork}
          loading={isChecking}
          style={{ padding: 0, minWidth: 'auto' }}
        >
          <SyncOutlined />
        </Button>
      </div>
    </div>
  );
};

// Data sync status component
export const DataSyncStatus: React.FC<{
  lastSync?: Date;
  syncInProgress?: boolean;
  syncError?: string;
  onSync?: () => void;
}> = ({ 
  lastSync, 
  syncInProgress = false, 
  syncError, 
  onSync 
}) => {
  const { connectionInfo } = useNetworkMonitor();

  const formatSyncTime = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days}天前`;
    if (hours > 0) return `${hours}小时前`;
    if (minutes > 0) return `${minutes}分钟前`;
    return '刚刚';
  };

  const getSyncStatus = () => {
    if (syncInProgress) return { text: '同步中...', color: '#1890ff' };
    if (syncError) return { text: '同步失败', color: '#ff4d4f' };
    if (!connectionInfo.isOnline) return { text: '离线', color: '#faad14' };
    if (lastSync) return { text: `已同步 (${formatSyncTime(lastSync)})`, color: '#52c41a' };
    return { text: '未同步', color: '#d9d9d9' };
  };

  const status = getSyncStatus();

  return (
    <div style={{ 
      display: 'flex', 
      alignItems: 'center', 
      gap: 8,
      padding: '4px 8px',
      borderRadius: 4,
      backgroundColor: '#fafafa',
      border: '1px solid #f0f0f0'
    }}>
      <Badge 
        status={syncInProgress ? 'processing' : syncError ? 'error' : 'success'} 
      />
      <span style={{ fontSize: 12, color: status.color }}>
        {status.text}
      </span>
      {onSync && connectionInfo.isOnline && !syncInProgress && (
        <Button 
          type="text" 
          size="small" 
          onClick={onSync}
          style={{ padding: 0, minWidth: 'auto', fontSize: 12 }}
        >
          <SyncOutlined />
        </Button>
      )}
    </div>
  );
};

// Connection quality meter
export const ConnectionQualityMeter: React.FC = () => {
  const { connectionInfo } = useNetworkMonitor();

  const getQualityPercentage = () => {
    switch (connectionInfo.quality) {
      case 'excellent':
        return 100;
      case 'good':
        return 75;
      case 'fair':
        return 50;
      case 'poor':
        return 25;
      case 'offline':
        return 0;
      default:
        return 0;
    }
  };

  const getQualityColor = () => {
    const percentage = getQualityPercentage();
    if (percentage >= 75) return '#52c41a';
    if (percentage >= 50) return '#1890ff';
    if (percentage >= 25) return '#faad14';
    return '#ff4d4f';
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ display: 'flex', gap: 2 }}>
        {[1, 2, 3, 4].map(level => (
          <div
            key={level}
            style={{
              width: 4,
              height: 8 + level * 2,
              backgroundColor: getQualityPercentage() >= level * 25 ? getQualityColor() : '#f0f0f0',
              borderRadius: 1,
            }}
          />
        ))}
      </div>
      <span style={{ fontSize: 12, color: '#666' }}>
        {Math.round(connectionInfo.latency)}ms
      </span>
    </div>
  );
};

export default NetworkStatusIndicator;