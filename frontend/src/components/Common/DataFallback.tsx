import React, { useState, useEffect } from 'react';
import { Result, Button, Card, Alert, Space, Typography, Tag, Divider } from 'antd';
import { 
  WifiOutlined, 
  DisconnectOutlined, 
  ReloadOutlined, 
  ClockCircleOutlined,
  WarningOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';

const { Text, Paragraph } = Typography;

// Network status hook
export const useNetworkStatus = () => {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [lastOnline, setLastOnline] = useState<Date | null>(null);

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      setLastOnline(new Date());
    };

    const handleOffline = () => {
      setIsOnline(false);
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return { isOnline, lastOnline };
};

// Cache management
interface CacheEntry<T> {
  data: T;
  timestamp: number;
  expiresAt: number;
}

class DataCache {
  private static instance: DataCache;
  private cache = new Map<string, CacheEntry<any>>();
  private readonly DEFAULT_TTL = 5 * 60 * 1000; // 5 minutes

  static getInstance(): DataCache {
    if (!DataCache.instance) {
      DataCache.instance = new DataCache();
    }
    return DataCache.instance;
  }

  set<T>(key: string, data: T, ttl: number = this.DEFAULT_TTL): void {
    const entry: CacheEntry<T> = {
      data,
      timestamp: Date.now(),
      expiresAt: Date.now() + ttl,
    };
    this.cache.set(key, entry);
  }

  get<T>(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return null;
    }

    return entry.data;
  }

  has(key: string): boolean {
    const entry = this.cache.get(key);
    if (!entry) return false;
    
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return false;
    }
    
    return true;
  }

  getWithMetadata<T>(key: string): { data: T; timestamp: number; age: number } | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return null;
    }

    return {
      data: entry.data,
      timestamp: entry.timestamp,
      age: Date.now() - entry.timestamp,
    };
  }

  clear(): void {
    this.cache.clear();
  }

  size(): number {
    return this.cache.size;
  }

  keys(): string[] {
    return Array.from(this.cache.keys());
  }
}

// Data fallback component props
interface DataFallbackProps {
  error?: Error | string | null;
  loading?: boolean;
  data?: any;
  cacheKey?: string;
  children: React.ReactNode;
  onRetry?: () => void;
  fallbackData?: any;
  showCacheInfo?: boolean;
  allowStaleData?: boolean;
  staleDataWarning?: string;
  className?: string;
}

// Main data fallback component
export const DataFallback: React.FC<DataFallbackProps> = ({
  error,
  loading,
  data,
  cacheKey,
  children,
  onRetry,
  fallbackData,
  showCacheInfo = false,
  allowStaleData = true,
  staleDataWarning = '显示的是缓存数据，可能不是最新的',
  className,
}) => {
  const { isOnline } = useNetworkStatus();
  const [cache] = useState(() => DataCache.getInstance());
  const [usingCachedData, setUsingCachedData] = useState(false);
  const [cacheMetadata, setCacheMetadata] = useState<{ timestamp: number; age: number } | null>(null);

  // Try to get cached data when there's an error or no data
  const getCachedData = () => {
    if (!cacheKey) return null;
    
    const cachedResult = cache.getWithMetadata(cacheKey);
    if (cachedResult) {
      setCacheMetadata({ timestamp: cachedResult.timestamp, age: cachedResult.age });
      return cachedResult.data;
    }
    
    return null;
  };

  // Cache successful data
  useEffect(() => {
    if (data && cacheKey && !error && !loading) {
      cache.set(cacheKey, data);
      setUsingCachedData(false);
      setCacheMetadata(null);
    }
  }, [data, cacheKey, error, loading, cache]);

  // Determine what data to show
  const getDisplayData = () => {
    // If we have fresh data, use it
    if (data && !error && !loading) {
      return { data, isCached: false };
    }

    // If there's an error or no data, try cached data
    if ((error || !data) && allowStaleData) {
      const cachedData = getCachedData();
      if (cachedData) {
        setUsingCachedData(true);
        return { data: cachedData, isCached: true };
      }
    }

    // Fall back to provided fallback data
    if (fallbackData) {
      return { data: fallbackData, isCached: false };
    }

    return { data: null, isCached: false };
  };

  const { data: displayData, isCached } = getDisplayData();

  // Format cache age
  const formatAge = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) return `${hours}小时前`;
    if (minutes > 0) return `${minutes}分钟前`;
    return `${seconds}秒前`;
  };

  // Network status indicator
  const NetworkStatus: React.FC = () => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      {isOnline ? (
        <>
          <WifiOutlined style={{ color: '#52c41a' }} />
          <Text type="success">在线</Text>
        </>
      ) : (
        <>
          <DisconnectOutlined style={{ color: '#ff4d4f' }} />
          <Text type="danger">离线</Text>
        </>
      )}
    </div>
  );

  // Cache info component
  const CacheInfo: React.FC = () => {
    if (!showCacheInfo || !cacheMetadata) return null;

    return (
      <Card size="small" style={{ marginBottom: 16 }}>
        <Space split={<Divider type="vertical" />}>
          <Space>
            <ClockCircleOutlined />
            <Text type="secondary">缓存时间: {formatAge(cacheMetadata.age)}</Text>
          </Space>
          <Space>
            <InfoCircleOutlined />
            <Text type="secondary">缓存大小: {cache.size()} 项</Text>
          </Space>
        </Space>
      </Card>
    );
  };

  // Error state with fallback options
  if (error && !displayData) {
    const errorMessage = typeof error === 'string' ? error : error.message;
    
    return (
      <div className={className}>
        <Result
          status="error"
          title="数据加载失败"
          subTitle={
            <Space direction="vertical" align="center">
              <Text>{errorMessage}</Text>
              <NetworkStatus />
            </Space>
          }
          extra={[
            onRetry && (
              <Button type="primary" key="retry" icon={<ReloadOutlined />} onClick={onRetry}>
                重试
              </Button>
            ),
            !isOnline && (
              <Button key="offline" disabled>
                等待网络连接
              </Button>
            ),
          ].filter(Boolean)}
        />
      </div>
    );
  }

  // Loading state
  if (loading && !displayData) {
    return <div className={className}>{children}</div>;
  }

  // Success state with data
  if (displayData) {
    return (
      <div className={className}>
        {/* Show warning for cached/stale data */}
        {isCached && (
          <Alert
            message="使用缓存数据"
            description={staleDataWarning}
            type="warning"
            icon={<WarningOutlined />}
            showIcon
            style={{ marginBottom: 16 }}
            action={
              onRetry && (
                <Button size="small" type="text" onClick={onRetry}>
                  刷新
                </Button>
              )
            }
          />
        )}

        {/* Show offline warning */}
        {!isOnline && (
          <Alert
            message="离线模式"
            description="当前处于离线状态，显示的数据可能不是最新的"
            type="info"
            icon={<DisconnectOutlined />}
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}

        {/* Cache information */}
        <CacheInfo />

        {/* Render children with data */}
        {React.cloneElement(children as React.ReactElement, { data: displayData })}

        {/* Network status in corner */}
        {!isOnline && (
          <div style={{ 
            position: 'fixed', 
            bottom: 16, 
            right: 16, 
            zIndex: 1000 
          }}>
            <Tag color="red" icon={<DisconnectOutlined />}>
              离线
            </Tag>
          </div>
        )}
      </div>
    );
  }

  // Empty state
  return (
    <div className={className}>
      <Result
        status="404"
        title="暂无数据"
        subTitle="没有可显示的数据"
        extra={
          onRetry && (
            <Button type="primary" icon={<ReloadOutlined />} onClick={onRetry}>
              重新加载
            </Button>
          )
        }
      />
    </div>
  );
};

// Hook for data caching
export const useDataCache = <T,>(key: string, ttl?: number) => {
  const [cache] = useState(() => DataCache.getInstance());

  const setCache = (data: T) => {
    cache.set(key, data, ttl);
  };

  const getCache = (): T | null => {
    return cache.get<T>(key);
  };

  const hasCache = (): boolean => {
    return cache.has(key);
  };

  const getCacheWithMetadata = () => {
    return cache.getWithMetadata<T>(key);
  };

  const clearCache = () => {
    cache.clear();
  };

  return {
    setCache,
    getCache,
    hasCache,
    getCacheWithMetadata,
    clearCache,
  };
};

// Hook for offline-aware data fetching
export const useOfflineData = <T,>(
  fetchFn: () => Promise<T>,
  cacheKey: string,
  options: {
    ttl?: number;
    retryInterval?: number;
    maxRetries?: number;
  } = {}
) => {
  const { isOnline } = useNetworkStatus();
  const { setCache, getCache, getCacheWithMetadata } = useDataCache<T>(cacheKey, options.ttl);
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  const fetchData = async () => {
    if (!isOnline) {
      // Try to use cached data when offline
      const cachedData = getCache();
      if (cachedData) {
        setData(cachedData);
        return;
      }
    }

    setLoading(true);
    setError(null);

    try {
      const result = await fetchFn();
      setData(result);
      setCache(result);
      setRetryCount(0);
    } catch (err) {
      setError(err as Error);
      
      // Try cached data on error
      const cachedData = getCache();
      if (cachedData) {
        setData(cachedData);
      }

      // Auto-retry logic
      if (options.maxRetries && retryCount < options.maxRetries) {
        setTimeout(() => {
          setRetryCount(prev => prev + 1);
          fetchData();
        }, options.retryInterval || 5000);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [isOnline]); // Refetch when coming back online

  const retry = () => {
    setRetryCount(0);
    fetchData();
  };

  const cacheMetadata = getCacheWithMetadata();

  return {
    data,
    loading,
    error,
    retry,
    isOnline,
    isCached: !!cacheMetadata && data === cacheMetadata.data,
    cacheAge: cacheMetadata?.age,
  };
};

export default DataFallback;