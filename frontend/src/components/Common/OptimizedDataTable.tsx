import React, { useMemo, useCallback, useState } from 'react';
import { Table, Input, Button, Space, Card } from 'antd';
import { SearchOutlined, ReloadOutlined } from '@ant-design/icons';
import { VirtualTable } from './VirtualTable';
import { usePerformanceOptimization, useListOptimization } from '../../hooks/usePerformanceOptimization';
import { useOptimizedQuery } from '../../hooks/useOptimizedQuery';

interface OptimizedDataTableProps<T = any> {
  dataSource: T[];
  columns: any[];
  title?: string;
  searchable?: boolean;
  searchFields?: string[];
  enableVirtualization?: boolean;
  virtualizationThreshold?: number;
  pageSize?: number;
  height?: number;
  onRefresh?: () => void;
  loading?: boolean;
  rowKey?: string | ((record: T) => string);
}

export const OptimizedDataTable: React.FC<OptimizedDataTableProps> = ({
  dataSource = [],
  columns = [],
  title,
  searchable = true,
  searchFields = [],
  enableVirtualization = true,
  virtualizationThreshold = 100,
  pageSize = 50,
  height = 400,
  onRefresh,
  loading = false,
  rowKey = 'key'
}) => {
  const [searchText, setSearchText] = useState('');
  const [filteredData, setFilteredData] = useState(dataSource);

  // Performance optimization
  const optimization = usePerformanceOptimization({
    componentName: 'OptimizedDataTable',
    enableCaching: true,
    enableDebouncing: true,
    debounceDelay: 300,
    enableMemoryMonitoring: true
  });

  // List optimization
  const listOptimization = useListOptimization(filteredData, {
    pageSize,
    enableVirtualization,
    virtualizationThreshold
  });

  // Optimized search function
  const debouncedSearch = useMemo(
    () => optimization.debounce((searchValue: string) => {
      if (!searchValue.trim()) {
        setFilteredData(dataSource);
        return;
      }

      const filtered = dataSource.filter(record => {
        if (searchFields.length > 0) {
          return searchFields.some(field => 
            String(record[field]).toLowerCase().includes(searchValue.toLowerCase())
          );
        } else {
          return Object.values(record).some(value =>
            String(value).toLowerCase().includes(searchValue.toLowerCase())
          );
        }
      });

      setFilteredData(filtered);
    }, 300),
    [dataSource, searchFields, optimization]
  );

  // Handle search
  const handleSearch = useCallback((value: string) => {
    setSearchText(value);
    debouncedSearch(value);
  }, [debouncedSearch]);

  // Handle refresh
  const handleRefresh = useCallback(() => {
    setSearchText('');
    setFilteredData(dataSource);
    onRefresh?.();
  }, [dataSource, onRefresh]);

  // Update filtered data when dataSource changes
  React.useEffect(() => {
    if (!searchText) {
      setFilteredData(dataSource);
    } else {
      debouncedSearch(searchText);
    }
  }, [dataSource, searchText, debouncedSearch]);

  // Enhanced columns with performance optimizations
  const optimizedColumns = useMemo(() => {
    return columns.map(column => ({
      ...column,
      render: column.render ? optimization.throttle(column.render, 50) : undefined
    }));
  }, [columns, optimization]);

  // Table header with search and controls
  const tableHeader = useMemo(() => (
    <div style={{ 
      display: 'flex', 
      justifyContent: 'space-between', 
      alignItems: 'center',
      marginBottom: 16 
    }}>
      <div>
        {title && <h3 style={{ margin: 0 }}>{title}</h3>}
        <span style={{ color: '#666', fontSize: '12px' }}>
          显示 {listOptimization.items.length} / {dataSource.length} 条记录
        </span>
      </div>
      
      <Space>
        {searchable && (
          <Input
            placeholder="搜索..."
            prefix={<SearchOutlined />}
            value={searchText}
            onChange={(e) => handleSearch(e.target.value)}
            style={{ width: 200 }}
            allowClear
          />
        )}
        
        {onRefresh && (
          <Button
            icon={<ReloadOutlined />}
            onClick={handleRefresh}
            loading={loading}
          >
            刷新
          </Button>
        )}
      </Space>
    </div>
  ), [title, listOptimization.items.length, dataSource.length, searchable, searchText, handleSearch, onRefresh, handleRefresh, loading]);

  // Performance metrics display (development only)
  const performanceInfo = useMemo(() => {
    if (process.env.NODE_ENV !== 'development') return null;
    
    const metrics = optimization.getMetrics();
    return (
      <div style={{ 
        fontSize: '11px', 
        color: '#999', 
        marginTop: 8,
        padding: '4px 8px',
        background: '#f5f5f5',
        borderRadius: 4
      }}>
        渲染时间: {metrics.current.renderTime.toFixed(2)}ms | 
        缓存命中率: {(metrics.current.cacheHitRate * 100).toFixed(1)}% |
        {metrics.current.memoryUsage && 
          ` 内存使用: ${(metrics.current.memoryUsage.used / 1024 / 1024).toFixed(1)}MB`
        }
      </div>
    );
  }, [optimization]);

  return (
    <Card>
      {tableHeader}
      
      {listOptimization.shouldVirtualize ? (
        <VirtualTable
          dataSource={listOptimization.items}
          columns={optimizedColumns}
          height={height}
          rowKey={rowKey}
          loading={loading}
        />
      ) : (
        <Table
          dataSource={listOptimization.items}
          columns={optimizedColumns}
          rowKey={rowKey}
          loading={loading}
          pagination={{
            current: listOptimization.currentPage,
            total: dataSource.length,
            pageSize,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => 
              `第 ${range[0]}-${range[1]} 条，共 ${total} 条`,
            onChange: (page) => listOptimization.setCurrentPage(page)
          }}
          scroll={{ y: height }}
        />
      )}
      
      {performanceInfo}
    </Card>
  );
};

// Example usage component
export const OptimizedDataTableExample: React.FC = () => {
  // Mock data for demonstration
  const mockData = useMemo(() => 
    Array.from({ length: 1000 }, (_, index) => ({
      key: index,
      id: index + 1,
      name: `Item ${index + 1}`,
      value: Math.random() * 1000,
      status: Math.random() > 0.5 ? 'active' : 'inactive',
      timestamp: new Date(Date.now() - Math.random() * 86400000).toISOString()
    })), []
  );

  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 80,
      sorter: (a: any, b: any) => a.id - b.id
    },
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
      width: 150
    },
    {
      title: '数值',
      dataIndex: 'value',
      key: 'value',
      width: 120,
      render: (value: number) => value.toFixed(2),
      sorter: (a: any, b: any) => a.value - b.value
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => (
        <span style={{ 
          color: status === 'active' ? '#52c41a' : '#ff4d4f' 
        }}>
          {status === 'active' ? '活跃' : '非活跃'}
        </span>
      )
    },
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 180,
      render: (timestamp: string) => new Date(timestamp).toLocaleString()
    }
  ];

  const handleRefresh = useCallback(() => {
    console.log('Refreshing data...');
  }, []);

  return (
    <OptimizedDataTable
      dataSource={mockData}
      columns={columns}
      title="优化数据表格示例"
      searchable={true}
      searchFields={['name', 'status']}
      enableVirtualization={true}
      virtualizationThreshold={100}
      pageSize={50}
      height={500}
      onRefresh={handleRefresh}
    />
  );
};