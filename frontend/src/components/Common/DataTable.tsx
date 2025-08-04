import React, { useState } from 'react';
import { Table, Input, Button, Space, Tooltip } from 'antd';
import { ReloadOutlined, DownloadOutlined } from '@ant-design/icons';
import type { TableProps } from 'antd';
import type { ColumnsType, TablePaginationConfig } from 'antd/es/table';
import type { FilterValue, SorterResult } from 'antd/es/table/interface';

interface DataTableProps<T> extends Omit<TableProps<T>, 'columns'> {
  columns: ColumnsType<T>;
  data?: T[];
  loading?: boolean;
  searchable?: boolean;
  searchPlaceholder?: string;
  refreshable?: boolean;
  exportable?: boolean;
  onRefresh?: () => void;
  onExport?: () => void;
  onSearch?: (value: string) => void;
  pagination?: TablePaginationConfig | false;
  showTotal?: boolean;
  totalText?: string;
  className?: string;
}

export function DataTable<T extends Record<string, any>>({
  columns,
  data = [],
  loading = false,
  searchable = false,
  searchPlaceholder = '搜索...',
  refreshable = false,
  exportable = false,
  onRefresh,
  onExport,
  onSearch,
  pagination = {
    showSizeChanger: true,
    showQuickJumper: true,
    showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条，共 ${total} 条`,
    pageSizeOptions: ['10', '20', '50', '100'],
    defaultPageSize: 20,
  },
  showTotal = true,
  totalText = '总计',
  className,
  ...tableProps
}: DataTableProps<T>) {
  const [, setSearchValue] = useState('');
  const [filteredData, setFilteredData] = useState<T[]>(data);

  React.useEffect(() => {
    setFilteredData(data);
  }, [data]);

  const handleSearch = (value: string) => {
    setSearchValue(value);
    if (onSearch) {
      onSearch(value);
    } else {
      // Default local search
      const filtered = data.filter((item) =>
        Object.values(item).some((val) =>
          String(val).toLowerCase().includes(value.toLowerCase())
        )
      );
      setFilteredData(filtered);
    }
  };

  const handleTableChange = (
    pagination: TablePaginationConfig,
    filters: Record<string, FilterValue | null>,
    sorter: SorterResult<T> | SorterResult<T>[]
  ) => {
    if (tableProps.onChange) {
      tableProps.onChange(pagination, filters, sorter, { currentDataSource: filteredData, action: 'paginate' });
    }
  };

  const renderToolbar = () => {
    if (!searchable && !refreshable && !exportable) return null;

    return (
      <div style={{ marginBottom: 16, display: 'flex', justifyContent: 'space-between' }}>
        <div>
          {searchable && (
            <Input.Search
              placeholder={searchPlaceholder}
              allowClear
              style={{ width: 300 }}
              onSearch={handleSearch}
              onChange={(e) => {
                if (!e.target.value) {
                  handleSearch('');
                }
              }}
            />
          )}
        </div>
        <Space>
          {refreshable && (
            <Tooltip title="刷新">
              <Button
                icon={<ReloadOutlined />}
                onClick={onRefresh}
                loading={loading}
              />
            </Tooltip>
          )}
          {exportable && (
            <Tooltip title="导出">
              <Button
                icon={<DownloadOutlined />}
                onClick={onExport}
              />
            </Tooltip>
          )}
        </Space>
      </div>
    );
  };

  const enhancedPagination = pagination ? {
    ...pagination,
    showTotal: showTotal ? (total: number, range: [number, number]) => 
      `第 ${range[0]}-${range[1]} 条，共 ${total} 条` : undefined,
  } : false;

  return (
    <div className={className}>
      {renderToolbar()}
      <Table<T>
        columns={columns}
        dataSource={filteredData}
        loading={loading}
        pagination={enhancedPagination}
        onChange={handleTableChange}
        scroll={{ x: 'max-content' }}
        size="middle"
        {...tableProps}
      />
    </div>
  );
}

export default DataTable;