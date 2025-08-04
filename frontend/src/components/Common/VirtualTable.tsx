import React, { useMemo, useCallback, useState, useEffect } from 'react';
import { Table, TableProps } from 'antd';
import { FixedSizeList as List } from 'react-window';
import { debounce } from 'lodash';

interface VirtualTableProps<T = any> extends Omit<TableProps<T>, 'pagination'> {
  height?: number;
  itemHeight?: number;
  overscan?: number;
  enableVirtualization?: boolean;
  onScroll?: (scrollTop: number) => void;
}

// Virtual table row component
const VirtualTableRow: React.FC<{
  index: number;
  style: React.CSSProperties;
  data: {
    columns: any[];
    dataSource: any[];
    rowKey: string | ((record: any) => string);
  };
}> = ({ index, style, data }) => {
  const { columns, dataSource, rowKey } = data;
  const record = dataSource[index];
  const key = typeof rowKey === 'function' ? rowKey(record) : record[rowKey];

  return (
    <div style={style} key={key}>
      <div style={{ display: 'flex', alignItems: 'center', padding: '8px 16px' }}>
        {columns.map((column: any, colIndex: number) => (
          <div
            key={column.key || colIndex}
            style={{
              flex: column.width ? `0 0 ${column.width}px` : 1,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              marginRight: '16px'
            }}
          >
            {column.render 
              ? column.render(record[column.dataIndex], record, index)
              : record[column.dataIndex]
            }
          </div>
        ))}
      </div>
    </div>
  );
};

export const VirtualTable: React.FC<VirtualTableProps> = ({
  dataSource = [],
  columns = [],
  height = 400,
  itemHeight = 54,
  overscan = 5,
  enableVirtualization = true,
  rowKey = 'key',
  onScroll,
  ...tableProps
}) => {
  const [scrollTop, setScrollTop] = useState(0);

  // Debounced scroll handler
  const debouncedScrollHandler = useMemo(
    () => debounce((scrollTop: number) => {
      setScrollTop(scrollTop);
      onScroll?.(scrollTop);
    }, 16),
    [onScroll]
  );

  // Handle scroll
  const handleScroll = useCallback(({ scrollTop }: { scrollTop: number }) => {
    debouncedScrollHandler(scrollTop);
  }, [debouncedScrollHandler]);

  // Cleanup debounced function
  useEffect(() => {
    return () => {
      debouncedScrollHandler.cancel();
    };
  }, [debouncedScrollHandler]);

  // If virtualization is disabled or dataset is small, use regular table
  if (!enableVirtualization || dataSource.length < 100) {
    return (
      <Table
        {...tableProps}
        dataSource={dataSource}
        columns={columns}
        rowKey={rowKey}
        scroll={{ y: height }}
        pagination={false}
      />
    );
  }

  // Virtual table implementation
  return (
    <div style={{ height, border: '1px solid #f0f0f0' }}>
      {/* Table header */}
      <div style={{ 
        display: 'flex', 
        background: '#fafafa', 
        borderBottom: '1px solid #f0f0f0',
        padding: '8px 16px',
        fontWeight: 'bold'
      }}>
        {columns.map((column: any, index: number) => (
          <div
            key={column.key || index}
            style={{
              flex: column.width ? `0 0 ${column.width}px` : 1,
              marginRight: '16px'
            }}
          >
            {column.title}
          </div>
        ))}
      </div>

      {/* Virtual list */}
      <List
        height={height - 40} // Subtract header height
        itemCount={dataSource.length}
        itemSize={itemHeight}
        overscanCount={overscan}
        onScroll={handleScroll}
        itemData={{
          columns,
          dataSource,
          rowKey
        }}
      >
        {VirtualTableRow}
      </List>
    </div>
  );
};