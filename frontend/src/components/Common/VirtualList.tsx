import React, { useMemo, useCallback, useState, useRef, useEffect } from 'react';
import { FixedSizeList as List, VariableSizeList } from 'react-window';
import { debounce } from 'lodash';
import { Spin } from 'antd';

interface VirtualListProps<T = any> {
  items: T[];
  height: number;
  itemHeight: number | ((index: number) => number);
  renderItem: (item: T, index: number) => React.ReactNode;
  overscan?: number;
  onScroll?: (scrollTop: number) => void;
  onEndReached?: () => void;
  endReachedThreshold?: number;
  loading?: boolean;
  loadingComponent?: React.ReactNode;
  emptyComponent?: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

// Virtual list item wrapper
const VirtualListItem: React.FC<{
  index: number;
  style: React.CSSProperties;
  data: {
    items: any[];
    renderItem: (item: any, index: number) => React.ReactNode;
  };
}> = ({ index, style, data }) => {
  const { items, renderItem } = data;
  const item = items[index];

  return (
    <div style={style}>
      {renderItem(item, index)}
    </div>
  );
};

export const VirtualList: React.FC<VirtualListProps> = ({
  items,
  height,
  itemHeight,
  renderItem,
  overscan = 5,
  onScroll,
  onEndReached,
  endReachedThreshold = 0.8,
  loading = false,
  loadingComponent,
  emptyComponent,
  className,
  style
}) => {
  const [scrollTop, setScrollTop] = useState(0);
  const listRef = useRef<any>(null);
  const isVariableHeight = typeof itemHeight === 'function';

  // Debounced scroll handler
  const debouncedScrollHandler = useMemo(
    () => debounce((scrollTop: number, scrollHeight: number, clientHeight: number) => {
      setScrollTop(scrollTop);
      onScroll?.(scrollTop);

      // Check if we've reached the end
      if (onEndReached && scrollTop + clientHeight >= scrollHeight * endReachedThreshold) {
        onEndReached();
      }
    }, 16),
    [onScroll, onEndReached, endReachedThreshold]
  );

  // Handle scroll
  const handleScroll = useCallback(({ scrollTop, scrollHeight, clientHeight }: any) => {
    debouncedScrollHandler(scrollTop, scrollHeight, clientHeight);
  }, [debouncedScrollHandler]);

  // Cleanup debounced function
  useEffect(() => {
    return () => {
      debouncedScrollHandler.cancel();
    };
  }, [debouncedScrollHandler]);

  // Scroll to item
  const scrollToItem = useCallback((index: number, align?: 'start' | 'center' | 'end' | 'smart') => {
    if (listRef.current) {
      listRef.current.scrollToItem(index, align);
    }
  }, []);

  // Empty state
  if (items.length === 0 && !loading) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', ...style }}>
        {emptyComponent || <div>暂无数据</div>}
      </div>
    );
  }

  const itemData = {
    items,
    renderItem
  };

  return (
    <div className={className} style={{ height, ...style }}>
      {isVariableHeight ? (
        <VariableSizeList
          ref={listRef}
          height={height}
          itemCount={items.length}
          itemSize={itemHeight as (index: number) => number}
          overscanCount={overscan}
          onScroll={handleScroll}
          itemData={itemData}
        >
          {VirtualListItem}
        </VariableSizeList>
      ) : (
        <List
          ref={listRef}
          height={height}
          itemCount={items.length}
          itemSize={itemHeight as number}
          overscanCount={overscan}
          onScroll={handleScroll}
          itemData={itemData}
        >
          {VirtualListItem}
        </List>
      )}
      
      {loading && (
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          padding: '16px' 
        }}>
          {loadingComponent || <Spin />}
        </div>
      )}
    </div>
  );
};

// Hook for virtual list with infinite loading
export const useVirtualList = <T,>(
  fetchData: (page: number, pageSize: number) => Promise<{ data: T[]; hasMore: boolean }>,
  pageSize: number = 50
) => {
  const [items, setItems] = useState<T[]>([]);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [page, setPage] = useState(1);

  const loadMore = useCallback(async () => {
    if (loading || !hasMore) return;

    setLoading(true);
    try {
      const result = await fetchData(page, pageSize);
      setItems(prev => [...prev, ...result.data]);
      setHasMore(result.hasMore);
      setPage(prev => prev + 1);
    } catch (error) {
      console.error('Failed to load more data:', error);
    } finally {
      setLoading(false);
    }
  }, [fetchData, page, pageSize, loading, hasMore]);

  const reset = useCallback(() => {
    setItems([]);
    setPage(1);
    setHasMore(true);
    setLoading(false);
  }, []);

  // Load initial data
  useEffect(() => {
    if (items.length === 0 && !loading) {
      loadMore();
    }
  }, []);

  return {
    items,
    loading,
    hasMore,
    loadMore,
    reset
  };
};