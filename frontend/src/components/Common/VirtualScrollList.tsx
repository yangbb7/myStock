import React, { useMemo, useCallback, useState, useRef, useEffect } from 'react';
import { FixedSizeList as List, VariableSizeList, ListChildComponentProps } from 'react-window';
import { debounce, throttle } from 'lodash';
import { Spin, Empty } from 'antd';

interface VirtualScrollListProps<T = any> {
  items: T[];
  height: number;
  itemHeight?: number | ((index: number) => number);
  width?: number | string;
  overscan?: number;
  threshold?: number;
  renderItem: (item: T, index: number, style: React.CSSProperties) => React.ReactNode;
  onScroll?: (scrollTop: number, scrollLeft: number) => void;
  onItemsRendered?: (startIndex: number, endIndex: number) => void;
  loading?: boolean;
  empty?: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
  enableInfiniteScroll?: boolean;
  onLoadMore?: () => void;
  hasNextPage?: boolean;
  loadingMore?: boolean;
}

// Memoized list item component
const ListItem = React.memo<ListChildComponentProps & { 
  renderItem: (item: any, index: number, style: React.CSSProperties) => React.ReactNode;
  items: any[];
}>(({ index, style, renderItem, items }) => {
  const item = items[index];
  return (
    <div style={style}>
      {renderItem(item, index, style)}
    </div>
  );
});

ListItem.displayName = 'VirtualScrollListItem';

export const VirtualScrollList = <T extends any>({
  items,
  height,
  itemHeight = 50,
  width = '100%',
  overscan = 5,
  threshold = 100,
  renderItem,
  onScroll,
  onItemsRendered,
  loading = false,
  empty,
  className,
  style,
  enableInfiniteScroll = false,
  onLoadMore,
  hasNextPage = false,
  loadingMore = false,
}: VirtualScrollListProps<T>) => {
  const listRef = useRef<any>(null);
  const [isScrolling, setIsScrolling] = useState(false);
  const scrollTimeoutRef = useRef<NodeJS.Timeout>();

  // Determine if we should use virtualization
  const shouldVirtualize = useMemo(() => {
    return items.length > threshold;
  }, [items.length, threshold]);

  // Throttled scroll handler
  const throttledScrollHandler = useMemo(
    () => throttle(({ scrollTop, scrollLeft }: { scrollTop: number; scrollLeft: number }) => {
      onScroll?.(scrollTop, scrollLeft);
    }, 16),
    [onScroll]
  );

  // Debounced scroll end handler
  const debouncedScrollEndHandler = useMemo(
    () => debounce(() => {
      setIsScrolling(false);
    }, 150),
    []
  );

  // Handle scroll events
  const handleScroll = useCallback((scrollProps: { scrollTop: number; scrollLeft: number }) => {
    setIsScrolling(true);
    throttledScrollHandler(scrollProps);
    debouncedScrollEndHandler();

    // Infinite scroll logic
    if (enableInfiniteScroll && hasNextPage && !loadingMore && onLoadMore) {
      const { scrollTop } = scrollProps;
      const totalHeight = typeof itemHeight === 'function' 
        ? items.reduce((sum, _, index) => sum + (itemHeight as Function)(index), 0)
        : items.length * (itemHeight as number);
      
      // Trigger load more when 80% scrolled
      if (scrollTop > totalHeight * 0.8 - height) {
        onLoadMore();
      }
    }
  }, [throttledScrollHandler, debouncedScrollEndHandler, enableInfiniteScroll, hasNextPage, loadingMore, onLoadMore, items, itemHeight, height]);

  // Handle items rendered
  const handleItemsRendered = useCallback(({ visibleStartIndex, visibleStopIndex }: {
    visibleStartIndex: number;
    visibleStopIndex: number;
  }) => {
    onItemsRendered?.(visibleStartIndex, visibleStopIndex);
  }, [onItemsRendered]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      throttledScrollHandler.cancel();
      debouncedScrollEndHandler.cancel();
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
    };
  }, [throttledScrollHandler, debouncedScrollEndHandler]);

  // Memoized item data
  const itemData = useMemo(() => ({
    items,
    renderItem,
  }), [items, renderItem]);

  // Loading state
  if (loading) {
    return (
      <div 
        style={{ 
          height, 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center',
          ...style 
        }}
        className={className}
      >
        <Spin size="large" />
      </div>
    );
  }

  // Empty state
  if (items.length === 0) {
    return (
      <div 
        style={{ 
          height, 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center',
          ...style 
        }}
        className={className}
      >
        {empty || <Empty description="暂无数据" />}
      </div>
    );
  }

  // Non-virtualized list for small datasets
  if (!shouldVirtualize) {
    return (
      <div 
        style={{ 
          height, 
          width,
          overflow: 'auto',
          ...style 
        }}
        className={className}
      >
        {items.map((item, index) => (
          <div key={index}>
            {renderItem(item, index, {})}
          </div>
        ))}
        {enableInfiniteScroll && loadingMore && (
          <div style={{ padding: '16px', textAlign: 'center' }}>
            <Spin />
          </div>
        )}
      </div>
    );
  }

  // Variable height list
  if (typeof itemHeight === 'function') {
    return (
      <VariableSizeList
        ref={listRef}
        height={height}
        width={width}
        itemCount={items.length + (enableInfiniteScroll && loadingMore ? 1 : 0)}
        itemSize={(index) => {
          if (enableInfiniteScroll && loadingMore && index === items.length) {
            return 60; // Loading indicator height
          }
          return (itemHeight as Function)(index);
        }}
        overscanCount={overscan}
        onScroll={handleScroll}
        onItemsRendered={handleItemsRendered}
        itemData={itemData}
        style={style}
        className={className}
      >
        {({ index, style, data }) => {
          if (enableInfiniteScroll && loadingMore && index === items.length) {
            return (
              <div style={{ ...style, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <Spin />
              </div>
            );
          }
          return <ListItem index={index} style={style} {...data} />;
        }}
      </VariableSizeList>
    );
  }

  // Fixed height list
  return (
    <List
      ref={listRef}
      height={height}
      width={width}
      itemCount={items.length + (enableInfiniteScroll && loadingMore ? 1 : 0)}
      itemSize={itemHeight as number}
      overscanCount={overscan}
      onScroll={handleScroll}
      onItemsRendered={handleItemsRendered}
      itemData={itemData}
      style={style}
      className={className}
    >
      {({ index, style, data }) => {
        if (enableInfiniteScroll && loadingMore && index === items.length) {
          return (
            <div style={{ ...style, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <Spin />
            </div>
          );
        }
        return <ListItem index={index} style={style} {...data} />;
      }}
    </List>
  );
};

// Hook for virtual scroll optimization
export const useVirtualScrollOptimization = <T extends any>(
  items: T[],
  options: {
    pageSize?: number;
    threshold?: number;
    preloadPages?: number;
  } = {}
) => {
  const {
    pageSize = 50,
    threshold = 100,
    preloadPages = 2
  } = options;

  const [currentPage, setCurrentPage] = useState(1);
  const [loadedPages, setLoadedPages] = useState(new Set([1]));

  // Calculate visible items based on loaded pages
  const visibleItems = useMemo(() => {
    const maxPage = Math.max(...Array.from(loadedPages));
    return items.slice(0, maxPage * pageSize);
  }, [items, loadedPages, pageSize]);

  // Load more pages when needed
  const loadMorePages = useCallback((startIndex: number, endIndex: number) => {
    const startPage = Math.floor(startIndex / pageSize) + 1;
    const endPage = Math.floor(endIndex / pageSize) + 1;
    
    const newPages = new Set(loadedPages);
    let hasNewPages = false;

    // Load current visible pages plus preload pages
    for (let page = startPage; page <= endPage + preloadPages; page++) {
      if (!newPages.has(page) && (page - 1) * pageSize < items.length) {
        newPages.add(page);
        hasNewPages = true;
      }
    }

    if (hasNewPages) {
      setLoadedPages(newPages);
    }
  }, [loadedPages, pageSize, preloadPages, items.length]);

  // Reset when items change
  useEffect(() => {
    setCurrentPage(1);
    setLoadedPages(new Set([1]));
  }, [items.length]);

  return {
    visibleItems,
    shouldVirtualize: items.length > threshold,
    onItemsRendered: loadMorePages,
    totalItems: items.length,
    loadedItems: visibleItems.length
  };
};