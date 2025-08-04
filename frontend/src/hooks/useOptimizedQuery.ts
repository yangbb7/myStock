import { useQuery, UseQueryOptions, UseQueryResult } from '@tanstack/react-query';
import { useCallback, useMemo, useRef } from 'react';
import { apiCache, generateApiCacheKey } from '../utils/cacheManager';
import { debounce } from 'lodash';

interface OptimizedQueryOptions<T> extends Omit<UseQueryOptions<T>, 'queryKey' | 'queryFn'> {
  endpoint: string;
  params?: Record<string, any>;
  enableCache?: boolean;
  cacheTTL?: number;
  debounceDelay?: number;
  enableDebounce?: boolean;
}

// Optimized query hook with caching and debouncing
export const useOptimizedQuery = <T = any>(
  options: OptimizedQueryOptions<T>,
  queryFn: () => Promise<T>
): UseQueryResult<T> => {
  const {
    endpoint,
    params,
    enableCache = true,
    cacheTTL = 5 * 60 * 1000, // 5 minutes
    debounceDelay = 300,
    enableDebounce = false,
    ...queryOptions
  } = options;

  const debouncedQueryFn = useRef(
    debounce(queryFn, debounceDelay)
  );

  // Generate cache key
  const cacheKey = useMemo(() => 
    generateApiCacheKey(endpoint, params), 
    [endpoint, params]
  );

  // Query key for React Query
  const queryKey = useMemo(() => 
    [endpoint, params], 
    [endpoint, params]
  );

  // Enhanced query function with caching
  const enhancedQueryFn = useCallback(async (): Promise<T> => {
    // Check cache first
    if (enableCache) {
      const cached = apiCache.get(cacheKey);
      if (cached !== null) {
        return cached;
      }
    }

    // Execute query function
    const actualQueryFn = enableDebounce ? debouncedQueryFn.current : queryFn;
    const result = await actualQueryFn();

    // Cache the result
    if (enableCache) {
      apiCache.set(cacheKey, result, cacheTTL);
    }

    return result;
  }, [queryFn, enableCache, cacheKey, cacheTTL, enableDebounce]);

  return useQuery({
    queryKey,
    queryFn: enhancedQueryFn,
    ...queryOptions
  });
};

// Hook for optimized real-time queries
export const useOptimizedRealTimeQuery = <T = any>(
  options: OptimizedQueryOptions<T> & {
    refetchInterval?: number;
    maxRefetchRate?: number;
  },
  queryFn: () => Promise<T>
): UseQueryResult<T> => {
  const {
    refetchInterval = 1000,
    maxRefetchRate = 100, // minimum 100ms between refetches
    ...restOptions
  } = options;

  const lastRefetchTime = useRef(0);

  // Throttled refetch interval
  const throttledRefetchInterval = useMemo(() => {
    return Math.max(refetchInterval, maxRefetchRate);
  }, [refetchInterval, maxRefetchRate]);

  // Enhanced query function with rate limiting
  const rateLimitedQueryFn = useCallback(async (): Promise<T> => {
    const now = Date.now();
    const timeSinceLastRefetch = now - lastRefetchTime.current;

    if (timeSinceLastRefetch < maxRefetchRate) {
      // If we're refetching too fast, return cached data if available
      const cached = apiCache.get(generateApiCacheKey(restOptions.endpoint, restOptions.params));
      if (cached !== null) {
        return cached;
      }
    }

    lastRefetchTime.current = now;
    return queryFn();
  }, [queryFn, maxRefetchRate, restOptions.endpoint, restOptions.params]);

  return useOptimizedQuery(
    {
      ...restOptions,
      refetchInterval: throttledRefetchInterval,
      refetchIntervalInBackground: false,
      refetchOnWindowFocus: false,
    },
    rateLimitedQueryFn
  );
};

// Hook for batch queries optimization
export const useBatchOptimizedQueries = <T = any>(
  queries: Array<{
    endpoint: string;
    params?: Record<string, any>;
    queryFn: () => Promise<T>;
    options?: OptimizedQueryOptions<T>;
  }>
) => {
  // Group queries by endpoint for potential batching
  const groupedQueries = useMemo(() => {
    const groups: Record<string, typeof queries> = {};
    queries.forEach(query => {
      if (!groups[query.endpoint]) {
        groups[query.endpoint] = [];
      }
      groups[query.endpoint].push(query);
    });
    return groups;
  }, [queries]);

  // Execute queries with staggered timing to avoid overwhelming the server
  const results = queries.map((query, index) => {
    const staggerDelay = Math.floor(index / 5) * 50; // 50ms delay every 5 queries
    
    return useOptimizedQuery(
      {
        endpoint: query.endpoint,
        params: query.params,
        ...query.options,
        refetchInterval: query.options?.refetchInterval 
          ? query.options.refetchInterval + staggerDelay 
          : undefined,
      },
      query.queryFn
    );
  });

  return results;
};

// Hook for memory-efficient infinite queries
export const useOptimizedInfiniteQuery = <T = any>(
  options: OptimizedQueryOptions<T> & {
    pageSize?: number;
    maxPages?: number;
  },
  queryFn: (page: number, pageSize: number) => Promise<{ data: T[]; hasMore: boolean }>
) => {
  const {
    pageSize = 50,
    maxPages = 10,
    ...restOptions
  } = options;

  const pages = useRef<Array<{ data: T[]; hasMore: boolean }>>([]);
  const currentPage = useRef(1);

  const loadPage = useCallback(async (page: number): Promise<{ data: T[]; hasMore: boolean }> => {
    // Check if we already have this page cached
    if (pages.current[page - 1]) {
      return pages.current[page - 1];
    }

    const result = await queryFn(page, pageSize);
    
    // Store the page, but limit total pages to prevent memory issues
    if (pages.current.length >= maxPages) {
      pages.current.shift(); // Remove oldest page
    }
    
    pages.current[page - 1] = result;
    return result;
  }, [queryFn, pageSize, maxPages]);

  const loadMore = useCallback(async () => {
    const nextPage = currentPage.current + 1;
    const result = await loadPage(nextPage);
    currentPage.current = nextPage;
    return result;
  }, [loadPage]);

  const reset = useCallback(() => {
    pages.current = [];
    currentPage.current = 1;
  }, []);

  // Get all loaded data
  const allData = useMemo(() => {
    return pages.current.flatMap(page => page.data);
  }, [pages.current]);

  return {
    data: allData,
    loadMore,
    reset,
    hasMore: pages.current[pages.current.length - 1]?.hasMore ?? true,
    currentPage: currentPage.current,
    totalPages: pages.current.length
  };
};