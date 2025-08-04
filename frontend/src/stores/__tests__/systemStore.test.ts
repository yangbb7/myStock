import { describe, it, expect, beforeEach } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useSystemStore } from '../systemStore';

describe('systemStore', () => {
  beforeEach(() => {
    // Reset store state before each test
    const { result } = renderHook(() => useSystemStore());
    act(() => {
      result.current.setTheme('light');
      // Reset to initial state - systemStatus starts as null
    });
  });

  it('should initialize with default state', () => {
    const { result } = renderHook(() => useSystemStore());

    expect(result.current.theme).toBe('light');
    expect(result.current.systemStatus).toBeNull();
  });

  it('should set theme', () => {
    const { result } = renderHook(() => useSystemStore());

    act(() => {
      result.current.setTheme('dark');
    });

    expect(result.current.theme).toBe('dark');
  });

  it('should set system status', () => {
    const { result } = renderHook(() => useSystemStore());

    const mockStatus = {
      isRunning: true,
      uptime: 3600,
      modules: {
        data: { initialized: true },
        strategy: { initialized: true },
      },
    };

    act(() => {
      result.current.setSystemStatus(mockStatus);
    });

    expect(result.current.systemStatus).toEqual(mockStatus);
  });

  it('should persist theme setting', () => {
    const { result: result1 } = renderHook(() => useSystemStore());

    act(() => {
      result1.current.setTheme('dark');
    });

    // Create a new hook instance to simulate page reload
    const { result: result2 } = renderHook(() => useSystemStore());
    
    expect(result2.current.theme).toBe('dark');
  });
});