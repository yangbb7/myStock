import * as echarts from 'echarts';

// Light theme configuration
export const lightTheme = {
  color: [
    '#1890ff',
    '#52c41a',
    '#faad14',
    '#f5222d',
    '#722ed1',
    '#fa8c16',
    '#13c2c2',
    '#eb2f96',
    '#a0d911',
    '#fa541c',
  ],
  backgroundColor: '#ffffff',
  textStyle: {
    color: '#333333',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  },
  title: {
    textStyle: {
      color: '#333333',
      fontSize: 16,
      fontWeight: 'normal',
    },
  },
  legend: {
    textStyle: {
      color: '#666666',
      fontSize: 12,
    },
  },
  tooltip: {
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    borderColor: 'transparent',
    textStyle: {
      color: '#ffffff',
      fontSize: 12,
    },
  },
  grid: {
    borderColor: '#f0f0f0',
  },
  categoryAxis: {
    axisLine: {
      lineStyle: {
        color: '#d9d9d9',
      },
    },
    axisTick: {
      lineStyle: {
        color: '#d9d9d9',
      },
    },
    axisLabel: {
      color: '#666666',
    },
    splitLine: {
      lineStyle: {
        color: '#f0f0f0',
      },
    },
  },
  valueAxis: {
    axisLine: {
      lineStyle: {
        color: '#d9d9d9',
      },
    },
    axisTick: {
      lineStyle: {
        color: '#d9d9d9',
      },
    },
    axisLabel: {
      color: '#666666',
    },
    splitLine: {
      lineStyle: {
        color: '#f0f0f0',
      },
    },
  },
};

// Dark theme configuration
export const darkTheme = {
  color: [
    '#4dabf7',
    '#69db7c',
    '#ffd43b',
    '#ff6b6b',
    '#9775fa',
    '#ff922b',
    '#22b8cf',
    '#f06595',
    '#94d82d',
    '#fd7e14',
  ],
  backgroundColor: '#141414',
  textStyle: {
    color: '#ffffff',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  },
  title: {
    textStyle: {
      color: '#ffffff',
      fontSize: 16,
      fontWeight: 'normal',
    },
  },
  legend: {
    textStyle: {
      color: '#d9d9d9',
      fontSize: 12,
    },
  },
  tooltip: {
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    borderColor: '#434343',
    textStyle: {
      color: '#ffffff',
      fontSize: 12,
    },
  },
  grid: {
    borderColor: '#434343',
  },
  categoryAxis: {
    axisLine: {
      lineStyle: {
        color: '#434343',
      },
    },
    axisTick: {
      lineStyle: {
        color: '#434343',
      },
    },
    axisLabel: {
      color: '#d9d9d9',
    },
    splitLine: {
      lineStyle: {
        color: '#434343',
      },
    },
  },
  valueAxis: {
    axisLine: {
      lineStyle: {
        color: '#434343',
      },
    },
    axisTick: {
      lineStyle: {
        color: '#434343',
      },
    },
    axisLabel: {
      color: '#d9d9d9',
    },
    splitLine: {
      lineStyle: {
        color: '#434343',
      },
    },
  },
};

// Trading-specific theme
export const tradingTheme = {
  ...lightTheme,
  color: [
    '#ef232a', // Red for bearish
    '#14b143', // Green for bullish
    '#1890ff', // Blue for neutral
    '#faad14', // Yellow for warning
    '#722ed1', // Purple for special
    '#fa8c16', // Orange for highlight
    '#13c2c2', // Cyan for info
    '#eb2f96', // Pink for accent
  ],
  candlestick: {
    itemStyle: {
      color: '#ef232a',
      color0: '#14b143',
      borderColor: '#ef232a',
      borderColor0: '#14b143',
    },
  },
};

// Register themes with ECharts
export const registerThemes = () => {
  echarts.registerTheme('light', lightTheme);
  echarts.registerTheme('dark', darkTheme);
  echarts.registerTheme('trading', tradingTheme);
};

// Theme configuration manager
export class ChartThemeManager {
  private static instance: ChartThemeManager;
  private currentTheme: string = 'light';
  private listeners: Array<(theme: string) => void> = [];

  static getInstance(): ChartThemeManager {
    if (!ChartThemeManager.instance) {
      ChartThemeManager.instance = new ChartThemeManager();
    }
    return ChartThemeManager.instance;
  }

  setTheme(theme: string) {
    this.currentTheme = theme;
    this.notifyListeners();
  }

  getTheme(): string {
    return this.currentTheme;
  }

  subscribe(listener: (theme: string) => void) {
    this.listeners.push(listener);
    return () => {
      const index = this.listeners.indexOf(listener);
      if (index > -1) {
        this.listeners.splice(index, 1);
      }
    };
  }

  private notifyListeners() {
    this.listeners.forEach(listener => listener(this.currentTheme));
  }
}

// Common chart configurations
export const commonChartConfig = {
  animation: true,
  animationDuration: 300,
  animationEasing: 'cubicOut',
  textStyle: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  },
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    borderColor: 'transparent',
    textStyle: {
      color: '#ffffff',
      fontSize: 12,
    },
    axisPointer: {
      type: 'cross',
      crossStyle: {
        color: '#999',
      },
    },
  },
  legend: {
    type: 'scroll',
    orient: 'horizontal',
    top: 10,
    textStyle: {
      fontSize: 12,
    },
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: '3%',
    top: '15%',
    containLabel: true,
  },
  dataZoom: [
    {
      type: 'inside',
      start: 80,
      end: 100,
    },
    {
      type: 'slider',
      start: 80,
      end: 100,
      height: 20,
    },
  ],
};

// Utility functions for chart formatting
export const formatters = {
  currency: (value: number, currency = '¥') => `${currency}${value.toFixed(2)}`,
  percentage: (value: number) => `${(value * 100).toFixed(2)}%`,
  number: (value: number, decimals = 2) => value.toFixed(decimals),
  volume: (value: number) => {
    if (value >= 1e8) return `${(value / 1e8).toFixed(1)}亿`;
    if (value >= 1e4) return `${(value / 1e4).toFixed(1)}万`;
    return value.toLocaleString();
  },
  time: (timestamp: string | number | Date, format = 'HH:mm:ss') => {
    const dayjs = require('dayjs');
    return dayjs(timestamp).format(format);
  },
  date: (timestamp: string | number | Date, format = 'YYYY-MM-DD') => {
    const dayjs = require('dayjs');
    return dayjs(timestamp).format(format);
  },
};

export default {
  lightTheme,
  darkTheme,
  tradingTheme,
  registerThemes,
  ChartThemeManager,
  commonChartConfig,
  formatters,
};