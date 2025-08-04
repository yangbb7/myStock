import React, { createContext, useContext, useEffect } from 'react';
import { ConfigProvider, theme } from 'antd';
import { useSystemStore } from '../../stores/systemStore';
import { registerThemes, ChartThemeManager } from '../Charts/ChartTheme';

interface ThemeContextValue {
  theme: 'light' | 'dark';
  setTheme: (theme: 'light' | 'dark') => void;
}

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

interface ThemeProviderProps {
  children: React.ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const { theme: currentTheme, setTheme } = useSystemStore();
  const chartThemeManager = ChartThemeManager.getInstance();

  // Register chart themes on mount
  useEffect(() => {
    registerThemes();
  }, []);

  // Sync chart theme with app theme
  useEffect(() => {
    chartThemeManager.setTheme(currentTheme === 'dark' ? 'dark' : 'light');
  }, [currentTheme, chartThemeManager]);

  // Apply theme to document body
  useEffect(() => {
    document.body.className = currentTheme === 'dark' ? 'dark-theme' : 'light-theme';
    
    // Update CSS custom properties for theme
    const root = document.documentElement;
    if (currentTheme === 'dark') {
      root.style.setProperty('--bg-color', '#141414');
      root.style.setProperty('--text-color', '#ffffff');
      root.style.setProperty('--border-color', '#434343');
      root.style.setProperty('--card-bg', '#1f1f1f');
    } else {
      root.style.setProperty('--bg-color', '#ffffff');
      root.style.setProperty('--text-color', '#000000');
      root.style.setProperty('--border-color', '#d9d9d9');
      root.style.setProperty('--card-bg', '#ffffff');
    }
  }, [currentTheme]);

  const antdTheme = {
    algorithm: currentTheme === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm,
    token: {
      colorPrimary: '#1890ff',
      borderRadius: 6,
      ...(currentTheme === 'dark' ? {
        colorBgContainer: '#1f1f1f',
        colorBgElevated: '#262626',
        colorBgLayout: '#141414',
        colorText: '#ffffff',
        colorTextSecondary: '#d9d9d9',
        colorBorder: '#434343',
      } : {
        colorBgContainer: '#ffffff',
        colorBgElevated: '#ffffff',
        colorBgLayout: '#f0f2f5',
        colorText: '#000000',
        colorTextSecondary: '#666666',
        colorBorder: '#d9d9d9',
      }),
    },
    components: {
      Layout: {
        headerBg: currentTheme === 'dark' ? '#1f1f1f' : '#ffffff',
        siderBg: currentTheme === 'dark' ? '#1f1f1f' : '#ffffff',
        bodyBg: currentTheme === 'dark' ? '#141414' : '#f0f2f5',
      },
      Menu: {
        itemBg: 'transparent',
        itemSelectedBg: currentTheme === 'dark' ? '#1890ff20' : '#e6f7ff',
        itemHoverBg: currentTheme === 'dark' ? '#ffffff10' : '#f5f5f5',
      },
      Card: {
        headerBg: 'transparent',
      },
      Table: {
        headerBg: currentTheme === 'dark' ? '#262626' : '#fafafa',
        rowHoverBg: currentTheme === 'dark' ? '#ffffff08' : '#f5f5f5',
      },
    },
  };

  const contextValue: ThemeContextValue = {
    theme: currentTheme,
    setTheme,
  };

  return (
    <ThemeContext.Provider value={contextValue}>
      <ConfigProvider theme={antdTheme}>
        {children}
      </ConfigProvider>
    </ThemeContext.Provider>
  );
};

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

export default ThemeProvider;