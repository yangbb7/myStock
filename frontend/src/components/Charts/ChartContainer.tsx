import React, { useState, useEffect } from 'react';
import { Card, Space, Button, Select, Tooltip, Dropdown, MenuProps } from 'antd';
import {
  FullscreenOutlined,
  FullscreenExitOutlined,
  DownloadOutlined,
  SettingOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import { BaseChart, BaseChartProps } from './BaseChart';
import { ChartThemeManager } from './ChartTheme';

interface ChartContainerProps extends BaseChartProps {
  title?: string;
  subtitle?: string;
  showToolbar?: boolean;
  showThemeSelector?: boolean;
  showFullscreen?: boolean;
  showDownload?: boolean;
  showRefresh?: boolean;
  showSettings?: boolean;
  onRefresh?: () => void;
  onDownload?: (format: 'png' | 'jpg' | 'svg') => void;
  onSettingsChange?: (settings: any) => void;
  customActions?: React.ReactNode;
  fullscreenContainer?: string; // CSS selector for fullscreen container
}

export const ChartContainer: React.FC<ChartContainerProps> = ({
  title,
  subtitle,
  showToolbar = true,
  showThemeSelector = true,
  showFullscreen = true,
  showDownload = true,
  showRefresh = false,
  showSettings = false,
  onRefresh,
  onDownload,
  onSettingsChange,
  customActions,
  fullscreenContainer,
  ...chartProps
}) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [currentTheme, setCurrentTheme] = useState('light');
  const [chartInstance, setChartInstance] = useState<echarts.ECharts | null>(null);

  const themeManager = ChartThemeManager.getInstance();

  useEffect(() => {
    const unsubscribe = themeManager.subscribe(setCurrentTheme);
    setCurrentTheme(themeManager.getTheme());
    return unsubscribe;
  }, [themeManager]);

  const handleFullscreen = () => {
    if (!isFullscreen) {
      const element = fullscreenContainer 
        ? document.querySelector(fullscreenContainer)
        : document.documentElement;
      
      if (element?.requestFullscreen) {
        element.requestFullscreen();
        setIsFullscreen(true);
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
        setIsFullscreen(false);
      }
    }
  };

  const handleThemeChange = (theme: string) => {
    themeManager.setTheme(theme);
  };

  const handleDownload = (format: 'png' | 'jpg' | 'svg') => {
    if (chartInstance) {
      const url = chartInstance.getDataURL({
        type: format === 'jpg' ? 'jpeg' : format,
        pixelRatio: 2,
        backgroundColor: currentTheme === 'dark' ? '#141414' : '#ffffff',
      });
      
      const link = document.createElement('a');
      link.download = `chart.${format}`;
      link.href = url;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
    
    if (onDownload) {
      onDownload(format);
    }
  };

  const downloadMenuItems: MenuProps['items'] = [
    {
      key: 'png',
      label: 'PNG 格式',
      onClick: () => handleDownload('png'),
    },
    {
      key: 'jpg',
      label: 'JPG 格式',
      onClick: () => handleDownload('jpg'),
    },
    {
      key: 'svg',
      label: 'SVG 格式',
      onClick: () => handleDownload('svg'),
    },
  ];

  const renderToolbar = () => {
    if (!showToolbar) return null;

    return (
      <Space size="small">
        {showThemeSelector && (
          <Select
            size="small"
            value={currentTheme}
            onChange={handleThemeChange}
            style={{ width: 80 }}
            options={[
              { label: '浅色', value: 'light' },
              { label: '深色', value: 'dark' },
              { label: '交易', value: 'trading' },
            ]}
          />
        )}

        {showRefresh && (
          <Tooltip title="刷新">
            <Button
              size="small"
              icon={<ReloadOutlined />}
              onClick={onRefresh}
            />
          </Tooltip>
        )}

        {showDownload && (
          <Dropdown menu={{ items: downloadMenuItems }} placement="bottomRight">
            <Tooltip title="下载">
              <Button
                size="small"
                icon={<DownloadOutlined />}
              />
            </Tooltip>
          </Dropdown>
        )}

        {showSettings && (
          <Tooltip title="设置">
            <Button
              size="small"
              icon={<SettingOutlined />}
              onClick={() => onSettingsChange?.({})}
            />
          </Tooltip>
        )}

        {showFullscreen && (
          <Tooltip title={isFullscreen ? '退出全屏' : '全屏'}>
            <Button
              size="small"
              icon={isFullscreen ? <FullscreenExitOutlined /> : <FullscreenOutlined />}
              onClick={handleFullscreen}
            />
          </Tooltip>
        )}

        {customActions}
      </Space>
    );
  };

  const cardTitle = (
    <div>
      <div style={{ fontSize: 16, fontWeight: 500 }}>{title}</div>
      {subtitle && (
        <div style={{ fontSize: 12, color: '#666', fontWeight: 'normal' }}>
          {subtitle}
        </div>
      )}
    </div>
  );

  return (
    <Card
      title={title ? cardTitle : undefined}
      extra={renderToolbar()}
      bordered={true}
      bodyStyle={{ 
        padding: 0,
        height: isFullscreen ? '100vh' : undefined,
      }}
      style={{
        height: isFullscreen ? '100vh' : undefined,
        position: isFullscreen ? 'fixed' : 'relative',
        top: isFullscreen ? 0 : undefined,
        left: isFullscreen ? 0 : undefined,
        width: isFullscreen ? '100vw' : undefined,
        zIndex: isFullscreen ? 9999 : undefined,
      }}
    >
      <BaseChart
        {...chartProps}
        theme={currentTheme}
        onChartReady={setChartInstance}
        height={isFullscreen ? 'calc(100vh - 60px)' : chartProps.height}
      />
    </Card>
  );
};

export default ChartContainer;