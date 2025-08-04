import React from 'react';
import { Card, Statistic } from 'antd';
import type { StatisticProps } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined } from '@ant-design/icons';

interface StatisticCardProps {
  title: string;
  value: string | number;
  precision?: number;
  prefix?: React.ReactNode;
  suffix?: React.ReactNode;
  valueStyle?: React.CSSProperties;
  loading?: boolean;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string | number;
  extra?: React.ReactNode;
  bordered?: boolean;
  size?: 'default' | 'small';
  className?: string;
}

export const StatisticCard: React.FC<StatisticCardProps> = ({
  title,
  value,
  precision,
  prefix,
  suffix,
  valueStyle,
  loading = false,
  trend,
  trendValue,
  extra,
  bordered = true,
  size = 'default',
  className,
}) => {
  const getTrendIcon = () => {
    switch (trend) {
      case 'up':
        return <ArrowUpOutlined style={{ color: '#3f8600' }} />;
      case 'down':
        return <ArrowDownOutlined style={{ color: '#cf1322' }} />;
      default:
        return null;
    }
  };

  const getTrendColor = () => {
    switch (trend) {
      case 'up':
        return '#3f8600';
      case 'down':
        return '#cf1322';
      default:
        return undefined;
    }
  };

  const statisticProps: StatisticProps = {
    title,
    value,
    precision,
    prefix,
    suffix,
    valueStyle: {
      ...valueStyle,
      color: valueStyle?.color || getTrendColor(),
    },
    loading,
  };

  return (
    <Card
      bordered={bordered}
      size={size}
      className={className}
      extra={extra}
      loading={loading}
    >
      <Statistic {...statisticProps} />
      {trend && trendValue && (
        <div style={{ marginTop: 8, fontSize: '14px', color: getTrendColor() }}>
          {getTrendIcon()} {trendValue}
        </div>
      )}
    </Card>
  );
};

export default StatisticCard;