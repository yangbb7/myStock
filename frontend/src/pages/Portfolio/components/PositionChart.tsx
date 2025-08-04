import React, { useMemo } from 'react';
import { Card } from 'antd';
import { BaseChart } from '../../../components/Charts/BaseChart';
import type { EChartsOption } from 'echarts';

interface PositionData {
  symbol: string;
  marketValue: number;
  weight: number;
  unrealizedPnl: number;
}

interface PositionChartProps {
  data: PositionData[];
  type: 'pie' | 'bar';
  loading?: boolean;
}

export const PositionChart: React.FC<PositionChartProps> = ({
  data,
  type,
  loading = false,
}) => {
  const chartOption = useMemo((): EChartsOption => {
    if (!data || data.length === 0) {
      return {
        title: {
          text: '暂无数据',
          left: 'center',
          top: 'middle',
          textStyle: {
            color: '#999',
            fontSize: 14,
          },
        },
      };
    }

    // Sort data by market value for better visualization
    const sortedData = [...data].sort((a, b) => b.marketValue - a.marketValue);

    if (type === 'pie') {
      return {
        title: {
          text: '持仓分布',
          left: 'center',
          textStyle: {
            fontSize: 16,
            fontWeight: 'bold',
          },
        },
        tooltip: {
          trigger: 'item',
          formatter: (params: any) => {
            const { name, value, percent } = params;
            return `
              <div>
                <strong>${name}</strong><br/>
                市值: ¥${value.toLocaleString(undefined, { minimumFractionDigits: 2 })}<br/>
                占比: ${percent}%
              </div>
            `;
          },
        },
        legend: {
          type: 'scroll',
          orient: 'vertical',
          right: 10,
          top: 20,
          bottom: 20,
          data: sortedData.map(item => item.symbol),
        },
        series: [
          {
            name: '持仓分布',
            type: 'pie',
            radius: ['40%', '70%'],
            center: ['40%', '50%'],
            avoidLabelOverlap: false,
            itemStyle: {
              borderRadius: 10,
              borderColor: '#fff',
              borderWidth: 2,
            },
            label: {
              show: false,
              position: 'center',
            },
            emphasis: {
              label: {
                show: true,
                fontSize: 20,
                fontWeight: 'bold',
              },
            },
            labelLine: {
              show: false,
            },
            data: sortedData.map((item, index) => ({
              value: item.marketValue,
              name: item.symbol,
              itemStyle: {
                color: `hsl(${(index * 137.5) % 360}, 70%, 60%)`,
              },
            })),
          },
        ],
      };
    } else {
      // Bar chart
      return {
        title: {
          text: '持仓市值排名',
          left: 'center',
          textStyle: {
            fontSize: 16,
            fontWeight: 'bold',
          },
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'shadow',
          },
          formatter: (params: any) => {
            const data = params[0];
            const item = sortedData.find(d => d.symbol === data.name);
            return `
              <div>
                <strong>${data.name}</strong><br/>
                市值: ¥${data.value.toLocaleString(undefined, { minimumFractionDigits: 2 })}<br/>
                权重: ${item?.weight.toFixed(2)}%<br/>
                盈亏: ¥${item?.unrealizedPnl.toFixed(2)}
              </div>
            `;
          },
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          containLabel: true,
        },
        xAxis: {
          type: 'category',
          data: sortedData.map(item => item.symbol),
          axisLabel: {
            rotate: 45,
            interval: 0,
          },
        },
        yAxis: {
          type: 'value',
          name: '市值 (¥)',
          axisLabel: {
            formatter: (value: number) => {
              if (value >= 10000) {
                return `${(value / 10000).toFixed(1)}万`;
              }
              return value.toLocaleString();
            },
          },
        },
        series: [
          {
            name: '市值',
            type: 'bar',
            data: sortedData.map((item, index) => ({
              value: item.marketValue,
              itemStyle: {
                color: item.unrealizedPnl >= 0 ? '#52c41a' : '#ff4d4f',
                borderRadius: [4, 4, 0, 0],
              },
            })),
            emphasis: {
              itemStyle: {
                shadowBlur: 10,
                shadowOffsetX: 0,
                shadowColor: 'rgba(0, 0, 0, 0.5)',
              },
            },
          },
        ],
      };
    }
  }, [data, type]);

  return (
    <BaseChart
      option={chartOption}
      loading={loading}
      height={400}
      title={type === 'pie' ? '持仓分布' : '持仓排名'}
    />
  );
};

export default PositionChart;