import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BaseChart } from '../BaseChart';

// Mock ECharts
const mockChart = {
  setOption: vi.fn(),
  resize: vi.fn(),
  dispose: vi.fn(),
  on: vi.fn(),
  off: vi.fn(),
  showLoading: vi.fn(),
  hideLoading: vi.fn(),
};

const mockECharts = {
  init: vi.fn(() => mockChart),
  dispose: vi.fn(),
  getInstanceByDom: vi.fn(),
};

vi.mock('echarts', () => ({
  default: mockECharts,
}));

describe('BaseChart', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should render chart container', () => {
    const option = {
      title: { text: 'Test Chart' },
      xAxis: { type: 'category', data: ['A', 'B', 'C'] },
      yAxis: { type: 'value' },
      series: [{ data: [1, 2, 3], type: 'line' }],
    };

    render(<BaseChart option={option} />);

    expect(screen.getByTestId('chart-container')).toBeInTheDocument();
  });

  it('should initialize ECharts instance', () => {
    const option = {
      title: { text: 'Test Chart' },
      series: [{ data: [1, 2, 3], type: 'line' }],
    };

    render(<BaseChart option={option} />);

    expect(mockECharts.init).toHaveBeenCalled();
    expect(mockChart.setOption).toHaveBeenCalledWith(option, true);
  });

  it('should handle loading state', () => {
    const option = { series: [{ data: [1, 2, 3], type: 'line' }] };

    render(<BaseChart option={option} loading={true} />);

    expect(mockChart.showLoading).toHaveBeenCalled();
  });

  it('should hide loading when not loading', () => {
    const option = { series: [{ data: [1, 2, 3], type: 'line' }] };

    const { rerender } = render(<BaseChart option={option} loading={true} />);
    
    rerender(<BaseChart option={option} loading={false} />);

    expect(mockChart.hideLoading).toHaveBeenCalled();
  });

  it('should update chart when option changes', () => {
    const initialOption = { series: [{ data: [1, 2, 3], type: 'line' }] };
    const updatedOption = { series: [{ data: [4, 5, 6], type: 'line' }] };

    const { rerender } = render(<BaseChart option={initialOption} />);
    
    rerender(<BaseChart option={updatedOption} />);

    expect(mockChart.setOption).toHaveBeenCalledWith(updatedOption, true);
  });

  it('should handle resize', () => {
    const option = { series: [{ data: [1, 2, 3], type: 'line' }] };

    render(<BaseChart option={option} />);

    // Simulate window resize
    window.dispatchEvent(new Event('resize'));

    expect(mockChart.resize).toHaveBeenCalled();
  });

  it('should dispose chart on unmount', () => {
    const option = { series: [{ data: [1, 2, 3], type: 'line' }] };

    const { unmount } = render(<BaseChart option={option} />);
    
    unmount();

    expect(mockChart.dispose).toHaveBeenCalled();
  });

  it('should apply custom height and width', () => {
    const option = { series: [{ data: [1, 2, 3], type: 'line' }] };

    render(<BaseChart option={option} height="400px" width="600px" />);

    const container = screen.getByTestId('chart-container');
    expect(container).toHaveStyle({ height: '400px', width: '600px' });
  });

  it('should handle chart events', () => {
    const onChartClick = vi.fn();
    const option = { series: [{ data: [1, 2, 3], type: 'line' }] };

    render(<BaseChart option={option} onEvents={{ click: onChartClick }} />);

    expect(mockChart.on).toHaveBeenCalledWith('click', onChartClick);
  });

  it('should handle empty data gracefully', () => {
    const option = { series: [] };

    render(<BaseChart option={option} />);

    expect(mockECharts.init).toHaveBeenCalled();
    expect(mockChart.setOption).toHaveBeenCalledWith(option, true);
  });

  it('should apply theme', () => {
    const option = { series: [{ data: [1, 2, 3], type: 'line' }] };

    render(<BaseChart option={option} theme="dark" />);

    expect(mockECharts.init).toHaveBeenCalledWith(
      expect.any(Element),
      'dark'
    );
  });
});