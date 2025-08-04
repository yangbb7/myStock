import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatisticCard } from '../StatisticCard';

describe('StatisticCard', () => {
  it('should render title and value correctly', () => {
    render(
      <StatisticCard
        title="Test Title"
        value={123.45}
        precision={2}
        prefix="$"
        suffix="USD"
      />
    );

    expect(screen.getByText('Test Title')).toBeInTheDocument();
    expect(screen.getByText('$123.45USD')).toBeInTheDocument();
  });

  it('should apply positive value style', () => {
    render(
      <StatisticCard
        title="Profit"
        value={100}
        valueStyle={{ color: '#3f8600' }}
      />
    );

    const valueElement = screen.getByText('100');
    expect(valueElement).toHaveStyle({ color: '#3f8600' });
  });

  it('should apply negative value style', () => {
    render(
      <StatisticCard
        title="Loss"
        value={-100}
        valueStyle={{ color: '#cf1322' }}
      />
    );

    const valueElement = screen.getByText('-100');
    expect(valueElement).toHaveStyle({ color: '#cf1322' });
  });

  it('should handle loading state', () => {
    render(
      <StatisticCard
        title="Loading Value"
        value={undefined}
        loading={true}
      />
    );

    expect(screen.getByText('Loading Value')).toBeInTheDocument();
    // Should show loading skeleton or spinner
    expect(document.querySelector('.ant-skeleton')).toBeInTheDocument();
  });

  it('should format large numbers correctly', () => {
    render(
      <StatisticCard
        title="Large Number"
        value={1234567.89}
        precision={2}
        formatter={(value) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
      />
    );

    expect(screen.getByText('1,234,567.89')).toBeInTheDocument();
  });

  it('should handle zero values', () => {
    render(
      <StatisticCard
        title="Zero Value"
        value={0}
        precision={2}
      />
    );

    expect(screen.getByText('0.00')).toBeInTheDocument();
  });

  it('should render with custom className', () => {
    const { container } = render(
      <StatisticCard
        title="Custom Class"
        value={100}
        className="custom-statistic"
      />
    );

    expect(container.querySelector('.custom-statistic')).toBeInTheDocument();
  });
});