import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { vi } from 'vitest';
import { Sidebar } from '../Sidebar';

// Mock react-router-dom hooks
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
    useLocation: () => ({ pathname: '/dashboard' }),
  };
});

describe('Sidebar', () => {
  const defaultProps = {
    collapsed: false,
    onCollapse: jest.fn(),
    isMobile: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders all menu items', () => {
    render(
      <BrowserRouter>
        <Sidebar {...defaultProps} />
      </BrowserRouter>
    );

    expect(screen.getByText('系统仪表板')).toBeInTheDocument();
    expect(screen.getByText('策略管理')).toBeInTheDocument();
    expect(screen.getByText('实时数据')).toBeInTheDocument();
    expect(screen.getByText('订单管理')).toBeInTheDocument();
    expect(screen.getByText('投资组合')).toBeInTheDocument();
    expect(screen.getByText('风险监控')).toBeInTheDocument();
    expect(screen.getByText('回测分析')).toBeInTheDocument();
    expect(screen.getByText('系统管理')).toBeInTheDocument();
  });

  it('shows logo when not collapsed', () => {
    render(
      <BrowserRouter>
        <Sidebar {...defaultProps} collapsed={false} />
      </BrowserRouter>
    );

    expect(screen.getByText('myQuant')).toBeInTheDocument();
  });

  it('shows abbreviated logo when collapsed', () => {
    render(
      <BrowserRouter>
        <Sidebar {...defaultProps} collapsed={true} />
      </BrowserRouter>
    );

    expect(screen.getByText('mQ')).toBeInTheDocument();
  });

  it('handles menu item click and navigation', () => {
    render(
      <BrowserRouter>
        <Sidebar {...defaultProps} />
      </BrowserRouter>
    );

    const strategyMenuItem = screen.getByText('策略管理');
    fireEvent.click(strategyMenuItem);

    expect(mockNavigate).toHaveBeenCalledWith('/strategy');
  });

  it('closes sidebar on mobile after navigation', () => {
    const onCollapse = vi.fn();
    
    render(
      <BrowserRouter>
        <Sidebar {...defaultProps} isMobile={true} onCollapse={onCollapse} />
      </BrowserRouter>
    );

    const menuItem = screen.getByText('策略管理');
    fireEvent.click(menuItem);

    expect(onCollapse).toHaveBeenCalledWith(true);
  });

  it('shows mobile overlay when not collapsed on mobile', () => {
    const { container } = render(
      <BrowserRouter>
        <Sidebar {...defaultProps} isMobile={true} collapsed={false} />
      </BrowserRouter>
    );

    const overlay = container.querySelector('div[style*="position: fixed"]');
    expect(overlay).toBeInTheDocument();
  });

  it('handles mobile overlay click', () => {
    const onCollapse = vi.fn();
    const { container } = render(
      <BrowserRouter>
        <Sidebar {...defaultProps} isMobile={true} collapsed={false} onCollapse={onCollapse} />
      </BrowserRouter>
    );

    const overlay = container.querySelector('div[style*="position: fixed"]');
    if (overlay) {
      fireEvent.click(overlay);
      expect(onCollapse).toHaveBeenCalledWith(true);
    }
  });
});