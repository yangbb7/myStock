import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { vi } from 'vitest';
import { Header } from '../Header';
import { ThemeProvider } from '../ThemeProvider';

// Mock the stores
vi.mock('../../../stores/systemStore', () => ({
  useSystemStore: () => ({
    systemStatus: { isRunning: true, uptime: 3600, modules: {} },
    theme: 'light',
    setTheme: vi.fn(),
  }),
}));

vi.mock('../../../stores/userStore', () => ({
  useUserStore: () => ({
    user: { id: '1', name: '测试用户', email: 'test@example.com', role: 'admin' },
    isAuthenticated: true,
    logout: vi.fn(),
  }),
}));

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <BrowserRouter>
          {children}
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
};

describe('Header', () => {
  const defaultProps = {
    collapsed: false,
    onCollapse: jest.fn(),
    isMobile: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders header with system title', () => {
    render(
      <TestWrapper>
        <Header {...defaultProps} />
      </TestWrapper>
    );

    expect(screen.getByText('myQuant 量化交易系统')).toBeInTheDocument();
  });

  it('shows system status indicator', () => {
    render(
      <TestWrapper>
        <Header {...defaultProps} />
      </TestWrapper>
    );

    expect(screen.getByText('运行中')).toBeInTheDocument();
  });

  it('handles collapse button click', () => {
    const onCollapse = vi.fn();
    
    render(
      <TestWrapper>
        <Header {...defaultProps} onCollapse={onCollapse} />
      </TestWrapper>
    );

    const collapseButton = screen.getByRole('button');
    fireEvent.click(collapseButton);

    expect(onCollapse).toHaveBeenCalledWith(true);
  });

  it('shows correct collapse icon based on state', () => {
    const { rerender } = render(
      <TestWrapper>
        <Header {...defaultProps} collapsed={false} />
      </TestWrapper>
    );

    expect(screen.getByLabelText('menu-fold')).toBeInTheDocument();

    rerender(
      <TestWrapper>
        <Header {...defaultProps} collapsed={true} />
      </TestWrapper>
    );

    expect(screen.getByLabelText('menu-unfold')).toBeInTheDocument();
  });

  it('hides title on mobile', () => {
    render(
      <TestWrapper>
        <Header {...defaultProps} isMobile={true} />
      </TestWrapper>
    );

    expect(screen.queryByText('myQuant 量化交易系统')).not.toBeInTheDocument();
  });

  it('shows theme toggle switch', () => {
    render(
      <TestWrapper>
        <Header {...defaultProps} />
      </TestWrapper>
    );

    expect(screen.getByRole('switch')).toBeInTheDocument();
  });

  it('shows user avatar and name', () => {
    render(
      <TestWrapper>
        <Header {...defaultProps} />
      </TestWrapper>
    );

    expect(screen.getByText('测试用户')).toBeInTheDocument();
  });

  it('shows notification bell', () => {
    render(
      <TestWrapper>
        <Header {...defaultProps} />
      </TestWrapper>
    );

    expect(screen.getByLabelText('bell')).toBeInTheDocument();
  });
});