import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { vi } from 'vitest';
import { MainLayout } from '../MainLayout';
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

describe('MainLayout', () => {
  beforeEach(() => {
    // Mock window.innerWidth for responsive tests
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 1024,
    });
  });

  it('renders main layout with all components', () => {
    render(
      <TestWrapper>
        <MainLayout>
          <div data-testid="content">Test Content</div>
        </MainLayout>
      </TestWrapper>
    );

    expect(screen.getByText('myQuant 量化交易系统')).toBeInTheDocument();
    expect(screen.getByTestId('content')).toBeInTheDocument();
    expect(screen.getByText('系统仪表板')).toBeInTheDocument();
  });

  it('handles sidebar collapse correctly', () => {
    const onSidebarCollapse = vi.fn();
    
    render(
      <TestWrapper>
        <MainLayout onSidebarCollapse={onSidebarCollapse}>
          <div>Test Content</div>
        </MainLayout>
      </TestWrapper>
    );

    const collapseButton = screen.getByRole('button');
    fireEvent.click(collapseButton);

    expect(onSidebarCollapse).toHaveBeenCalledWith(true);
  });

  it('shows breadcrumb when enabled', () => {
    render(
      <TestWrapper>
        <MainLayout showBreadcrumb={true}>
          <div>Test Content</div>
        </MainLayout>
      </TestWrapper>
    );

    // Breadcrumb should be present in the DOM
    expect(document.querySelector('.ant-breadcrumb')).toBeInTheDocument();
  });

  it('hides breadcrumb when disabled', () => {
    render(
      <TestWrapper>
        <MainLayout showBreadcrumb={false}>
          <div>Test Content</div>
        </MainLayout>
      </TestWrapper>
    );

    // Breadcrumb should not be present
    expect(document.querySelector('.ant-breadcrumb')).not.toBeInTheDocument();
  });

  it('shows footer when enabled', () => {
    render(
      <TestWrapper>
        <MainLayout showFooter={true}>
          <div>Test Content</div>
        </MainLayout>
      </TestWrapper>
    );

    expect(screen.getByText(/myQuant 量化交易系统/)).toBeInTheDocument();
  });

  it('handles mobile responsive design', () => {
    // Mock mobile screen size
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 600,
    });

    render(
      <TestWrapper>
        <MainLayout>
          <div>Test Content</div>
        </MainLayout>
      </TestWrapper>
    );

    // Trigger resize event
    fireEvent(window, new Event('resize'));

    // On mobile, the layout should adapt
    expect(screen.getByText('Test Content')).toBeInTheDocument();
  });
});