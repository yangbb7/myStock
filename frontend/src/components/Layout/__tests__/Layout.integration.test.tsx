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

describe('Layout Integration', () => {
  it('renders complete layout with all components', () => {
    render(
      <TestWrapper>
        <MainLayout>
          <div data-testid="test-content">Test Content</div>
        </MainLayout>
      </TestWrapper>
    );

    // Check if all main layout components are present
    expect(screen.getByText('myQuant 量化交易系统')).toBeInTheDocument();
    expect(screen.getByText('系统仪表板')).toBeInTheDocument();
    expect(screen.getByText('策略管理')).toBeInTheDocument();
    expect(screen.getByTestId('test-content')).toBeInTheDocument();
    expect(screen.getByText(/myQuant 量化交易系统.*Created with/)).toBeInTheDocument();
  });

  it('handles theme switching', () => {
    render(
      <TestWrapper>
        <MainLayout>
          <div>Content</div>
        </MainLayout>
      </TestWrapper>
    );

    // Find and click the theme toggle switch
    const themeSwitch = screen.getByRole('switch');
    expect(themeSwitch).toBeInTheDocument();
    
    fireEvent.click(themeSwitch);
    // Theme change would be handled by the mocked store
  });

  it('handles sidebar collapse', () => {
    render(
      <TestWrapper>
        <MainLayout>
          <div>Content</div>
        </MainLayout>
      </TestWrapper>
    );

    // Find and click the collapse button
    const collapseButton = screen.getAllByRole('button')[0]; // First button should be collapse
    fireEvent.click(collapseButton);
    
    // The sidebar should respond to the collapse action
    expect(collapseButton).toBeInTheDocument();
  });

  it('shows system status indicator', () => {
    render(
      <TestWrapper>
        <MainLayout>
          <div>Content</div>
        </MainLayout>
      </TestWrapper>
    );

    expect(screen.getByText('运行中')).toBeInTheDocument();
  });

  it('shows user information', () => {
    render(
      <TestWrapper>
        <MainLayout>
          <div>Content</div>
        </MainLayout>
      </TestWrapper>
    );

    expect(screen.getByText('测试用户')).toBeInTheDocument();
  });
});