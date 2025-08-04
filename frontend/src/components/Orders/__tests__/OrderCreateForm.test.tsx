import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { vi } from 'vitest';
import OrderCreateForm from '../OrderCreateForm';
import * as api from '../../../services/api';

// Mock the API
vi.mock('../../../services/api', () => ({
  api: {
    data: {
      getSymbols: vi.fn(),
    },
    portfolio: {
      getSummary: vi.fn(),
    },
    risk: {
      getMetrics: vi.fn(),
    },
    order: {
      createOrder: vi.fn(),
    },
  },
}));

const mockApi = api.api as any;

describe('OrderCreateForm', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });

    // Setup default mock responses
    mockApi.data.getSymbols.mockResolvedValue(['000001.SZ', '000002.SZ', '600000.SH']);
    mockApi.portfolio.getSummary.mockResolvedValue({
      totalValue: 1000000,
      cashBalance: 500000,
      positions: {},
      unrealizedPnl: 0,
      positionsCount: 0,
    });
    mockApi.risk.getMetrics.mockResolvedValue({
      dailyPnl: 0,
      currentDrawdown: 0,
      riskLimits: {
        maxPositionSize: 0.2,
        maxDrawdownLimit: 0.1,
        maxDailyLoss: 50000,
      },
      riskUtilization: {
        dailyLossRatio: 0,
        drawdownRatio: 0,
      },
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  const renderComponent = (props = {}) => {
    return render(
      <QueryClientProvider client={queryClient}>
        <OrderCreateForm {...props} />
      </QueryClientProvider>
    );
  };

  it('should render order creation form', async () => {
    renderComponent();

    expect(screen.getByRole('heading', { name: '创建订单' })).toBeInTheDocument();
    expect(screen.getByLabelText('股票代码')).toBeInTheDocument();
    expect(screen.getByLabelText('买卖方向')).toBeInTheDocument();
    expect(screen.getByLabelText('订单类型')).toBeInTheDocument();
    expect(screen.getByLabelText('数量')).toBeInTheDocument();
  });

  it('should show price field for limit orders', async () => {
    renderComponent();

    // Change order type to LIMIT
    const orderTypeSelect = screen.getByLabelText('订单类型');
    fireEvent.mouseDown(orderTypeSelect);
    
    await waitFor(() => {
      const limitOption = screen.getByText('限价单');
      fireEvent.click(limitOption);
    });

    expect(screen.getByLabelText('价格')).toBeInTheDocument();
  });

  it('should calculate estimated value for limit orders', async () => {
    renderComponent();

    // Change to limit order
    const orderTypeSelect = screen.getByLabelText('订单类型');
    fireEvent.mouseDown(orderTypeSelect);
    await waitFor(() => {
      fireEvent.click(screen.getByText('限价单'));
    });

    // Fill in quantity and price
    const quantityInput = screen.getByLabelText('数量');
    const priceInput = screen.getByLabelText('价格');

    fireEvent.change(quantityInput, { target: { value: '1000' } });
    fireEvent.change(priceInput, { target: { value: '10.50' } });

    await waitFor(() => {
      expect(screen.getByText(/预估金额: ¥10,500/)).toBeInTheDocument();
    });
  });

  it('should show risk warnings when insufficient funds', async () => {
    // Mock portfolio with low cash balance
    mockApi.portfolio.getSummary.mockResolvedValue({
      totalValue: 100000,
      cashBalance: 5000,
      positions: {},
      unrealizedPnl: 0,
      positionsCount: 0,
    });

    renderComponent();

    // Change to limit order and fill large order
    const orderTypeSelect = screen.getByLabelText('订单类型');
    fireEvent.mouseDown(orderTypeSelect);
    await waitFor(() => {
      fireEvent.click(screen.getByText('限价单'));
    });

    const quantityInput = screen.getByLabelText('数量');
    const priceInput = screen.getByLabelText('价格');

    fireEvent.change(quantityInput, { target: { value: '1000' } });
    fireEvent.change(priceInput, { target: { value: '10.00' } });

    await waitFor(() => {
      expect(screen.getByText('风险提示')).toBeInTheDocument();
      expect(screen.getByText(/资金不足/)).toBeInTheDocument();
    });
  });

  it('should submit order successfully', async () => {
    const mockOnSuccess = vi.fn();
    mockApi.order.createOrder.mockResolvedValue({ orderId: 'ORDER123' });

    renderComponent({ onSuccess: mockOnSuccess });

    // Fill required fields
    const symbolSelect = screen.getByLabelText('股票代码');
    fireEvent.mouseDown(symbolSelect);
    await waitFor(() => {
      const options = screen.getAllByText('000001.SZ');
      fireEvent.click(options[0]);
    });

    const quantityInput = screen.getByLabelText('数量');
    fireEvent.change(quantityInput, { target: { value: '100' } });

    // Submit form
    const submitButton = screen.getByText('创建订单');
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(mockApi.order.createOrder).toHaveBeenCalledWith({
        symbol: '000001.SZ',
        side: 'BUY',
        quantity: 100,
        orderType: 'MARKET',
        timeInForce: 'DAY',
      });
      expect(mockOnSuccess).toHaveBeenCalledWith('ORDER123');
    });
  });
});