import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import { SettingsModal } from '../SettingsModal';

// Mock the system store
vi.mock('../../../stores/systemStore', () => ({
  useSystemStore: () => ({
    theme: 'light',
    setTheme: vi.fn(),
  }),
}));

describe('SettingsModal', () => {
  const defaultProps = {
    visible: true,
    onClose: jest.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders settings modal when visible', () => {
    render(<SettingsModal {...defaultProps} />);

    expect(screen.getByText('系统设置')).toBeInTheDocument();
    expect(screen.getByText('外观设置')).toBeInTheDocument();
    expect(screen.getByText('数据刷新设置')).toBeInTheDocument();
    expect(screen.getByText('通知设置')).toBeInTheDocument();
  });

  it('does not render when not visible', () => {
    render(<SettingsModal {...defaultProps} visible={false} />);

    expect(screen.queryByText('系统设置')).not.toBeInTheDocument();
  });

  it('shows all form fields', () => {
    render(<SettingsModal {...defaultProps} />);

    expect(screen.getByText('主题模式')).toBeInTheDocument();
    expect(screen.getByText('语言设置')).toBeInTheDocument();
    expect(screen.getByText('紧凑模式')).toBeInTheDocument();
    expect(screen.getByText('显示动画')).toBeInTheDocument();
    expect(screen.getByText('自动刷新')).toBeInTheDocument();
    expect(screen.getByText('刷新间隔（秒）')).toBeInTheDocument();
    expect(screen.getByText('启用通知')).toBeInTheDocument();
    expect(screen.getByText('声音提醒')).toBeInTheDocument();
  });

  it('calls onClose when cancel button is clicked', () => {
    const onClose = vi.fn();
    render(<SettingsModal {...defaultProps} onClose={onClose} />);

    const cancelButton = screen.getByText('取消');
    fireEvent.click(cancelButton);

    expect(onClose).toHaveBeenCalled();
  });

  it('saves settings when save button is clicked', async () => {
    const onClose = vi.fn();
    render(<SettingsModal {...defaultProps} onClose={onClose} />);

    const saveButton = screen.getByText('保存');
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(onClose).toHaveBeenCalled();
    });
  });

  it('shows tooltips for form fields', () => {
    render(<SettingsModal {...defaultProps} />);

    // Check if tooltip icons are present
    const tooltipIcons = screen.getAllByLabelText('question-circle');
    expect(tooltipIcons.length).toBeGreaterThan(0);
  });
});