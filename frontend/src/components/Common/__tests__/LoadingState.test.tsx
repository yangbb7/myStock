import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { LoadingState } from '../LoadingState';

describe('LoadingState', () => {
  it('should render default loading spinner', () => {
    render(<LoadingState />);
    
    expect(document.querySelector('.ant-spin')).toBeInTheDocument();
  });

  it('should render with custom message', () => {
    render(<LoadingState message="Loading data..." />);
    
    expect(screen.getByText('Loading data...')).toBeInTheDocument();
  });

  it('should render with large size', () => {
    render(<LoadingState size="large" />);
    
    expect(document.querySelector('.ant-spin-lg')).toBeInTheDocument();
  });

  it('should render with small size', () => {
    render(<LoadingState size="small" />);
    
    expect(document.querySelector('.ant-spin-sm')).toBeInTheDocument();
  });

  it('should render skeleton when type is skeleton', () => {
    render(<LoadingState type="skeleton" />);
    
    expect(document.querySelector('.ant-skeleton')).toBeInTheDocument();
  });

  it('should render skeleton with custom rows', () => {
    render(<LoadingState type="skeleton" rows={5} />);
    
    const skeleton = document.querySelector('.ant-skeleton');
    expect(skeleton).toBeInTheDocument();
  });

  it('should render progress bar when type is progress', () => {
    render(<LoadingState type="progress" progress={50} />);
    
    expect(document.querySelector('.ant-progress')).toBeInTheDocument();
  });

  it('should render with custom className', () => {
    const { container } = render(<LoadingState className="custom-loading" />);
    
    expect(container.querySelector('.custom-loading')).toBeInTheDocument();
  });

  it('should render centered by default', () => {
    const { container } = render(<LoadingState />);
    
    expect(container.firstChild).toHaveStyle({
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center'
    });
  });

  it('should render with custom height', () => {
    const { container } = render(<LoadingState height="200px" />);
    
    expect(container.firstChild).toHaveStyle({ height: '200px' });
  });
});