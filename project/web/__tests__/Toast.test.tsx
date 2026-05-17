/**
 * IsoCortex — Toast System Tests
 * ================================
 */

import React from 'react';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { ToastProvider, useToast } from '@/components/ui/Toast';

// Test helper component that uses useToast
function ToastTestConsumer() {
  const { addToast, toasts } = useToast();
  return (
    <div>
      <button onClick={() => addToast('Success message', 'success')}>Add Success</button>
      <button onClick={() => addToast('Error message', 'error')}>Add Error</button>
      <button onClick={() => addToast('Warning message', 'warning')}>Add Warning</button>
      <button onClick={() => addToast('Info message', 'info')}>Add Info</button>
      <span data-testid="toast-count">{toasts.length}</span>
    </div>
  );
}

describe('ToastProvider', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('renders children', () => {
    render(
      <ToastProvider>
        <div>Child content</div>
      </ToastProvider>
    );
    expect(screen.getByText('Child content')).toBeInTheDocument();
  });

  it('adds and renders a toast', () => {
    render(
      <ToastProvider>
        <ToastTestConsumer />
      </ToastProvider>
    );

    expect(screen.getByTestId('toast-count').textContent).toBe('0');

    act(() => {
      fireEvent.click(screen.getByText('Add Success'));
    });

    expect(screen.getByTestId('toast-count').textContent).toBe('1');
    expect(screen.getByText('Success message')).toBeInTheDocument();
  });

  it('renders different toast types', () => {
    render(
      <ToastProvider>
        <ToastTestConsumer />
      </ToastProvider>
    );

    act(() => {
      fireEvent.click(screen.getByText('Add Error'));
    });
    expect(screen.getByText('Error message')).toBeInTheDocument();

    act(() => {
      fireEvent.click(screen.getByText('Add Warning'));
    });
    expect(screen.getByText('Warning message')).toBeInTheDocument();

    act(() => {
      fireEvent.click(screen.getByText('Add Info'));
    });
    expect(screen.getByText('Info message')).toBeInTheDocument();
  });

  it('removes toast when clicked', () => {
    render(
      <ToastProvider>
        <ToastTestConsumer />
      </ToastProvider>
    );

    act(() => {
      fireEvent.click(screen.getByText('Add Success'));
    });

    expect(screen.getByText('Success message')).toBeInTheDocument();

    act(() => {
      fireEvent.click(screen.getByText('Success message'));
    });

    expect(screen.queryByText('Success message')).not.toBeInTheDocument();
  });

  it('auto-removes toast after 4 seconds', () => {
    render(
      <ToastProvider>
        <ToastTestConsumer />
      </ToastProvider>
    );

    act(() => {
      fireEvent.click(screen.getByText('Add Success'));
    });

    expect(screen.getByText('Success message')).toBeInTheDocument();

    act(() => {
      jest.advanceTimersByTime(4100);
    });

    expect(screen.queryByText('Success message')).not.toBeInTheDocument();
  });
});

describe('useToast', () => {
  it('throws error when used outside ToastProvider', () => {
    // Suppress the expected console.error
    const spy = jest.spyOn(console, 'error').mockImplementation();

    expect(() => {
      render(<ToastTestConsumer />);
    }).toThrow('useToast must be used within ToastProvider');

    spy.mockRestore();
  });
});
