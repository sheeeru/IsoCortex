/**
 * IsoCortex — Login Page Tests
 * ==============================
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import LoginPage from '@/app/login/page';
import { ToastProvider } from '@/components/ui/Toast';
import { AuthProvider } from '@/lib/auth';

// Mock Next.js router
const mockPush = jest.fn();
jest.mock('next/navigation', () => ({
  useRouter: () => ({ push: mockPush, replace: mockPush }),
  usePathname: () => '/login',
}));

// Mock the API module
jest.mock('@/lib/api', () => ({
  authApi: {
    login: jest.fn(),
    me: jest.fn(),
    setup: jest.fn(),
  },
  ApiError: class extends Error {
    status: number;
    detail: string;
    constructor(status: number, detail: string) {
      super(detail);
      this.status = status;
      this.detail = detail;
      this.name = 'ApiError';
    }
  },
  setTokens: jest.fn(),
  clearTokens: jest.fn(),
  getToken: jest.fn().mockReturnValue(null),
}));

// Mock fetch for health check
global.fetch = jest.fn();

const { authApi } = jest.requireMock('@/lib/api');

function renderLoginPage() {
  return render(
    <AuthProvider>
      <ToastProvider>
        <LoginPage />
      </ToastProvider>
    </AuthProvider>
  );
}

describe('LoginPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockPush.mockClear();
    // Make health check return quickly
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'healthy' }),
    });
  });

  it('renders the login form with branding', async () => {
    await act(async () => {
      renderLoginPage();
    });

    await waitFor(() => {
      expect(screen.getByText('IsoCortex')).toBeInTheDocument();
    });

    expect(screen.getByText('Sign In')).toBeInTheDocument();
    expect(screen.getByText('Semantic Search Engine')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Enter your username')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Enter your password')).toBeInTheDocument();
    expect(screen.getByText('Sign In')).toBeInTheDocument();
  });

  it('has disabled submit button when fields are empty', async () => {
    await act(async () => {
      renderLoginPage();
    });

    await waitFor(() => {
      expect(screen.getByText('Sign In')).toBeInTheDocument();
    });

    const submitBtn = screen.getByText('Sign In');
    expect(submitBtn).toBeDisabled();
  });

  it('enables submit button when both fields are filled', async () => {
    await act(async () => {
      renderLoginPage();
    });

    await waitFor(() => {
      expect(screen.getByText('Sign In')).toBeInTheDocument();
    });

    const usernameInput = screen.getByPlaceholderText('Enter your username');
    const passwordInput = screen.getByPlaceholderText('Enter your password');

    await act(async () => {
      fireEvent.change(usernameInput, { target: { value: 'admin' } });
    });
    await act(async () => {
      fireEvent.change(passwordInput, { target: { value: 'password123' } });
    });

    const submitBtn = screen.getByText('Sign In');
    expect(submitBtn).not.toBeDisabled();
  });

  it('shows error message on login failure', async () => {
    authApi.login.mockRejectedValue(
      new (jest.requireMock('@/lib/api').ApiError)(401, 'Invalid credentials')
    );

    await act(async () => {
      renderLoginPage();
    });

    await waitFor(() => {
      expect(screen.getByText('Sign In')).toBeInTheDocument();
    });

    const usernameInput = screen.getByPlaceholderText('Enter your username');
    const passwordInput = screen.getByPlaceholderText('Enter your password');

    await act(async () => {
      fireEvent.change(usernameInput, { target: { value: 'admin' } });
    });
    await act(async () => {
      fireEvent.change(passwordInput, { target: { value: 'wrong' } });
    });
    await act(async () => {
      fireEvent.click(screen.getByText('Sign In'));
    });

    await waitFor(() => {
      expect(screen.getByText('Invalid credentials')).toBeInTheDocument();
    });
  });

  it('has a link to the setup page', async () => {
    await act(async () => {
      renderLoginPage();
    });

    await waitFor(() => {
      expect(screen.getByText('Run initial setup')).toBeInTheDocument();
    });

    await act(async () => {
      fireEvent.click(screen.getByText('Run initial setup'));
    });

    expect(mockPush).toHaveBeenCalledWith('/setup');
  });

  it('shows IC logo', async () => {
    await act(async () => {
      renderLoginPage();
    });

    await waitFor(() => {
      expect(screen.getByText('IC')).toBeInTheDocument();
    });
  });
});
