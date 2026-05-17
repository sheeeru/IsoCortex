'use client';

import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { authApi, clearTokens, getToken } from './api';
import { User } from './types';

interface AuthContextType {
  user: User | null;
  loading: boolean;
  isAuthenticated: boolean;
  isAdmin: boolean;
  isFirstRun: boolean | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  loading: true,
  isAuthenticated: false,
  isAdmin: false,
  isFirstRun: null,
  login: async () => {},
  logout: () => {},
  refreshUser: async () => {},
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [isFirstRun, setIsFirstRun] = useState<boolean | null>(null);

  const checkFirstRun = useCallback(async () => {
    try {
      // Check if setup has been completed on the backend
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8900'}/api/v1/auth/setup`
      );
      if (res.status === 409) {
        // Setup already completed
        setIsFirstRun(false);
      } else if (res.status === 200 || res.status === 201) {
        // Setup endpoint is available (not yet completed)
        setIsFirstRun(true);
      }
    } catch {
      // API not reachable yet
      setIsFirstRun(null);
    }
  }, []);

  const refreshUser = useCallback(async () => {
    const token = getToken();
    if (!token) {
      setUser(null);
      setLoading(false);
      return;
    }

    try {
      const userData = await authApi.me();
      setUser(userData);
      setIsFirstRun(false);
    } catch {
      // Token invalid
      clearTokens();
      setUser(null);
      setIsFirstRun(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshUser();
    checkFirstRun();
  }, [refreshUser, checkFirstRun]);

  const login = async (username: string, password: string) => {
    const { setTokens } = await import('./api');
    const response = await authApi.login({ username, password });
    setTokens(response.access_token, response.refresh_token);
    await refreshUser();
  };

  const logout = () => {
    clearTokens();
    setUser(null);
    window.location.href = '/login';
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        isAuthenticated: !!user,
        isAdmin: user?.role === 'admin',
        isFirstRun,
        login,
        logout,
        refreshUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}

export default AuthContext;
