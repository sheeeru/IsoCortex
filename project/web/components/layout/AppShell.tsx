'use client';

import React, { useState, useEffect, useCallback, ReactNode } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import { AuthProvider } from '@/lib/auth';
import { ToastProvider } from '@/components/ui/Toast';
import Sidebar from '@/components/layout/Sidebar';
import Header from '@/components/layout/Header';

const PUBLIC_PATHS = ['/login', '/setup'];

function AuthGuard({ children }: { children: ReactNode }) {
  const { isAuthenticated, loading, isFirstRun } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (loading) return;

    if (!isAuthenticated && !PUBLIC_PATHS.includes(pathname)) {
      if (isFirstRun) {
        router.replace('/setup');
      } else {
        router.replace('/login');
      }
    }
  }, [isAuthenticated, loading, isFirstRun, pathname, router]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: '#0F0A1A' }}>
        <div className="text-center">
          <div
            className="w-12 h-12 mx-auto mb-4 rounded-xl flex items-center justify-center font-bold text-white text-lg"
            style={{ backgroundColor: '#311B5B' }}
          >
            IC
          </div>
          <div
            className="w-8 h-8 mx-auto border-2 rounded-full animate-spin"
            style={{ borderColor: '#3D2A5C', borderTopColor: '#C59B47' }}
          />
        </div>
      </div>
    );
  }

  if (!isAuthenticated && !PUBLIC_PATHS.includes(pathname)) {
    return null;
  }

  return <>{children}</>;
}

function AppLayout({ children }: { children: ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const pathname = usePathname();
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated || PUBLIC_PATHS.includes(pathname)) {
    return <>{children}</>;
  }

  const pageTitle = pathname
    .split('/')
    .filter(Boolean)
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
    .join(' / ');

  return (
    <div className="min-h-screen flex" style={{ backgroundColor: '#0F0A1A' }}>
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <div className="flex-1 flex flex-col min-w-0">
        <Header onMenuClick={() => setSidebarOpen(true)} title={pageTitle} />
        <main className="flex-1 p-4 sm:p-6 lg:p-8 overflow-y-auto">{children}</main>
      </div>
    </div>
  );
}

export default function AppShell({ children }: { children: ReactNode }) {
  return (
    <AuthProvider>
      <ToastProvider>
        <AuthGuard>
          <AppLayout>{children}</AppLayout>
        </AuthGuard>
      </ToastProvider>
    </AuthProvider>
  );
}
