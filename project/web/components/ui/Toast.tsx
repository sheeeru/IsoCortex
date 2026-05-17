'use client';

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';

interface Toast {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info' | 'warning';
}

interface ToastContextType {
  toasts: Toast[];
  addToast: (message: string, type?: Toast['type']) => void;
  removeToast: (id: string) => void;
}

const ToastContext = createContext<ToastContextType>({
  toasts: [],
  addToast: () => {},
  removeToast: () => {},
});

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = useCallback((message: string, type: Toast['type'] = 'info') => {
    const id = Math.random().toString(36).slice(2);
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ toasts, addToast, removeToast }}>
      {children}
      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </ToastContext.Provider>
  );
}

function ToastContainer({ toasts, removeToast }: { toasts: Toast[]; removeToast: (id: string) => void }) {
  if (toasts.length === 0) return null;

  const colors = {
    success: { bg: 'rgba(16, 185, 129, 0.15)', border: '#10B981', text: '#6EE7B7', icon: '✓' },
    error: { bg: 'rgba(239, 68, 68, 0.15)', border: '#EF4444', text: '#FCA5A5', icon: '✕' },
    warning: { bg: 'rgba(245, 158, 11, 0.15)', border: '#F59E0B', text: '#FCD34D', icon: '⚠' },
    info: { bg: 'rgba(99, 102, 241, 0.15)', border: '#6366F1', text: '#A5B4FC', icon: 'ℹ' },
  };

  return (
    <div className="fixed top-4 right-4 z-[100] flex flex-col gap-2 max-w-sm">
      {toasts.map((toast) => {
        const c = colors[toast.type];
        return (
          <div
            key={toast.id}
            className="toast-enter flex items-start gap-3 px-4 py-3 rounded-lg border shadow-lg cursor-pointer"
            style={{
              backgroundColor: c.bg,
              borderColor: c.border,
              color: c.text,
            }}
            onClick={() => removeToast(toast.id)}
          >
            <span className="text-sm font-bold mt-0.5">{c.icon}</span>
            <p className="text-sm flex-1">{toast.message}</p>
          </div>
        );
      })}
    </div>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used within ToastProvider');
  return ctx;
}
