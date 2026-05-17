'use client';

import React, { useState } from 'react';
import { useAuth } from '@/lib/auth';
import { useToast } from '@/components/ui/Toast';
import { ApiError } from '@/lib/api';
import { healthApi } from '@/lib/api';
import { HealthResponse } from '@/lib/types';
import {
  Cog6ToothIcon,
  ServerIcon,
  KeyIcon,
  InformationCircleIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  XCircleIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';

export default function SettingsPage() {
  const { user, refreshUser } = useAuth();
  const { addToast } = useToast();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [checkingHealth, setCheckingHealth] = useState(false);
  const [apiUrl] = useState(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8900');

  const checkHealth = async () => {
    setCheckingHealth(true);
    try {
      const res = await healthApi.check();
      setHealth(res);
    } catch {
      setHealth(null);
      addToast('Failed to connect to API', 'error');
    } finally {
      setCheckingHealth(false);
    }
  };

  return (
    <div className="space-y-6 max-w-3xl">
      {/* API Connection */}
      <div className="rounded-xl border p-5" style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}>
        <div className="flex items-center gap-2 mb-4">
          <ServerIcon className="w-5 h-5 text-gray-400" />
          <h3 className="text-sm font-semibold text-white">API Connection</h3>
        </div>

        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-400">API Base URL</span>
            <code className="text-sm px-2 py-1 rounded" style={{ backgroundColor: '#0F0A1A', color: '#C59B47' }}>
              {apiUrl}
            </code>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-400">Health Status</span>
            {health ? (
              <span className="flex items-center gap-1.5 text-sm">
                {health.status === 'healthy' ? (
                  <CheckCircleIcon className="w-4 h-4 text-emerald-400" />
                ) : health.status === 'degraded' ? (
                  <ExclamationTriangleIcon className="w-4 h-4 text-amber-400" />
                ) : (
                  <XCircleIcon className="w-4 h-4 text-red-400" />
                )}
                <span style={{ color: health.status === 'healthy' ? '#10B981' : health.status === 'degraded' ? '#F59E0B' : '#EF4444' }}>
                  {health.status.toUpperCase()}
                </span>
              </span>
            ) : (
              <span className="text-sm text-gray-500">Not checked</span>
            )}
          </div>

          {health && (
            <>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-400">Version</span>
                <span className="text-sm text-gray-300">v{health.version}</span>
              </div>
              <div className="pt-2 border-t" style={{ borderColor: '#2D1F45' }}>
                <p className="text-xs text-gray-500 mb-2">Component Status</p>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                  {Object.entries(health.components).map(([name, status]) => (
                    <div key={name} className="flex items-center gap-2 px-3 py-2 rounded-lg" style={{ backgroundColor: '#0F0A1A' }}>
                      <div
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: status === 'healthy' ? '#10B981' : status === 'degraded' ? '#F59E0B' : '#EF4444' }}
                      />
                      <span className="text-xs text-gray-300 capitalize">{name}</span>
                      <span className="text-xs ml-auto" style={{ color: status === 'healthy' ? '#10B981' : '#EF4444' }}>
                        {status}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          <button
            onClick={checkHealth}
            disabled={checkingHealth}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-white transition-all"
            style={{ backgroundColor: '#311B5B' }}
          >
            <ArrowPathIcon className={`w-4 h-4 ${checkingHealth ? 'animate-spin' : ''}`} />
            Check Connection
          </button>
        </div>
      </div>

      {/* User Profile */}
      <div className="rounded-xl border p-5" style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}>
        <div className="flex items-center gap-2 mb-4">
          <KeyIcon className="w-5 h-5 text-gray-400" />
          <h3 className="text-sm font-semibold text-white">User Profile</h3>
        </div>

        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-400">Username</span>
            <span className="text-sm text-gray-200 font-medium">{user?.username || '—'}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-400">Email</span>
            <span className="text-sm text-gray-300">{user?.email || '—'}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-400">Role</span>
            <span
              className="px-2 py-0.5 rounded-full text-xs font-semibold capitalize"
              style={{
                backgroundColor: user?.role === 'admin' ? 'rgba(197, 155, 71, 0.15)' : 'rgba(49, 27, 91, 0.4)',
                color: user?.role === 'admin' ? '#C59B47' : '#8B7EC8',
              }}
            >
              {user?.role || '—'}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-400">User ID</span>
            <code className="text-xs px-2 py-1 rounded" style={{ backgroundColor: '#0F0A1A', color: '#6B7280' }}>
              {user?.id || '—'}
            </code>
          </div>
        </div>
      </div>

      {/* About */}
      <div className="rounded-xl border p-5" style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}>
        <div className="flex items-center gap-2 mb-4">
          <InformationCircleIcon className="w-5 h-5 text-gray-400" />
          <h3 className="text-sm font-semibold text-white">About IsoCortex</h3>
        </div>
        <div className="space-y-2">
          <p className="text-sm text-gray-400 leading-relaxed">
            IsoCortex is a high-performance semantic search engine built with HNSW indexing
            and vector embeddings. It provides sub-millisecond approximate nearest neighbor
            search with support for cosine, L2, and inner product distance metrics.
          </p>
          <div className="pt-3 border-t space-y-2" style={{ borderColor: '#2D1F45' }}>
            <div className="flex justify-between text-sm">
              <span className="text-gray-500">Web UI Version</span>
              <span className="text-gray-300">1.0.0</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-500">Environment</span>
              <span className="text-gray-300">
                <code className="px-1.5 py-0.5 rounded text-xs" style={{ backgroundColor: '#0F0A1A' }}>
                  {process.env.NODE_ENV || 'production'}
                </code>
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
