'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { adminApi, ApiError } from '@/lib/api';
import { SystemStats, RateLimitStatus } from '@/lib/types';
import { PageSpinner } from '@/components/ui/Spinner';
import { useToast } from '@/components/ui/Toast';
import { useAuth } from '@/lib/auth';
import { formatNumber, formatBytes, formatUptime } from '@/lib/utils';
import {
  ChartBarIcon,
  CircleStackIcon,
  DocumentTextIcon,
  MagnifyingGlassIcon,
  UsersIcon,
  ServerIcon,
  ClockIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';

function MetricCard({
  icon: Icon,
  label,
  value,
  sub,
  color,
}: {
  icon: React.ComponentType<{ className?: string; style?: React.CSSProperties }>;
  label: string;
  value: string | number;
  sub?: string;
  color: string;
}) {
  return (
    <div className="rounded-xl border p-5" style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-gray-500 uppercase tracking-wider">{label}</p>
          <p className="text-2xl font-bold text-white mt-1">{value}</p>
          {sub && <p className="text-xs text-gray-500 mt-1">{sub}</p>}
        </div>
        <div className="p-2 rounded-lg" style={{ backgroundColor: `${color}12` }}>
          <Icon className="w-5 h-5" style={{ color }} />
        </div>
      </div>
    </div>
  );
}

export default function AnalyticsPage() {
  const { isAdmin } = useAuth();
  const { addToast } = useToast();
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [rateLimits, setRateLimits] = useState<RateLimitStatus | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [s, r] = await Promise.allSettled([
        adminApi.getStats(),
        adminApi.getRateLimits(),
      ]);
      if (s.status === 'fulfilled') setStats(s.value);
      if (r.status === 'fulfilled') setRateLimits(r.value);
    } catch {
      addToast('Failed to load analytics', 'error');
    } finally {
      setLoading(false);
    }
  }, [addToast]);

  useEffect(() => {
    if (isAdmin) fetchData();
  }, [isAdmin, fetchData]);

  if (!isAdmin) {
    return (
      <div className="rounded-xl border p-16 text-center" style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}>
        <ExclamationTriangleIcon className="w-10 h-10 text-amber-500 mx-auto mb-4" />
        <h3 className="text-gray-300 text-lg font-medium">Access Denied</h3>
        <p className="text-gray-500 text-sm mt-2">Admin privileges required to view analytics</p>
      </div>
    );
  }

  if (loading) return <PageSpinner />;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <p className="text-sm text-gray-400">System overview and usage metrics</p>
        <button
          onClick={fetchData}
          className="text-sm text-gray-400 hover:text-white transition-colors px-3 py-1.5 rounded-lg"
          style={{ backgroundColor: 'rgba(255,255,255,0.05)' }}
        >
          Refresh
        </button>
      </div>

      {/* Stats Grid */}
      {stats && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard icon={CircleStackIcon} label="Indexes" value={stats.total_indexes} color="#C59B47" />
            <MetricCard icon={DocumentTextIcon} label="Documents" value={formatNumber(stats.total_documents)} color="#10B981" />
            <MetricCard icon={MagnifyingGlassIcon} label="Total Searches" value={formatNumber(stats.total_searches)} color="#8B5CF6" />
            <MetricCard icon={UsersIcon} label="Users" value={stats.total_users} color="#3B82F6" />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <MetricCard
              icon={ClockIcon}
              label="Uptime"
              value={formatUptime(stats.uptime_seconds)}
              sub="Since last restart"
              color="#06B6D4"
            />
            <MetricCard
              icon={ServerIcon}
              label="Memory Usage"
              value={formatBytes(stats.memory_usage_bytes)}
              sub="Resident set size"
              color="#EC4899"
            />
            <MetricCard
              icon={ServerIcon}
              label="Disk Usage"
              value={formatBytes(stats.disk_usage_bytes)}
              sub="Storage consumed"
              color="#F59E0B"
            />
          </div>

          {/* Visual bars */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="rounded-xl border p-5" style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}>
              <h3 className="text-sm font-semibold text-white mb-4">Resource Distribution</h3>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-400">Vectors</span>
                    <span className="text-gray-300">{formatNumber(stats.total_vectors)}</span>
                  </div>
                  <div className="h-2 rounded-full overflow-hidden" style={{ backgroundColor: '#0F0A1A' }}>
                    <div className="h-full rounded-full" style={{ backgroundColor: '#8B5CF6', width: `${Math.min(100, (stats.total_vectors / Math.max(stats.total_vectors, 1)) * 100)}%` }} />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-400">Memory</span>
                    <span className="text-gray-300">{formatBytes(stats.memory_usage_bytes)}</span>
                  </div>
                  <div className="h-2 rounded-full overflow-hidden" style={{ backgroundColor: '#0F0A1A' }}>
                    <div className="h-full rounded-full" style={{ backgroundColor: '#EC4899', width: `${Math.min(100, (stats.memory_usage_bytes / (4 * 1024 * 1024 * 1024)) * 100)}%` }} />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-400">Disk</span>
                    <span className="text-gray-300">{formatBytes(stats.disk_usage_bytes)}</span>
                  </div>
                  <div className="h-2 rounded-full overflow-hidden" style={{ backgroundColor: '#0F0A1A' }}>
                    <div className="h-full rounded-full" style={{ backgroundColor: '#F59E0B', width: `${Math.min(100, (stats.disk_usage_bytes / (100 * 1024 * 1024 * 1024)) * 100)}%` }} />
                  </div>
                </div>
              </div>
            </div>

            {/* Rate Limits */}
            {rateLimits && (
              <div className="rounded-xl border p-5" style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}>
                <h3 className="text-sm font-semibold text-white mb-4">Rate Limits</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Status</span>
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${rateLimits.enabled ? 'text-emerald-400' : 'text-gray-500'}`}
                      style={{ backgroundColor: rateLimits.enabled ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255,255,255,0.05)' }}
                    >
                      {rateLimits.enabled ? 'Enabled' : 'Disabled'}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Requests/min</span>
                    <span className="text-gray-300">{rateLimits.requests_per_minute}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Burst size</span>
                    <span className="text-gray-300">{rateLimits.burst_size}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Current usage</span>
                    <span className="text-gray-300">{rateLimits.current_usage.requests} / {rateLimits.requests_per_minute}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Remaining</span>
                    <span className="text-gray-300">{rateLimits.current_usage.remaining}</span>
                  </div>
                  {rateLimits.current_usage.reset_at && (
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Resets at</span>
                      <span className="text-gray-300 text-xs">{new Date(rateLimits.current_usage.reset_at).toLocaleTimeString()}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </>
      )}

      {!stats && (
        <div className="rounded-xl border p-12 text-center" style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}>
          <ChartBarIcon className="w-10 h-10 text-gray-600 mx-auto mb-3" />
          <p className="text-gray-400 text-sm">Unable to load analytics data</p>
          <p className="text-gray-500 text-xs mt-1">Check your API connection and permissions</p>
        </div>
      )}
    </div>
  );
}
