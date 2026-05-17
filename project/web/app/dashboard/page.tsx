'use client';

import React, { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { indexesApi, adminApi, healthApi } from '@/lib/api';
import { Index, SystemStats, HealthResponse } from '@/lib/types';
import { formatNumber, formatUptime, formatBytes, timeAgo } from '@/lib/utils';
import { PageSpinner, Spinner } from '@/components/ui/Spinner';
import {
  CircleStackIcon,
  MagnifyingGlassIcon,
  DocumentTextIcon,
  UsersIcon,
  ServerIcon,
  ArrowTrendingUpIcon,
  ClockIcon,
  PlusIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
} from '@heroicons/react/24/outline';
import { useToast } from '@/components/ui/Toast';

function StatCard({
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
    <div
      className="rounded-xl p-5 border transition-all duration-200 hover:border-opacity-60"
      style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-400">{label}</p>
          <p className="text-2xl font-bold text-white mt-1">{value}</p>
          {sub && <p className="text-xs text-gray-500 mt-1">{sub}</p>}
        </div>
        <div
          className="p-2.5 rounded-lg"
          style={{ backgroundColor: `${color}15` }}
        >
          <Icon className="w-5 h-5" style={{ color }} />
        </div>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const { addToast } = useToast();
  const [indexes, setIndexes] = useState<Index[]>([]);
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchDashboard = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [idxRes, healthRes] = await Promise.allSettled([
        indexesApi.list(),
        healthApi.check(),
      ]);

      if (idxRes.status === 'fulfilled') setIndexes(idxRes.value);
      if (healthRes.status === 'fulfilled') setHealth(healthRes.value);

      try {
        const statsRes = await adminApi.getStats();
        setStats(statsRes);
      } catch {
        // Stats might not be available for non-admin
      }
    } catch {
      setError('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDashboard();
  }, [fetchDashboard]);

  if (loading) return <PageSpinner />;

  const totalVectors = indexes.reduce((sum, i) => sum + i.vector_count, 0);
  const totalDocs = indexes.reduce((sum, i) => sum + i.document_count, 0);
  const healthy = health?.status === 'healthy';

  return (
    <div className="space-y-6">
      {/* Health Banner */}
      {health && (
        <div
          className="flex items-center gap-3 px-4 py-3 rounded-xl border"
          style={{
            backgroundColor: healthy ? 'rgba(16, 185, 129, 0.08)' : 'rgba(245, 158, 11, 0.08)',
            borderColor: healthy ? 'rgba(16, 185, 129, 0.2)' : 'rgba(245, 158, 11, 0.2)',
          }}
        >
          {healthy ? (
            <CheckCircleIcon className="w-5 h-5 text-emerald-400 flex-shrink-0" />
          ) : (
            <ExclamationTriangleIcon className="w-5 h-5 text-amber-400 flex-shrink-0" />
          )}
          <div className="flex-1">
            <p className="text-sm font-medium" style={{ color: healthy ? '#6EE7B7' : '#FCD34D' }}>
              System {healthy ? 'Healthy' : health?.status?.toUpperCase()}
            </p>
            {health.uptime && (
              <p className="text-xs text-gray-500 mt-0.5">
                Uptime: {formatUptime(health.uptime)} &middot; v{health.version}
              </p>
            )}
          </div>
          <button
            onClick={fetchDashboard}
            className="text-xs text-gray-400 hover:text-white transition-colors px-2 py-1 rounded"
            style={{ backgroundColor: 'rgba(255,255,255,0.05)' }}
          >
            Refresh
          </button>
        </div>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={CircleStackIcon}
          label="Indexes"
          value={stats?.total_indexes ?? indexes.length}
          sub="Total indexes"
          color="#C59B47"
        />
        <StatCard
          icon={DocumentTextIcon}
          label="Documents"
          value={formatNumber(stats?.total_documents ?? totalDocs)}
          sub="Across all indexes"
          color="#10B981"
        />
        <StatCard
          icon={ArrowTrendingUpIcon}
          label="Vectors"
          value={formatNumber(stats?.total_vectors ?? totalVectors)}
          sub="Embeddings stored"
          color="#8B5CF6"
        />
        <StatCard
          icon={UsersIcon}
          label="Users"
          value={stats?.total_users ?? '—'}
          sub="Registered accounts"
          color="#3B82F6"
        />
      </div>

      {/* System Info */}
      {stats && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <StatCard
            icon={MagnifyingGlassIcon}
            label="Total Searches"
            value={formatNumber(stats.total_searches)}
            sub="All time"
            color="#F59E0B"
          />
          <StatCard
            icon={ServerIcon}
            label="Memory Usage"
            value={formatBytes(stats.memory_usage_bytes)}
            sub="RSS"
            color="#EC4899"
          />
          <StatCard
            icon={ClockIcon}
            label="Uptime"
            value={formatUptime(stats.uptime_seconds)}
            sub="Since last restart"
            color="#06B6D4"
          />
        </div>
      )}

      {/* Indexes List */}
      <div
        className="rounded-xl border overflow-hidden"
        style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b" style={{ borderColor: '#2D1F45' }}>
          <h2 className="text-base font-semibold text-white">Recent Indexes</h2>
          <Link
            href="/indexes"
            className="text-sm font-medium hover:underline"
            style={{ color: '#C59B47' }}
          >
            View all
          </Link>
        </div>

        {indexes.length === 0 ? (
          <div className="px-6 py-12 text-center">
            <CircleStackIcon className="w-10 h-10 text-gray-600 mx-auto mb-3" />
            <p className="text-gray-400 text-sm">No indexes created yet</p>
            <p className="text-gray-500 text-xs mt-1">
              Get started by creating your first index
            </p>
            <Link
              href="/indexes"
              className="inline-flex items-center gap-1.5 mt-4 px-4 py-2 rounded-lg text-sm font-medium text-white transition-colors"
              style={{ backgroundColor: '#311B5B' }}
            >
              <PlusIcon className="w-4 h-4" />
              Create Index
            </Link>
          </div>
        ) : (
          <div className="divide-y" style={{ borderColor: '#2D1F45' }}>
            {indexes.slice(0, 5).map((idx) => (
              <Link
                key={idx.name}
                href={`/indexes/${encodeURIComponent(idx.name)}`}
                className="flex items-center justify-between px-6 py-3 hover:bg-white/[0.02] transition-colors"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <div
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{
                      backgroundColor:
                        idx.status === 'ready'
                          ? '#10B981'
                          : idx.status === 'building'
                          ? '#F59E0B'
                          : '#EF4444',
                    }}
                  />
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-white truncate">{idx.name}</p>
                    <p className="text-xs text-gray-500">
                      {idx.dimension}D &middot; {idx.metric} &middot; {timeAgo(idx.updated_at)}
                    </p>
                  </div>
                </div>
                <div className="text-right flex-shrink-0 ml-4">
                  <p className="text-sm text-gray-300">
                    {formatNumber(idx.document_count)} docs
                  </p>
                  <p className="text-xs text-gray-500">
                    {formatNumber(idx.vector_count)} vectors
                  </p>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
