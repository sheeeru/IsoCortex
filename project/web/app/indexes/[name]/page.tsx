'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { indexesApi, ApiError } from '@/lib/api';
import { Index } from '@/lib/types';
import { PageSpinner, Spinner } from '@/components/ui/Spinner';
import { useToast } from '@/components/ui/Toast';
import { formatNumber, formatBytes, formatDate } from '@/lib/utils';
import Link from 'next/link';
import {
  ArrowLeftIcon,
  TrashIcon,
  ArrowDownTrayIcon,
  MagnifyingGlassIcon,
  DocumentTextIcon,
  Cog6ToothIcon,
} from '@heroicons/react/24/outline';

export default function IndexDetailPage() {
  const params = useParams();
  const router = useRouter();
  const { addToast } = useToast();
  const indexName = decodeURIComponent(params.name as string);

  const [index, setIndex] = useState<Index | null>(null);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [efSearch, setEfSearch] = useState('');
  const [saving, setSaving] = useState(false);

  const fetchIndex = useCallback(async () => {
    setLoading(true);
    try {
      const data = await indexesApi.get(indexName);
      setIndex(data);
      setEfSearch(String(data.config.ef_search || 40));
    } catch {
      addToast('Failed to load index', 'error');
    } finally {
      setLoading(false);
    }
  }, [indexName, addToast]);

  useEffect(() => {
    fetchIndex();
  }, [fetchIndex]);

  const handleDelete = async () => {
    setDeleting(true);
    try {
      await indexesApi.delete(indexName);
      addToast('Index deletion started', 'success');
      router.push('/indexes');
    } catch (err) {
      if (err instanceof ApiError) addToast(err.detail, 'error');
    } finally {
      setDeleting(false);
    }
  };

  const handleExport = async () => {
    setExporting(true);
    try {
      const job = await indexesApi.exportIndex(indexName);
      addToast(`Export started (Job: ${job.job_id})`, 'success');
    } catch (err) {
      if (err instanceof ApiError) addToast(err.detail, 'error');
    } finally {
      setExporting(false);
    }
  };

  const handleSaveConfig = async () => {
    if (!index) return;
    setSaving(true);
    try {
      const updated = await indexesApi.update(indexName, {
        ef_search: Number(efSearch),
      });
      setIndex(updated);
      addToast('Index configuration updated', 'success');
    } catch (err) {
      if (err instanceof ApiError) addToast(err.detail, 'error');
    } finally {
      setSaving(false);
    }
  };

  if (loading) return <PageSpinner />;

  if (!index) {
    return (
      <div className="text-center py-20">
        <p className="text-gray-400">Index not found</p>
        <Link href="/indexes" className="text-sm mt-2 inline-block" style={{ color: '#C59B47' }}>
          Back to Indexes
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Back */}
      <Link
        href="/indexes"
        className="inline-flex items-center gap-1.5 text-sm text-gray-400 hover:text-white transition-colors"
      >
        <ArrowLeftIcon className="w-4 h-4" />
        Back to Indexes
      </Link>

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <div
              className="w-3 h-3 rounded-full"
              style={{
                backgroundColor:
                  index.status === 'ready' ? '#10B981' : index.status === 'building' ? '#F59E0B' : '#EF4444',
              }}
            />
            <h1 className="text-2xl font-bold text-white">{index.name}</h1>
          </div>
          <p className="text-sm text-gray-500 mt-1">Created {formatDate(index.created_at)}</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleExport}
            disabled={exporting}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium text-gray-300 border transition-colors hover:bg-white/5"
            style={{ borderColor: '#2D1F45' }}
          >
            {exporting ? <Spinner size="sm" /> : <ArrowDownTrayIcon className="w-4 h-4" />}
            Export
          </button>
          <button
            onClick={handleDelete}
            disabled={deleting}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium text-red-400 border border-red-900/30 transition-colors hover:bg-red-500/10"
          >
            {deleting ? <Spinner size="sm" /> : <TrashIcon className="w-4 h-4" />}
            Delete
          </button>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {[
          { label: 'Status', value: index.status.toUpperCase(), color: index.status === 'ready' ? '#10B981' : '#F59E0B' },
          { label: 'Dimensions', value: index.dimension, color: '#C59B47' },
          { label: 'Metric', value: index.metric, color: '#8B5CF6' },
          { label: 'Documents', value: formatNumber(index.document_count), color: '#10B981' },
          { label: 'Vectors', value: formatNumber(index.vector_count), color: '#3B82F6' },
          { label: 'Size', value: formatBytes(index.size_bytes), color: '#F59E0B' },
        ].map((stat) => (
          <div key={stat.label} className="rounded-xl border p-4" style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}>
            <p className="text-xs text-gray-500">{stat.label}</p>
            <p className="text-lg font-bold mt-1" style={{ color: stat.color }}>{stat.value}</p>
          </div>
        ))}
      </div>

      {/* Configuration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-xl border p-5" style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}>
          <div className="flex items-center gap-2 mb-4">
            <Cog6ToothIcon className="w-4 h-4 text-gray-400" />
            <h3 className="text-sm font-semibold text-white">HNSW Configuration</h3>
          </div>
          <div className="space-y-3">
            {[
              { label: 'M (connections)', value: index.config.M },
              { label: 'EF Construction', value: index.config.ef_construction },
              { label: 'EF Search', value: index.config.ef_search },
              { label: 'Distance Metric', value: index.config.metric },
            ].map((item) => (
              <div key={item.label} className="flex justify-between text-sm">
                <span className="text-gray-400">{item.label}</span>
                <span className="text-gray-200 font-mono">{item.value ?? '—'}</span>
              </div>
            ))}
          </div>
          <div className="mt-4 pt-4 border-t" style={{ borderColor: '#2D1F45' }}>
            <label className="block text-xs text-gray-400 mb-1">Update EF Search</label>
            <div className="flex gap-2">
              <input
                type="number"
                value={efSearch}
                onChange={(e) => setEfSearch(e.target.value)}
                min={1}
                max={2048}
                className="flex-1 px-3 py-2 rounded-lg text-white text-sm"
                style={{ backgroundColor: '#0F0A1A', border: '1px solid #2D1F45' }}
              />
              <button
                onClick={handleSaveConfig}
                disabled={saving}
                className="px-4 py-2 rounded-lg text-sm font-medium text-white"
                style={{ backgroundColor: '#311B5B' }}
              >
                {saving ? <Spinner size="sm" /> : 'Save'}
              </button>
            </div>
          </div>
        </div>

        {/* Quick actions */}
        <div className="rounded-xl border p-5" style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}>
          <h3 className="text-sm font-semibold text-white mb-4">Quick Actions</h3>
          <div className="space-y-3">
            <Link
              href={`/search?index=${encodeURIComponent(index.name)}`}
              className="flex items-center gap-3 p-3 rounded-lg transition-colors hover:bg-white/5"
              style={{ border: '1px solid #2D1F45' }}
            >
              <MagnifyingGlassIcon className="w-5 h-5" style={{ color: '#C59B47' }} />
              <div>
                <p className="text-sm font-medium text-white">Search this Index</p>
                <p className="text-xs text-gray-500">Run a semantic search query</p>
              </div>
            </Link>
            <Link
              href={`/documents?index=${encodeURIComponent(index.name)}`}
              className="flex items-center gap-3 p-3 rounded-lg transition-colors hover:bg-white/5"
              style={{ border: '1px solid #2D1F45' }}
            >
              <DocumentTextIcon className="w-5 h-5" style={{ color: '#10B981' }} />
              <div>
                <p className="text-sm font-medium text-white">Browse Documents</p>
                <p className="text-xs text-gray-500">View and manage indexed documents</p>
              </div>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
