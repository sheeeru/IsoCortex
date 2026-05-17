'use client';

import React, { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { indexesApi, jobsApi, ApiError } from '@/lib/api';
import { Index, Job, CreateIndexRequest } from '@/lib/types';
import { PageSpinner, Spinner } from '@/components/ui/Spinner';
import { useToast } from '@/components/ui/Toast';
import { cn, formatNumber, formatBytes, timeAgo, formatDate } from '@/lib/utils';
import {
  PlusIcon,
  TrashIcon,
  XMarkIcon,
  CircleStackIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline';

export default function IndexesPage() {
  const { addToast } = useToast();
  const [indexes, setIndexes] = useState<Index[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [creating, setCreating] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  const fetchIndexes = useCallback(async () => {
    setLoading(true);
    try {
      const data = await indexesApi.list() as any;
      if (Array.isArray(data)) setIndexes(data);
      else if (data && Array.isArray(data.indexes)) setIndexes(data.indexes);
      else setIndexes([]);
    } catch {
      addToast('Failed to load indexes', 'error');
    } finally {
      setLoading(false);
    }
  }, [addToast]);

  useEffect(() => {
    fetchIndexes();
  }, [fetchIndexes]);

  const handleCreate = async (data: CreateIndexRequest) => {
    setCreating(true);
    try {
      const job = await indexesApi.create(data);
      addToast(`Index "${data.name}" creation started (Job: ${job.job_id})`, 'success');
      setShowCreate(false);
      pollJob(job.job_id);
    } catch (err) {
      if (err instanceof ApiError) addToast(err.detail, 'error');
      else addToast('Failed to create index', 'error');
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      const job = await indexesApi.delete(deleteTarget);
      addToast(`Index deletion started (Job: ${job.job_id})`, 'success');
      setDeleteTarget(null);
      pollJob(job.job_id);
    } catch (err) {
      if (err instanceof ApiError) addToast(err.detail, 'error');
      else addToast('Failed to delete index', 'error');
    } finally {
      setDeleting(false);
    }
  };

  const pollJob = async (jobId: string) => {
    const poll = async () => {
      try {
        const job = await jobsApi.get(jobId);
        if (job.status === 'completed' || job.status === 'failed') {
          fetchIndexes();
          if (job.status === 'completed') addToast('Job completed!', 'success');
          else addToast(`Job failed: ${job.error}`, 'error');
          return;
        }
        setTimeout(poll, 2000);
      } catch {
        setTimeout(poll, 2000);
      }
    };
    poll();
  };

  if (loading) return <PageSpinner />;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-400">{indexes.length} index{indexes.length !== 1 ? 'es' : ''} found</p>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-white transition-all"
          style={{ backgroundColor: '#311B5B', boxShadow: '0 0 20px rgba(49, 27, 91, 0.3)' }}
        >
          <PlusIcon className="w-4 h-4" />
          Create Index
        </button>
      </div>

      {/* Index Grid */}
      {indexes.length === 0 ? (
        <div className="rounded-xl border p-16 text-center" style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}>
          <CircleStackIcon className="w-12 h-12 text-gray-600 mx-auto mb-4" />
          <h3 className="text-gray-300 text-lg font-medium">No indexes yet</h3>
          <p className="text-gray-500 text-sm mt-2">Create your first semantic index to get started</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {indexes.map((idx: any) => (
            <Link
              key={idx.name}
              href={`/indexes/${encodeURIComponent(idx.name)}`}
              className="group rounded-xl border p-5 transition-all duration-200 hover:border-opacity-60 block"
              style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-2 min-w-0">
                  <div
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{
                      backgroundColor:
                        idx.status === 'ready' ? '#10B981' : idx.status === 'building' ? '#F59E0B' : '#EF4444',
                    }}
                  />
                  <h3 className="text-sm font-semibold text-white truncate">{idx.name}</h3>
                </div>
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={(e) => {
                      e.preventDefault();
                      setDeleteTarget(idx.name);
                    }}
                    className="p-1 rounded text-gray-500 hover:text-red-400 hover:bg-red-500/10"
                  >
                    <TrashIcon className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Status</span>
                  <span
                    className="font-medium capitalize"
                    style={{
                      color:
                        idx.status === 'ready' ? '#10B981' : idx.status === 'building' ? '#F59E0B' : '#EF4444',
                    }}
                  >
                    {idx.status}
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Dimensions</span>
                  <span className="text-gray-300">{idx.dimension || 384}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Metric</span>
                  <span className="text-gray-300">{idx.metric || 'cosine'}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Documents</span>
                  <span className="text-gray-300">{formatNumber(idx.document_count)}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Vectors</span>
                  <span className="text-gray-300">{formatNumber(idx.vector_count)}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Size</span>
                  <span className="text-gray-300">{formatBytes(idx.size_bytes)}</span>
                </div>
              </div>

              <div className="flex items-center justify-between mt-4 pt-3 border-t" style={{ borderColor: '#2D1F45' }}>
                <span className="text-xs text-gray-500">{timeAgo(idx.updated_at)}</span>
                <ChevronRightIcon className="w-4 h-4 text-gray-500 group-hover:text-gray-300 transition-colors" />
              </div>
            </Link>
          ))}
        </div>
      )}

      {/* Create Modal */}
      {showCreate && (
        <CreateIndexModal
          onSubmit={handleCreate}
          onClose={() => setShowCreate(false)}
          loading={creating}
        />
      )}

      {/* Delete Confirmation */}
      {deleteTarget && (
        <ConfirmModal
          title="Delete Index"
          message={`Are you sure you want to delete "${deleteTarget}"? This action will remove all associated vectors and cannot be undone.`}
          confirmText="Delete Index"
          loading={deleting}
          onConfirm={handleDelete}
          onCancel={() => setDeleteTarget(null)}
          danger
        />
      )}
    </div>
  );
}

function CreateIndexModal({
  onSubmit,
  onClose,
  loading,
}: {
  onSubmit: (data: CreateIndexRequest) => void;
  onClose: () => void;
  loading: boolean;
}) {
  const [name, setName] = useState('');
  const [dimension, setDimension] = useState(768);
  const [metric, setMetric] = useState<'cosine' | 'l2' | 'ip'>('cosine');
  const [efConstruction, setEfConstruction] = useState(128);
  const [M, setM] = useState(16);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({ name, dimension, metric, ef_construction: efConstruction, M });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div
        className="relative w-full max-w-md rounded-2xl border p-6"
        style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-white">Create New Index</h2>
          <button onClick={onClose} className="p-1 rounded text-gray-400 hover:text-white hover:bg-white/10">
            <XMarkIcon className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">Index Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, ''))}
              required
              placeholder="my-index"
              className="w-full px-4 py-2.5 rounded-lg text-white text-sm placeholder-gray-500"
              style={{ backgroundColor: '#0F0A1A', border: '1px solid #2D1F45' }}
            />
            <p className="text-xs text-gray-600 mt-1">Lowercase letters, numbers, hyphens, underscores</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">Dimensions</label>
            <input
              type="number"
              value={dimension}
              onChange={(e) => setDimension(Number(e.target.value))}
              required
              min={1}
              max={4096}
              className="w-full px-4 py-2.5 rounded-lg text-white text-sm"
              style={{ backgroundColor: '#0F0A1A', border: '1px solid #2D1F45' }}
            />
            <p className="text-xs text-gray-600 mt-1">Must match your embedding model output</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">Distance Metric</label>
            <select
              value={metric}
              onChange={(e) => setMetric(e.target.value as 'cosine' | 'l2' | 'ip')}
              className="w-full px-4 py-2.5 rounded-lg text-white text-sm appearance-none"
              style={{ backgroundColor: '#0F0A1A', border: '1px solid #2D1F45' }}
            >
              <option value="cosine">Cosine Similarity</option>
              <option value="l2">L2 (Euclidean)</option>
              <option value="ip">Inner Product</option>
            </select>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-1">M</label>
              <input
                type="number"
                value={M}
                onChange={(e) => setM(Number(e.target.value))}
                min={4}
                max={64}
                className="w-full px-4 py-2.5 rounded-lg text-white text-sm"
                style={{ backgroundColor: '#0F0A1A', border: '1px solid #2D1F45' }}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-1">EF Construction</label>
              <input
                type="number"
                value={efConstruction}
                onChange={(e) => setEfConstruction(Number(e.target.value))}
                min={8}
                max={512}
                className="w-full px-4 py-2.5 rounded-lg text-white text-sm"
                style={{ backgroundColor: '#0F0A1A', border: '1px solid #2D1F45' }}
              />
            </div>
          </div>

          <div className="flex gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 py-2.5 rounded-lg text-sm font-medium text-gray-300 border transition-colors"
              style={{ borderColor: '#2D1F45' }}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading || !name}
              className="flex-1 py-2.5 rounded-lg text-sm font-semibold text-white transition-all disabled:opacity-50"
              style={{ backgroundColor: '#311B5B' }}
            >
              {loading ? <Spinner size="sm" /> : 'Create Index'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function ConfirmModal({
  title,
  message,
  confirmText,
  loading,
  onConfirm,
  onCancel,
  danger,
}: {
  title: string;
  message: string;
  confirmText: string;
  loading: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  danger?: boolean;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
      <div className="absolute inset-0 bg-black/60" onClick={onCancel} />
      <div
        className="relative w-full max-w-sm rounded-2xl border p-6"
        style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}
      >
        <h2 className="text-lg font-semibold text-white mb-2">{title}</h2>
        <p className="text-sm text-gray-400 mb-6">{message}</p>
        <div className="flex gap-3">
          <button
            onClick={onCancel}
            disabled={loading}
            className="flex-1 py-2.5 rounded-lg text-sm font-medium text-gray-300 border"
            style={{ borderColor: '#2D1F45' }}
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={loading}
            className="flex-1 py-2.5 rounded-lg text-sm font-semibold text-white"
            style={{ backgroundColor: danger ? '#991B1B' : '#311B5B' }}
          >
            {loading ? <Spinner size="sm" /> : confirmText}
          </button>
        </div>
      </div>
    </div>
  );
}
