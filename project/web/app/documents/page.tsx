'use client';

import React, { useState, useEffect, useCallback, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { indexesApi, documentsApi, ApiError } from '@/lib/api';
import { Index, Document, PaginatedDocuments } from '@/lib/types';
import { PageSpinner, Spinner } from '@/components/ui/Spinner';
import { useToast } from '@/components/ui/Toast';
import { cn, truncate, formatDate } from '@/lib/utils';
import {
  DocumentTextIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  TrashIcon,
  XMarkIcon,
  EyeIcon,
} from '@heroicons/react/24/outline';

function DocumentsContent() {
  const searchParams = useSearchParams();
  const { addToast } = useToast();
  const preselectedIndex = searchParams.get('index') || '';

  const [indexes, setIndexes] = useState<Index[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(preselectedIndex);
  const [documents, setDocuments] = useState<PaginatedDocuments | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingIndexes, setLoadingIndexes] = useState(true);
  const [page, setPage] = useState(1);
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  useEffect(() => {
    indexesApi
      .list()
      .then(setIndexes)
      .catch(() => addToast('Failed to load indexes', 'error'))
      .finally(() => setLoadingIndexes(false));
  }, [addToast]);

  const fetchDocuments = useCallback(async () => {
    if (!selectedIndex) return;
    setLoading(true);
    try {
      const data = await documentsApi.list(selectedIndex, page, 20);
      setDocuments(data);
    } catch (err) {
      if (err instanceof ApiError) addToast(err.detail, 'error');
    } finally {
      setLoading(false);
    }
  }, [selectedIndex, page, addToast]);

  useEffect(() => {
    setPage(1);
    setDocuments(null);
  }, [selectedIndex]);

  useEffect(() => {
    if (selectedIndex) fetchDocuments();
  }, [selectedIndex, page, fetchDocuments]);

  const handleDelete = async (docId: string) => {
    if (!selectedIndex) return;
    setDeleting(docId);
    try {
      await documentsApi.delete(selectedIndex, docId);
      addToast('Document deleted', 'success');
      fetchDocuments();
    } catch (err) {
      if (err instanceof ApiError) addToast(err.detail, 'error');
    } finally {
      setDeleting(null);
    }
  };

  if (loadingIndexes) return <PageSpinner />;

  const readyIndexes = indexes.filter((i) => i.status === 'ready');

  return (
    <div className="space-y-6">
      {/* Index Selector */}
      <div>
        <label className="block text-sm font-medium text-gray-400 mb-1.5">Select Index</label>
        <select
          value={selectedIndex}
          onChange={(e) => setSelectedIndex(e.target.value)}
          className="w-full sm:w-80 px-4 py-2.5 rounded-lg text-white text-sm appearance-none"
          style={{ backgroundColor: '#1A1228', border: '1px solid #2D1F45' }}
        >
          <option value="">Choose an index...</option>
          {readyIndexes.map((idx) => (
            <option key={idx.name} value={idx.name}>
              {idx.name} ({idx.document_count} documents)
            </option>
          ))}
        </select>
      </div>

      {/* Documents Table */}
      {selectedIndex && (
        <>
          {loading ? (
            <PageSpinner />
          ) : documents && documents.documents.length > 0 ? (
            <>
              <div className="rounded-xl border overflow-hidden" style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}>
                <div className="px-5 py-3 border-b flex items-center justify-between" style={{ borderColor: '#2D1F45' }}>
                  <p className="text-sm text-gray-400">
                    {documents.total} document{documents.total !== 1 ? 's' : ''} &middot; Page {documents.page} of {documents.total_pages}
                  </p>
                </div>
                <div className="overflow-x-auto max-h-[70vh] overflow-y-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b" style={{ borderColor: '#2D1F45' }}>
                        <th className="text-left text-xs font-medium text-gray-500 px-5 py-3">Content</th>
                        <th className="text-left text-xs font-medium text-gray-500 px-5 py-3 hidden sm:table-cell">Source</th>
                        <th className="text-left text-xs font-medium text-gray-500 px-5 py-3 hidden md:table-cell">Chunk</th>
                        <th className="text-left text-xs font-medium text-gray-500 px-5 py-3 hidden lg:table-cell">Created</th>
                        <th className="text-right text-xs font-medium text-gray-500 px-5 py-3">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {documents.documents.map((doc) => (
                        <tr
                          key={doc.id}
                          className="border-b hover:bg-white/[0.02] transition-colors"
                          style={{ borderColor: '#2D1F45' }}
                        >
                          <td className="px-5 py-3">
                            <p className="text-sm text-gray-300 max-w-xs truncate">{doc.content}</p>
                            {doc.metadata && Object.keys(doc.metadata).length > 0 && (
                              <div className="flex flex-wrap gap-1 mt-1">
                                {Object.entries(doc.metadata).slice(0, 2).map(([k, v]) => (
                                  <span
                                    key={k}
                                    className="px-1.5 py-0.5 rounded text-[10px]"
                                    style={{ backgroundColor: 'rgba(49, 27, 91, 0.4)', color: '#8B7EC8' }}
                                  >
                                    {k}: {String(v).slice(0, 20)}
                                  </span>
                                ))}
                              </div>
                            )}
                          </td>
                          <td className="px-5 py-3 hidden sm:table-cell">
                            <span className="text-sm text-gray-400">{doc.source || '—'}</span>
                          </td>
                          <td className="px-5 py-3 hidden md:table-cell">
                            <span className="text-sm text-gray-400">#{doc.chunk_index}</span>
                          </td>
                          <td className="px-5 py-3 hidden lg:table-cell">
                            <span className="text-xs text-gray-500">{formatDate(doc.created_at)}</span>
                          </td>
                          <td className="px-5 py-3 text-right">
                            <div className="flex items-center justify-end gap-1">
                              <button
                                onClick={() => setSelectedDoc(doc)}
                                className="p-1.5 rounded text-gray-400 hover:text-white hover:bg-white/10"
                              >
                                <EyeIcon className="w-4 h-4" />
                              </button>
                              <button
                                onClick={() => handleDelete(doc.id)}
                                disabled={deleting === doc.id}
                                className="p-1.5 rounded text-gray-400 hover:text-red-400 hover:bg-red-500/10"
                              >
                                {deleting === doc.id ? <Spinner size="sm" /> : <TrashIcon className="w-4 h-4" />}
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Pagination */}
              {documents.total_pages > 1 && (
                <div className="flex items-center justify-center gap-2">
                  <button
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                    disabled={page <= 1}
                    className="p-2 rounded-lg border text-gray-400 hover:text-white disabled:opacity-30"
                    style={{ borderColor: '#2D1F45' }}
                  >
                    <ChevronLeftIcon className="w-4 h-4" />
                  </button>
                  {Array.from({ length: Math.min(5, documents.total_pages) }, (_, i) => {
                    let pageNum: number;
                    if (documents.total_pages <= 5) {
                      pageNum = i + 1;
                    } else if (page <= 3) {
                      pageNum = i + 1;
                    } else if (page >= documents.total_pages - 2) {
                      pageNum = documents.total_pages - 4 + i;
                    } else {
                      pageNum = page - 2 + i;
                    }
                    return (
                      <button
                        key={pageNum}
                        onClick={() => setPage(pageNum)}
                        className={cn(
                          'w-8 h-8 rounded-lg text-sm font-medium transition-colors',
                          page === pageNum
                            ? 'text-white'
                            : 'text-gray-400 hover:text-white hover:bg-white/5'
                        )}
                        style={page === pageNum ? { backgroundColor: '#311B5B' } : {}}
                      >
                        {pageNum}
                      </button>
                    );
                  })}
                  <button
                    onClick={() => setPage((p) => Math.min(documents.total_pages, p + 1))}
                    disabled={page >= documents.total_pages}
                    className="p-2 rounded-lg border text-gray-400 hover:text-white disabled:opacity-30"
                    style={{ borderColor: '#2D1F45' }}
                  >
                    <ChevronRightIcon className="w-4 h-4" />
                  </button>
                </div>
              )}
            </>
          ) : (
            <div className="rounded-xl border p-12 text-center" style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}>
              <DocumentTextIcon className="w-10 h-10 text-gray-600 mx-auto mb-3" />
              <p className="text-gray-400 text-sm">No documents found</p>
              <p className="text-gray-500 text-xs mt-1">Documents will appear here once indexed</p>
            </div>
          )}
        </>
      )}

      {!selectedIndex && (
        <div className="rounded-xl border p-16 text-center" style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}>
          <DocumentTextIcon className="w-12 h-12 text-gray-600 mx-auto mb-4" />
          <h3 className="text-gray-300 text-lg font-medium">Document Browser</h3>
          <p className="text-gray-500 text-sm mt-2">Select an index to browse its documents</p>
        </div>
      )}

      {/* Document Detail Modal */}
      {selectedDoc && (
        <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
          <div className="absolute inset-0 bg-black/60" onClick={() => setSelectedDoc(null)} />
          <div
            className="relative w-full max-w-2xl max-h-[80vh] rounded-2xl border overflow-hidden flex flex-col"
            style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}
          >
            <div className="flex items-center justify-between px-6 py-4 border-b" style={{ borderColor: '#2D1F45' }}>
              <div>
                <h3 className="text-sm font-semibold text-white">Document Details</h3>
                <p className="text-xs text-gray-500 mt-0.5">ID: {selectedDoc.id}</p>
              </div>
              <button
                onClick={() => setSelectedDoc(null)}
                className="p-1 rounded text-gray-400 hover:text-white hover:bg-white/10"
              >
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              <div>
                <h4 className="text-xs font-medium text-gray-500 uppercase mb-2">Content</h4>
                <div className="rounded-lg p-4 text-sm text-gray-300 leading-relaxed whitespace-pre-wrap" style={{ backgroundColor: '#0F0A1A' }}>
                  {selectedDoc.content}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="text-xs font-medium text-gray-500 uppercase mb-1">Source</h4>
                  <p className="text-sm text-gray-300">{selectedDoc.source || '—'}</p>
                </div>
                <div>
                  <h4 className="text-xs font-medium text-gray-500 uppercase mb-1">Chunk Index</h4>
                  <p className="text-sm text-gray-300">#{selectedDoc.chunk_index}</p>
                </div>
                <div>
                  <h4 className="text-xs font-medium text-gray-500 uppercase mb-1">Created</h4>
                  <p className="text-sm text-gray-300">{formatDate(selectedDoc.created_at)}</p>
                </div>
                <div>
                  <h4 className="text-xs font-medium text-gray-500 uppercase mb-1">Index</h4>
                  <p className="text-sm text-gray-300">{selectedDoc.index_name}</p>
                </div>
              </div>

              {selectedDoc.metadata && Object.keys(selectedDoc.metadata).length > 0 && (
                <div>
                  <h4 className="text-xs font-medium text-gray-500 uppercase mb-2">Metadata</h4>
                  <div className="rounded-lg overflow-hidden" style={{ border: '1px solid #2D1F45' }}>
                    {Object.entries(selectedDoc.metadata).map(([key, value], i) => (
                      <div
                        key={key}
                        className="flex justify-between px-4 py-2 text-sm"
                        style={{
                          backgroundColor: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.02)',
                          borderBottom: i < Object.entries(selectedDoc.metadata).length - 1 ? '1px solid #2D1F45' : 'none',
                        }}
                      >
                        <span className="text-gray-400">{key}</span>
                        <span className="text-gray-200 font-mono text-xs max-w-[200px] truncate">
                          {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function DocumentsPage() {
  return (
    <Suspense fallback={<PageSpinner />}>
      <DocumentsContent />
    </Suspense>
  );
}
