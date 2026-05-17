'use client';

import React, { useState, useEffect, useCallback, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { indexesApi, searchApi, ApiError } from '@/lib/api';
import { Index, SearchResponse, SearchResult } from '@/lib/types';
import { Spinner, PageSpinner } from '@/components/ui/Spinner';
import { useToast } from '@/components/ui/Toast';
import { cn, truncate } from '@/lib/utils';
import {
  MagnifyingGlassIcon,
  ChevronDownIcon,
  DocumentTextIcon,
  ClockIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline';

function SearchContent() {
  const { addToast } = useToast();
  const searchParams = useSearchParams();
  const [indexes, setIndexes] = useState<Index[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<string>(() => searchParams.get('index') || '');
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(10);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingIndexes, setLoadingIndexes] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    indexesApi
      .list()
      .then((data: any) => {
        if (Array.isArray(data)) setIndexes(data);
        else if (data && Array.isArray(data.indexes)) setIndexes(data.indexes);
        else setIndexes([]);
      })
      .catch(() => addToast('Failed to load indexes', 'error'))
      .finally(() => setLoadingIndexes(false));
  }, [addToast]);

  const handleSearch = useCallback(async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!query.trim() || !selectedIndex) {
      addToast('Please enter a query and select an index', 'warning');
      return;
    }

    setLoading(true);
    setResults(null);
    try {
      const res = await searchApi.search(selectedIndex, {
        query: query.trim(),
        top_k: topK,
        include_metadata: true,
      });
      setResults(res);
    } catch (err) {
      if (err instanceof ApiError) {
        addToast(err.detail, 'error');
      } else {
        addToast('Search failed', 'error');
      }
    } finally {
      setLoading(false);
    }
  }, [query, selectedIndex, topK, addToast]);

  if (loadingIndexes) return <PageSpinner />;

  return (
    <div className="space-y-6">
      {/* Search Form */}
      <div
        className="rounded-xl border p-6"
        style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}
      >
        <form onSubmit={handleSearch} className="space-y-4">
          {/* Query */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1.5">
              Search Query
            </label>
            <div className="relative">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter your semantic search query..."
                className="w-full pl-10 pr-4 py-3 rounded-lg text-white text-sm placeholder-gray-500"
                style={{ backgroundColor: '#0F0A1A', border: '1px solid #2D1F45' }}
              />
            </div>
          </div>

          {/* Index selector */}
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-400 mb-1.5">
                Index
              </label>
              <select
                value={selectedIndex}
                onChange={(e) => setSelectedIndex(e.target.value)}
                className="w-full px-4 py-2.5 rounded-lg text-white text-sm appearance-none cursor-pointer"
                style={{ backgroundColor: '#0F0A1A', border: '1px solid #2D1F45' }}
              >
                <option value="">Select an index...</option>
                {indexes
                  .filter((i) => i.status === 'ready')
                  .map((idx) => (
                    <option key={idx.name} value={idx.name}>
                      {idx.name} ({idx.document_count} docs, {idx.dimension}D)
                    </option>
                  ))}
              </select>
            </div>

            {/* Advanced toggle */}
            <div className="sm:w-48">
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center gap-1.5 text-sm font-medium mb-1.5"
                style={{ color: '#C59B47' }}
              >
                Advanced
                <ChevronDownIcon
                  className={cn('w-4 h-4 transition-transform', showAdvanced && 'rotate-180')}
                />
              </button>
              {showAdvanced && (
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Top K Results</label>
                  <input
                    type="number"
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                    min={1}
                    max={100}
                    className="w-full px-3 py-2 rounded-lg text-white text-sm"
                    style={{ backgroundColor: '#0F0A1A', border: '1px solid #2D1F45' }}
                  />
                </div>
              )}
            </div>
          </div>

          <button
            type="submit"
            disabled={loading || !query.trim() || !selectedIndex}
            className="px-6 py-2.5 rounded-lg text-sm font-semibold text-white transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            style={{
              backgroundColor: '#311B5B',
              boxShadow: loading ? 'none' : '0 0 20px rgba(49, 27, 91, 0.4)',
            }}
          >
            {loading ? (
              <>
                <Spinner size="sm" /> Searching...
              </>
            ) : (
              <>
                <SparklesIcon className="w-4 h-4" />
                Search
              </>
            )}
          </button>
        </form>
      </div>

      {/* Results */}
      {results && (
        <div className="space-y-4">
          {/* Results header */}
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-semibold text-white">
                Results
                <span className="ml-2 text-sm font-normal text-gray-400">
                  ({results.results.length} found in {results.latency_ms.toFixed(1)}ms)
                </span>
              </h2>
            </div>
          </div>

          {results.results.length === 0 ? (
            <div
              className="rounded-xl border p-12 text-center"
              style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}
            >
              <MagnifyingGlassIcon className="w-10 h-10 text-gray-600 mx-auto mb-3" />
              <p className="text-gray-400 text-sm">No results found</p>
              <p className="text-gray-500 text-xs mt-1">
                Try a different query or adjust your search parameters
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {results.results.map((result, i) => (
                <ResultCard key={result.id || i} result={result} rank={i + 1} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {!results && !loading && (
        <div
          className="rounded-xl border p-16 text-center"
          style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}
        >
          <SparklesIcon className="w-12 h-12 text-gray-600 mx-auto mb-4" />
          <h3 className="text-gray-300 text-lg font-medium">Semantic Search</h3>
          <p className="text-gray-500 text-sm mt-2 max-w-md mx-auto">
            Enter a natural language query and select an index to find semantically similar documents using vector embeddings.
          </p>
        </div>
      )}
    </div>
  );
}

export default function SearchPage() {
  return (
    <Suspense fallback={<PageSpinner />}>
      <SearchContent />
    </Suspense>
  );
}

function ResultCard({ result, rank }: { result: SearchResult; rank: number }) {
  const scorePercent = Math.round(result.score * 100);
  const scoreColor =
    scorePercent >= 80 ? '#10B981' : scorePercent >= 60 ? '#F59E0B' : '#6B7280';

  return (
    <div
      className="rounded-xl border p-5 transition-all duration-200 hover:border-opacity-60"
      style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}
    >
      <div className="flex items-start gap-4">
        {/* Rank badge */}
        <div
          className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold"
          style={{
            backgroundColor: rank <= 3 ? 'rgba(197, 155, 71, 0.15)' : 'rgba(255,255,255,0.05)',
            color: rank <= 3 ? '#C59B47' : '#6B7280',
          }}
        >
          #{rank}
        </div>

        <div className="flex-1 min-w-0">
          {/* Score & meta */}
          <div className="flex items-center gap-3 mb-2">
            <span
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold"
              style={{ backgroundColor: `${scoreColor}15`, color: scoreColor }}
            >
              {scorePercent}% match
            </span>
            {result.source && (
              <span className="flex items-center gap-1 text-xs text-gray-500">
                <DocumentTextIcon className="w-3 h-3" />
                {truncate(result.source, 30)}
              </span>
            )}
            {result.chunk_index != null && (
              <span className="text-xs text-gray-500">Chunk #{result.chunk_index}</span>
            )}
          </div>

          {/* Content */}
          <p className="text-sm text-gray-300 leading-relaxed line-clamp-3">
            {result.content}
          </p>

          {/* Metadata */}
          {result.metadata && Object.keys(result.metadata).length > 0 && (
            <div className="flex flex-wrap gap-1.5 mt-3">
              {Object.entries(result.metadata)
                .slice(0, 5)
                .map(([key, value]) => (
                  <span
                    key={key}
                    className="px-2 py-0.5 rounded text-xs"
                    style={{
                      backgroundColor: 'rgba(49, 27, 91, 0.4)',
                      color: '#8B7EC8',
                    }}
                  >
                    {key}: {String(value).slice(0, 40)}
                  </span>
                ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
