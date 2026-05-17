'use client';

import { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import {
  Search,
  FileText,
  Sparkles,
  Filter,
  Clock,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  Loader2,
} from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';

const mockDocuments = [
  {
    chunk_id: 1,
    text: 'The deployment process begins with the Docker image build. The multi-stage Dockerfile compiles the C++ HNSW engine in the builder stage, then copies only the runtime artifacts to a slim production image. Environment variables configure the API server and embedding model paths.',
    source_file: '/docs/deployment-guide.pdf',
    source_label: 'deployment-guide.pdf',
    format_category: 'pdf',
    score: 0.94,
    word_count: 52,
  },
  {
    chunk_id: 2,
    text: 'HNSW graph construction uses the Hierarchical Navigable Small World algorithm. Each node maintains connections to its nearest neighbors across multiple layers. The top layer is sparse for long-range jumps, while the bottom layer is dense for precise nearest neighbor search.',
    source_file: '/core/hnsw.hpp',
    source_label: 'hnsw.hpp',
    format_category: 'cpp',
    score: 0.91,
    word_count: 48,
  },
  {
    chunk_id: 3,
    text: 'Incremental indexing detects file changes using SHA-256 checksums. New files are fully processed (extract, chunk, embed, insert). Modified files have their old chunks removed and reprocessed. Deleted files trigger chunk removal from both the vector store and metadata.',
    source_file: '/ic/export/serializer.py',
    source_label: 'serializer.py',
    format_category: 'python',
    score: 0.87,
    word_count: 45,
  },
  {
    chunk_id: 4,
    text: 'The authentication system supports both API keys and JWT tokens. API keys are hashed with SHA-256 before storage. JWT tokens have configurable expiry (default 24h) and are signed with HS256. Three roles are supported: admin, editor, and viewer.',
    source_file: '/ic/cli/auth.py',
    source_label: 'auth.py',
    format_category: 'python',
    score: 0.82,
    word_count: 44,
  },
  {
    chunk_id: 5,
    text: 'SIMD acceleration auto-detects CPU capabilities at runtime. AVX2 processes 8 floats per instruction on x86_64. NEON provides similar acceleration on ARM64 (Apple Silicon, Raspberry Pi). A verification test ensures SIMD results match scalar within 1e-6 tolerance.',
    source_file: '/ic/core/cosine.hpp',
    source_label: 'cosine.hpp',
    format_category: 'cpp',
    score: 0.78,
    word_count: 46,
  },
  {
    chunk_id: 6,
    text: 'Rate limiting uses a sliding window counter stored in SQLite for persistence across restarts. Default is 100 requests per 60-second window per API key. Admin keys bypass rate limits. HTTP 429 responses include Retry-After header.',
    source_file: '/ic/server.py',
    source_label: 'server.py',
    format_category: 'python',
    score: 0.74,
    word_count: 41,
  },
];

const formatColors: Record<string, string> = {
  pdf: 'bg-red-500/15 text-red-400 border-red-500/30',
  python: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  cpp: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
  docx: 'bg-blue-500/15 text-blue-400 border-blue-500/30',
  xlsx: 'bg-green-500/15 text-green-400 border-green-500/30',
  html: 'bg-orange-500/15 text-orange-400 border-orange-500/30',
};

const sampleQueries = [
  'How does incremental indexing work?',
  'SIMD acceleration implementation',
  'Docker deployment process',
  'Authentication and RBAC',
  'HNSW graph construction algorithm',
];

export function Demo() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<typeof mockDocuments>([]);
  const [loading, setLoading] = useState(false);
  const [expandedChunk, setExpandedChunk] = useState<number | null>(null);
  const [minScore, setMinScore] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [totalResults, setTotalResults] = useState(0);
  const [copied, setCopied] = useState(false);
  const [activeFilter, setActiveFilter] = useState('all');

  const handleSearch = useCallback(
    (searchQuery?: string) => {
      const q = searchQuery || query;
      if (!q.trim()) return;

      setLoading(true);
      setResults([]);
      setExpandedChunk(null);

      setTimeout(() => {
        const filtered = mockDocuments.filter(
          (doc) => doc.score >= minScore && (activeFilter === 'all' || doc.format_category === activeFilter)
        );
        setResults(filtered);
        setTotalResults(mockDocuments.length);
        setElapsed(0.12 + Math.random() * 0.3);
        setLoading(false);
      }, 600 + Math.random() * 400);
    },
    [query, minScore, activeFilter]
  );

  const copyCommand = () => {
    navigator.clipboard.writeText(`ic search work-docs "${query}"`);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section id="demo" className="relative py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-iso-purple/10 border border-iso-purple/20 mb-4">
            <div className="w-1.5 h-1.5 rounded-full bg-iso-purple" />
            <span className="text-xs font-medium text-iso-purple tracking-wide uppercase">Interactive Demo</span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight">
            Try It <span className="gradient-text">Right Now</span>
          </h2>
          <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
            Search across the entire IsoCortex codebase. This is a live demo with mock data matching the exact API response format.
          </p>
        </div>

        <div className="glass-card rounded-2xl overflow-hidden">
          {/* Search Bar */}
          <div className="p-4 sm:p-6 border-b border-border/50">
            <div className="flex flex-col sm:flex-row gap-3">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <Input
                  placeholder="Ask anything about IsoCortex..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                  className="pl-10 pr-4 h-12 bg-secondary/50 border-border/50 focus:border-iso-purple/50 text-base"
                />
                <Sparkles className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-iso-gold/50" />
              </div>
              <Button
                onClick={() => handleSearch()}
                disabled={loading || !query.trim()}
                className="h-12 px-8 bg-iso-purple hover:bg-iso-purple-light text-primary-foreground font-medium"
              >
                {loading ? (
                  <Loader2 className="w-4 h-4 animate-spin mr-2" />
                ) : (
                  <Search className="w-4 h-4 mr-2" />
                )}
                Search
              </Button>
            </div>

            {/* Sample Queries */}
            <div className="mt-3 flex flex-wrap gap-2">
              {sampleQueries.map((sq) => (
                <button
                  key={sq}
                  onClick={() => {
                    setQuery(sq);
                    handleSearch(sq);
                  }}
                  className="px-3 py-1.5 text-xs text-muted-foreground bg-secondary/50 rounded-lg hover:bg-secondary hover:text-foreground transition-colors border border-transparent hover:border-border/50"
                >
                  {sq}
                </button>
              ))}
            </div>
          </div>

          {/* Filters */}
          <div className="px-4 sm:px-6 py-3 border-b border-border/50 bg-secondary/20 flex flex-col sm:flex-row items-start sm:items-center gap-3">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Filter className="w-4 h-4" />
              <span>Filters:</span>
            </div>
            <Select value={activeFilter} onValueChange={setActiveFilter}>
              <SelectTrigger className="w-36 h-8 text-xs bg-secondary/50 border-border/50">
                <SelectValue placeholder="File type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Formats</SelectItem>
                <SelectItem value="pdf">PDF</SelectItem>
                <SelectItem value="python">Python</SelectItem>
                <SelectItem value="cpp">C++</SelectItem>
              </SelectContent>
            </Select>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span>Min Score:</span>
              <Slider
                value={[minScore]}
                onValueChange={(v) => setMinScore(v[0])}
                min={0}
                max={1}
                step={0.05}
                className="w-24"
              />
              <span className="font-mono w-8">{minScore.toFixed(2)}</span>
            </div>
          </div>

          {/* Results */}
          <div className="p-4 sm:p-6 min-h-[400px]">
            {loading && (
              <div className="flex flex-col items-center justify-center py-20 gap-4">
                <div className="relative">
                  <Loader2 className="w-8 h-8 text-iso-purple animate-spin" />
                  <Sparkles className="w-4 h-4 text-iso-gold absolute -top-1 -right-1 animate-pulse" />
                </div>
                <p className="text-sm text-muted-foreground">Embedding query & searching HNSW graph...</p>
              </div>
            )}

            {!loading && results.length === 0 && (
              <div className="flex flex-col items-center justify-center py-20 gap-4 text-center">
                <div className="w-16 h-16 rounded-2xl bg-secondary/50 flex items-center justify-center">
                  <Search className="w-7 h-7 text-muted-foreground/50" />
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">
                    Search the IsoCortex codebase
                  </p>
                  <p className="text-xs text-muted-foreground/60 mt-1">
                    Try one of the sample queries above
                  </p>
                </div>
              </div>
            )}

            {!loading && results.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs text-muted-foreground">
                    {results.length} of {totalResults} results &middot;{' '}
                    <span className="font-mono">{elapsed.toFixed(2)}ms</span>
                  </p>
                  <button
                    onClick={copyCommand}
                    className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {copied ? <Check className="w-3 h-3 text-green-400" /> : <Copy className="w-3 h-3" />}
                    {copied ? 'Copied!' : 'Copy CLI command'}
                  </button>
                </div>

                {results.map((result) => (
                  <div
                    key={result.chunk_id}
                    className="group rounded-xl border border-border/50 bg-secondary/20 hover:bg-secondary/40 hover:border-iso-purple/30 transition-all cursor-pointer p-4"
                    onClick={() =>
                      setExpandedChunk(expandedChunk === result.chunk_id ? null : result.chunk_id)
                    }
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm leading-relaxed text-foreground/90 line-clamp-2">
                          {result.text}
                        </p>
                        {expandedChunk === result.chunk_id && (
                          <p className="mt-2 text-sm leading-relaxed text-foreground/70">
                            {result.text}
                          </p>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-3 mt-3 flex-wrap">
                      <Badge
                        variant="outline"
                        className={`text-[10px] font-mono px-2 py-0 h-5 ${
                          formatColors[result.format_category] || ''
                        }`}
                      >
                        {result.format_category}
                      </Badge>
                      <div className="flex items-center gap-1 text-xs text-muted-foreground">
                        <FileText className="w-3 h-3" />
                        <span className="truncate max-w-[200px] sm:max-w-[300px]">{result.source_label}</span>
                      </div>
                      <div className="flex items-center gap-1 text-xs text-muted-foreground">
                        <Clock className="w-3 h-3" />
                        <span>{result.word_count} words</span>
                      </div>
                      <div className="ml-auto flex items-center gap-2">
                        <div className="text-xs font-medium">
                          <span className="text-iso-gold">{(result.score * 100).toFixed(1)}%</span>
                          <span className="text-muted-foreground"> match</span>
                        </div>
                        {expandedChunk === result.chunk_id ? (
                          <ChevronUp className="w-4 h-4 text-muted-foreground" />
                        ) : (
                          <ChevronDown className="w-4 h-4 text-muted-foreground" />
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* API Response Preview */}
          {!loading && results.length > 0 && (
            <div className="border-t border-border/50 bg-secondary/10 px-4 sm:px-6 py-3">
              <div className="flex items-center gap-2 mb-2">
                <Badge variant="outline" className="text-[10px] font-mono h-5 px-2">
                  API Response
                </Badge>
                <Badge variant="outline" className="text-[10px] font-mono h-5 px-2 text-green-400 border-green-500/30">
                  200 OK
                </Badge>
              </div>
              <pre className="text-[11px] font-mono text-muted-foreground overflow-x-auto max-h-32">
{`POST /api/v1/indexes/default/search`}
{`{`}
{`  "query": "${query}",`}
{`  "index": "default",`}
{`  "k": ${results.length},`}
{`  "elapsed_ms": ${(elapsed * 1000).toFixed(0)},`}
{`  "results": [${results.slice(0, 2).map(r => `\n    { "chunk_id": ${r.chunk_id}, "score": ${r.score} }`).join(',')}${results.length > 2 ? ', ...' : ''}\n  ]`}
{`}`}
              </pre>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
