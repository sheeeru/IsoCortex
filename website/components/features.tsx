'use client';

import {
  FileText,
  Brain,
  Layers,
  Search,
  Cpu,
  Terminal,
  Globe,
  Shield,
  BarChart3,
  Key,
  Gauge,
  Container,
} from 'lucide-react';

const features = [
  {
    icon: FileText,
    title: 'Universal Ingestion',
    description: 'Automatically scan and process 20+ file formats across 7 categories — PDFs, Word docs, spreadsheets, presentations, source code, emails, and HTML. No preprocessing required.',
    tags: ['PDF', 'DOCX', 'XLSX', 'PPTX', '.py', '.js', 'HTML', 'EML'],
  },
  {
    icon: Brain,
    title: 'Neural Embeddings',
    description: '384-dimensional vectors via all-MiniLM-L6-v2. All inference runs locally with zero network calls. Configurable batching for optimal throughput.',
    tags: ['all-MiniLM-L6-v2', '384-dim', 'Local Inference'],
  },
  {
    icon: Layers,
    title: 'Sentence-Aware Chunking',
    description: 'Context-preserving ~120-word chunks at natural sentence boundaries with 25-30% overlap. Token guard ensures no chunk exceeds 256 tokens.',
    tags: ['Semantic Splits', '25-30% Overlap', 'Token Guard'],
  },
  {
    icon: Search,
    title: 'HNSW Graph Search',
    description: 'O(log N) approximate nearest neighbor search with >95% recall. Custom C++17 implementation with configurable M, efConstruction, and efSearch parameters.',
    tags: ['O(log N)', '>95% Recall', 'Configurable'],
  },
  {
    icon: Cpu,
    title: 'SIMD Acceleration',
    description: 'Auto-detected AVX2/SSE4.1 (x86_64) and NEON (ARM64) cosine distance. Up to 8x speedup on modern CPUs. Graceful scalar fallback.',
    tags: ['AVX2', 'NEON', '8x Faster'],
  },
  {
    icon: Terminal,
    title: 'Full CLI',
    description: '15 commands for complete control: create, search, update, list, delete, add, remove, serve, web, benchmark, validate, export, import, and auth.',
    tags: ['15 Commands', 'Pipeline Control'],
  },
  {
    icon: Globe,
    title: 'REST API Server',
    description: 'FastAPI server with 20+ endpoints for search, index management, authentication, and analytics. Auto-generated OpenAPI/Swagger documentation.',
    tags: ['FastAPI', '20+ Endpoints', 'OpenAPI'],
  },
  {
    icon: Shield,
    title: 'Zero Trust Privacy',
    description: '100% local processing. Zero outbound network requests. API binds to 127.0.0.1 by default. Compliant with GDPR, HIPAA, and SOC 2 data requirements.',
    tags: ['GDPR', 'HIPAA', 'Air-Gap Ready'],
  },
  {
    icon: BarChart3,
    title: 'Usage Analytics',
    description: 'Track search queries, index operations, and API key usage in local SQLite. Query frequency charts, popular documents, and usage trends dashboard.',
    tags: ['SQLite', 'Query Tracking', 'Charts'],
  },
  {
    icon: Key,
    title: 'Authentication & RBAC',
    description: 'API keys with SHA-256 hashing, JWT tokens with configurable expiry. Role-based access control with admin, editor, and viewer roles.',
    tags: ['JWT', 'API Keys', 'RBAC'],
  },
  {
    icon: Gauge,
    title: 'Incremental Indexing',
    description: 'SHA-256 checksum-based change detection. Process only new or modified files. Appends to existing HNSW graph without full rebuild for insertions.',
    tags: ['SHA-256', 'Delta Updates', 'Fast'],
  },
  {
    icon: Container,
    title: 'Docker Deployment',
    description: 'Single-command deployment with multi-stage Docker build. Compose orchestration for API server and web UI. Volume mounts for data persistence.',
    tags: ['Docker', 'Compose', 'One Command'],
  },
];

export function Features() {
  return (
    <section id="features" className="relative py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16 lg:mb-20">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-iso-purple/10 border border-iso-purple/20 mb-4">
            <div className="w-1.5 h-1.5 rounded-full bg-iso-purple" />
            <span className="text-xs font-medium text-iso-purple tracking-wide uppercase">Features</span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight">
            Everything You Need for{' '}
            <span className="gradient-text">Local Neural Search</span>
          </h2>
          <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
            From ingestion to retrieval, every component is engineered for performance,
            privacy, and reliability. No compromises.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {features.map((feature, index) => (
            <div
              key={feature.title}
              className="group glass-card rounded-2xl p-6 hover:border-iso-purple/40 transition-all duration-300 hover:-translate-y-0.5"
              style={{ animationDelay: `${index * 0.05}s` }}
            >
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-11 h-11 rounded-xl bg-iso-purple/10 border border-iso-purple/20 flex items-center justify-center group-hover:bg-iso-purple/20 transition-colors">
                  <feature.icon className="w-5 h-5 text-iso-gold" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-base font-semibold mb-1.5">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed mb-3">
                    {feature.description}
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {feature.tags.map((tag) => (
                      <span
                        key={tag}
                        className="px-2 py-0.5 text-[10px] font-medium rounded-md bg-secondary text-muted-foreground"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
