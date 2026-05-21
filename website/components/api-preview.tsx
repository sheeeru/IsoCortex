'use client';

import { useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Copy, Check, Terminal, Globe, Cpu, Database, Layers, Search, BarChart3, Key, Shield, Container, Braces } from 'lucide-react';

const codeExamples = [
  {
    id: 'cli',
    label: 'CLI',
    icon: Terminal,
    code: `# Install IsoCortex
pip install isocortex

# Create an index from a directory
ic create work-docs ./my_documents

# Search semantically
ic search work-docs "Q3 revenue data?"

# Incremental update (only changed files)
ic update work-docs

# List all indexes
ic list

# Start REST API server
ic serve --host 0.0.0.0 --port 8000

# Validate index integrity
ic validate work-docs

# Export index as portable bundle
ic export work-docs ./backups/`,
    lang: 'bash',
  },
  {
    id: 'python',
    label: 'Python SDK',
    icon: Layers,
    code: `from isocortex import IsoCortex

# Initialize client
client = IsoCortex("http://localhost:8000")

# Create index
client.create_index(
    name="work-docs",
    directory="./my_documents"
)

# Semantic search
results = client.search(
    index="work-docs",
    query="Q3 revenue data?",
    k=5,
    filters={"file_types": ["pdf", "docx"]}
)

for r in results:
    print(f"[{r.score:.2%}] {r.source_label}")
    print(f"  {r.text[:100]}...")

# Incremental update
status = client.update_index("work-docs")
print(f"Added: {status.added}, Modified: {status.modified}, Deleted: {status.deleted}")`,
    lang: 'python',
  },
  {
    id: 'api',
    label: 'REST API',
    icon: Globe,
    code: `# Create an index
curl -X POST http://localhost:8000/api/v1/indexes \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: ic_live_a1b2c3d4..." \\
  -d '{"name": "work-docs", "directory": "./my_documents"}'

# Semantic search
curl -X POST http://localhost:8000/api/v1/indexes/work-docs/search \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "Q3 revenue data?",
    "k": 5,
    "filters": {"min_score": 0.5}
  }'

# Response:
# {
#   "query": "Q3 revenue data?",
#   "elapsed_ms": 0.34,
#   "results": [
#     {
#       "chunk_id": 42,
#       "text": "The Q3 revenue report shows...",
#       "source_file": "/docs/finance.pdf",
#       "score": 0.9234
#     }
#   ]
# }`,
    lang: 'bash',
  },
];

const endpoints = [
  { method: 'POST', path: '/api/v1/indexes', desc: 'Create index from directory' },
  { method: 'POST', path: '/api/v1/indexes/{name}/search', desc: 'Semantic search' },
  { method: 'PUT', path: '/api/v1/indexes/{name}', desc: 'Incremental update' },
  { method: 'GET', path: '/api/v1/indexes', desc: 'List all indexes' },
  { method: 'GET', path: '/api/v1/indexes/{name}', desc: 'Get index details' },
  { method: 'DELETE', path: '/api/v1/indexes/{name}', desc: 'Delete index' },
  { method: 'GET', path: '/api/v1/indexes/{name}/documents', desc: 'List documents' },
  { method: 'POST', path: '/api/v1/auth/token', desc: 'Obtain JWT token' },
  { method: 'POST', path: '/api/v1/auth/keys', desc: 'Create API key' },
  { method: 'GET', path: '/api/v1/analytics/stats', desc: 'Usage statistics' },
  { method: 'GET', path: '/api/v1/health', desc: 'Health check' },
  { method: 'GET', path: '/docs', desc: 'OpenAPI documentation' },
];

const methodColors: Record<string, string> = {
  GET: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  POST: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
  PUT: 'bg-sky-500/15 text-sky-400 border-sky-500/30',
  DELETE: 'bg-red-500/15 text-red-400 border-red-500/30',
};

export function ApiPreview() {
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('cli');

  const copyCode = (id: string, code: string) => {
    navigator.clipboard.writeText(code).catch(() => {
      // Clipboard API may fail in non-HTTPS contexts — silent fallback
    });
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <section id="api" className="relative py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-iso-purple/10 border border-iso-purple/20 mb-4">
            <div className="w-1.5 h-1.5 rounded-full bg-iso-purple" />
            <span className="text-xs font-medium text-iso-purple tracking-wide uppercase">Integration Options</span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight">
            Fits Into{' '}
            <span className="gradient-text">Your Workflow</span>
          </h2>
          <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
            Three ways to use IsoCortex: command line for quick searches, web dashboard for teams,
            REST API for system integration.
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="bg-secondary/50 border border-border/50 p-1 h-auto">
            {codeExamples.map((example) => (
              <TabsTrigger
                key={example.id}
                value={example.id}
                className="gap-2 px-4 py-2.5 text-sm data-[state=active]:bg-iso-purple/20 data-[state=active]:text-iso-gold"
              >
                <example.icon className="w-4 h-4" />
                {example.label}
              </TabsTrigger>
            ))}
          </TabsList>

          {codeExamples.map((example) => (
            <TabsContent key={example.id} value={example.id}>
              <div className="glass-card rounded-2xl overflow-hidden">
                <div className="flex items-center justify-between px-4 py-2.5 border-b border-border/50 bg-secondary/20">
                  <div className="flex items-center gap-2">
                    <div className="flex gap-1.5">
                      <div className="w-2.5 h-2.5 rounded-full bg-red-500/60" />
                      <div className="w-2.5 h-2.5 rounded-full bg-amber-500/60" />
                      <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/60" />
                    </div>
                    <span className="text-xs text-muted-foreground font-mono">
                      isocortex-{example.lang}
                    </span>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 px-2 text-xs text-muted-foreground hover:text-foreground"
                    onClick={() => copyCode(example.id, example.code)}
                  >
                    {copiedId === example.id ? (
                      <Check className="w-3.5 h-3.5 text-green-400" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                  </Button>
                </div>
                <pre className="p-4 sm:p-6 overflow-x-auto text-sm font-mono leading-relaxed text-muted-foreground max-h-[500px]">
                  <code>{example.code}</code>
                </pre>
              </div>
            </TabsContent>
          ))}
        </Tabs>

        <div className="mt-12 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="glass-card rounded-2xl p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Globe className="w-5 h-5 text-iso-gold" />
              API Endpoints
            </h3>
            <div className="space-y-2 max-h-80 overflow-y-auto pr-2">
              {endpoints.map((ep) => (
                <div
                  key={ep.method + ep.path}
                  className="flex items-center gap-2 p-2 rounded-lg hover:bg-secondary/50 transition-colors"
                >
                  <Badge
                    variant="outline"
                    className={`text-[10px] font-mono px-1.5 py-0 h-5 min-w-[50px] justify-center ${
                      methodColors[ep.method]
                    }`}
                  >
                    {ep.method}
                  </Badge>
                  <code className="text-xs font-mono text-foreground/80 flex-shrink-0">
                    {ep.path}
                  </code>
                  <span className="text-[11px] text-muted-foreground/60 truncate">
                    {ep.desc}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="glass-card rounded-2xl p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Cpu className="w-5 h-5 text-iso-gold" />
              Tech Stack
            </h3>
            <div className="grid grid-cols-2 gap-3">
              {[
                { icon: Database, label: 'Python 3.10+', desc: 'Pipeline orchestration' },
                { icon: Cpu, label: 'C++17 + CMake', desc: 'HNSW engine + SIMD' },
                { icon: Layers, label: 'pybind11', desc: 'C++ ↔ Python bridge' },
                { icon: Globe, label: 'FastAPI + Uvicorn', desc: 'REST API server' },
                { icon: Search, label: 'ONNX Runtime', desc: 'Local embeddings' },
                { icon: Shield, label: 'python-jose', desc: 'JWT authentication' },
                { icon: BarChart3, label: 'SQLite (WAL)', desc: 'Analytics storage' },
                { icon: Container, label: 'Docker + Compose', desc: 'Single-command deploy' },
                { icon: Key, label: 'slowapi', desc: 'Rate limiting' },
                { icon: Braces, label: 'Prisma ORM', desc: 'Database migrations' },
              ].map((tech) => (
                <div
                  key={tech.label}
                  className="flex items-start gap-2.5 p-2.5 rounded-lg hover:bg-secondary/50 transition-colors"
                >
                  <tech.icon className="w-4 h-4 text-iso-purple mt-0.5 flex-shrink-0" />
                  <div>
                    <div className="text-xs font-semibold">{tech.label}</div>
                    <div className="text-[10px] text-muted-foreground">{tech.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
