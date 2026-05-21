'use client';

import {
  Shield,
  Zap,
  PackageOpen,
  Cpu,
  WifiOff,
  Codepen,
} from 'lucide-react';

const differentiators = [
  {
    icon: Shield,
    title: '100% Local Processing',
    description:
      'Zero cloud dependency. Every embedding, every query, every index operation runs entirely on your hardware. No telemetry, no phone-home, no exceptions.',
    contrast:
      'Unlike Pinecone and Weaviate, your data never touches any external server — making it ideal for GDPR, HIPAA, and SOC 2 compliance.',
  },
  {
    icon: Zap,
    title: 'Sub-Millisecond Search',
    description:
      'Custom C++17 HNSW engine with SIMD-accelerated cosine distance. Achieves >95% recall with O(log N) complexity on million-document indexes.',
    contrast:
      'Unlike Elasticsearch, which requires heavy cluster infrastructure for semantic search, IsoCortex delivers 35× faster queries on a single machine.',
  },
  {
    icon: PackageOpen,
    title: 'Zero Config Deployment',
    description:
      'Single Docker command to a fully running API server with web UI. No cluster nodes to coordinate, no Java runtime wrestling, no API keys to manage.',
    contrast:
      'Unlike Elasticsearch, which demands cluster planning and JVM tuning, IsoCortex deploys with one command and auto-configures optimal defaults.',
  },
  {
    icon: Cpu,
    title: 'SIMD Accelerated',
    description:
      'Auto-detects AVX2/AVX-512 on x86_64 and NEON on ARM64 at runtime. Up to 8× faster than pure Python implementations with zero code changes required.',
    contrast:
      'Unlike Weaviate and Pinecone, which depend on GPU instances for competitive latency, IsoCortex maximizes CPU performance — no expensive GPU required.',
  },
  {
    icon: WifiOff,
    title: 'Air-Gap Ready',
    description:
      'Works completely offline after the initial model download. No license checks, no update pings, no external dependencies at runtime.',
    contrast:
      'Unlike every cloud vector database, IsoCortex is designed from the ground up for classified environments, SCIFs, and air-gapped networks.',
  },
  {
    icon: Codepen,
    title: 'Open Source Core',
    description:
      'MIT-licensed search engine with full transparency into how your data is indexed, chunked, and queried. Read the code, modify it, trust it.',
    contrast:
      'Unlike Pinecone\'s proprietary black box, IsoCortex\'s core engine is open source. No vendor lock-in, no surprise API deprecations.',
  },
];

export function Comparison() {
  return (
    <section id="comparison" className="relative py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16 lg:mb-20">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-iso-purple/10 border border-iso-purple/20 mb-4">
            <div className="w-1.5 h-1.5 rounded-full bg-iso-purple" />
            <span className="text-xs font-medium text-iso-purple tracking-wide uppercase">
              Why IsoCortex
            </span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight">
            Built Different,{' '}
            <span className="gradient-text">By Design</span>
          </h2>
          <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
            Every architectural decision prioritizes your privacy, performance,
            and team productivity. Here&apos;s what sets IsoCortex apart.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {differentiators.map((item, index) => (
            <div
              key={item.title}
              className="group glass-card rounded-2xl p-6 hover:border-iso-purple/40 transition-all duration-300 hover:-translate-y-0.5"
              style={{ animationDelay: `${index * 0.05}s` }}
            >
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-11 h-11 rounded-xl bg-iso-purple/10 border border-iso-purple/20 flex items-center justify-center group-hover:bg-iso-purple/20 transition-colors">
                  <item.icon className="w-5 h-5 text-iso-gold" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-base font-semibold mb-2">
                    {item.title}
                  </h3>
                  <p className="text-sm text-muted-foreground leading-relaxed mb-3">
                    {item.description}
                  </p>
                  <p className="text-xs text-iso-purple/80 leading-relaxed bg-iso-purple/5 border border-iso-purple/10 rounded-lg px-3 py-2">
                    {item.contrast}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
