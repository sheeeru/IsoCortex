import { ArrowRight, Database, Cpu, Search, FileOutput } from 'lucide-react';

const steps = [
  {
    number: '01',
    icon: Database,
    title: 'Ingest',
    description: 'Point IsoCortex at any directory. It recursively scans, extracts text from 20+ formats, linearizes structured data (tables, slides, emails) into natural language.',
    tech: ['Scanner', 'Extractor', 'Linearizer'],
    color: 'text-iso-purple',
    borderColor: 'border-iso-purple/30',
    bgColor: 'bg-iso-purple/10',
  },
  {
    number: '02',
    icon: Cpu,
    title: 'Chunk & Embed',
    description: 'Text is split at sentence boundaries into ~120-word chunks with 25-30% overlap. Each chunk is vectorized locally into 384-dim embeddings via all-MiniLM-L6-v2 running on ONNX Runtime.',
    tech: ['Sentence-Aware Chunker', 'ONNX Embedder'],
    color: 'text-iso-gold',
    borderColor: 'border-iso-gold/30',
    bgColor: 'bg-iso-gold/10',
  },
  {
    number: '03',
    icon: Search,
    title: 'Index',
    description: 'Vectors are serialized to disk (vectors.bin + metadata.json) and loaded into an in-memory HNSW graph. Custom C++17 engine with SIMD-accelerated cosine distance.',
    tech: ['HNSW Graph', 'SIMD Cosine', 'pybind11 Bridge'],
    color: 'text-iso-purple-light',
    borderColor: 'border-iso-purple-light/30',
    bgColor: 'bg-iso-purple-light/10',
  },
  {
    number: '04',
    icon: FileOutput,
    title: 'Retrieve',
    description: 'Query is embedded locally, passed to the C++ core via zero-copy pybind11 bridge. Returns top-K results ranked by cosine similarity in sub-millisecond time.',
    tech: ['Zero-Copy Search', 'Ranked Results', 'CLI/API/Web'],
    color: 'text-iso-gold',
    borderColor: 'border-iso-gold/30',
    bgColor: 'bg-iso-gold/10',
  },
];

export function HowItWorks() {
  return (
    <section id="architecture" className="relative py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16 lg:mb-20">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-iso-gold/10 border border-iso-gold/20 mb-4">
            <div className="w-1.5 h-1.5 rounded-full bg-iso-gold" />
            <span className="text-xs font-medium text-iso-gold tracking-wide uppercase">Architecture</span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight">
            Four-Stage{' '}
            <span className="gradient-text">Retrieval Pipeline</span>
          </h2>
          <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
            A clean, modular architecture where every stage is independently testable and replaceable.
          </p>
        </div>

        <div className="relative">
          <div className="absolute left-1/2 top-0 bottom-0 w-px hidden lg:block">
            <div className="absolute inset-0 bg-gradient-to-b from-transparent via-iso-purple/30 via-iso-gold/30 to-transparent" />
          </div>

          <div className="space-y-8 lg:space-y-12">
            {steps.map((step, index) => (
              <div key={step.number} className="relative flex flex-col lg:flex-row items-center gap-6 lg:gap-12">
                {index % 2 === 0 ? (
                  <>
                    <div className="flex-1 w-full lg:text-right">
                      <StepCard step={step} align="right" />
                    </div>
                    <div className="hidden lg:flex items-center justify-center relative z-10">
                      <div className={`w-12 h-12 rounded-full ${step.bgColor} border ${step.borderColor} flex items-center justify-center`}>
                        <step.icon className={`w-5 h-5 ${step.color}`} />
                      </div>
                    </div>
                    <div className="flex-1 w-full" />
                  </>
                ) : (
                  <>
                    <div className="flex-1 w-full" />
                    <div className="hidden lg:flex items-center justify-center relative z-10">
                      <div className={`w-12 h-12 rounded-full ${step.bgColor} border ${step.borderColor} flex items-center justify-center`}>
                        <step.icon className={`w-5 h-5 ${step.color}`} />
                      </div>
                    </div>
                    <div className="flex-1 w-full lg:text-left">
                      <StepCard step={step} align="left" />
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="mt-16 glass-card rounded-2xl p-6 lg:p-8">
          <div className="flex flex-col lg:flex-row items-center gap-8">
            <div className="flex-1">
              <h3 className="text-lg font-semibold mb-2">Three-Tier Architecture</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                IsoCortex follows a clean separation of concerns across Presentation, Application, and Core Engine tiers.
                Each tier communicates through well-defined interfaces, enabling independent testing, scaling, and replacement.
              </p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 w-full lg:w-auto">
              {[
                { tier: 'Presentation', items: ['CLI', 'Next.js Web UI'] },
                { tier: 'Application', items: ['FastAPI REST Server', 'Index Manager', 'Auth & Analytics'] },
                { tier: 'Core Engine', items: ['Ingestion Pipeline', 'HNSW C++ Engine', 'Serialization Layer'] },
              ].map((tier) => (
                <div key={tier.tier} className="rounded-xl border border-border/50 p-4 bg-secondary/30">
                  <div className="text-xs font-semibold text-iso-gold uppercase tracking-wider mb-2">{tier.tier}</div>
                  <ul className="space-y-1">
                    {tier.items.map((item) => (
                      <li key={item} className="text-xs text-muted-foreground flex items-center gap-1.5">
                        <ArrowRight className="w-3 h-3 text-iso-purple/60" />
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function StepCard({
  step,
  align,
}: {
  step: (typeof steps)[0];
  align: 'left' | 'right';
}) {
  return (
    <div className="glass-card rounded-2xl p-6 hover:border-iso-purple/40 transition-colors">
      <div className={`flex items-center gap-3 mb-3 ${align === 'right' ? 'lg:flex-row-reverse lg:text-right' : ''}`}>
        <span className={`text-3xl font-bold ${step.color} opacity-30`}>{step.number}</span>
        <h3 className="text-lg font-semibold">{step.title}</h3>
      </div>
      <p className={`text-sm text-muted-foreground leading-relaxed mb-3 ${align === 'right' ? 'lg:text-right' : ''}`}>
        {step.description}
      </p>
      <div className={`flex flex-wrap gap-1.5 ${align === 'right' ? 'lg:justify-end' : ''}`}>
        {step.tech.map((t) => (
          <span key={t} className="px-2 py-0.5 text-[10px] font-medium rounded-md bg-secondary text-muted-foreground">
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}
