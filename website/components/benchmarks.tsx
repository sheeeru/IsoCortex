'use client';

import { Check, X, Minus, Zap, Trophy, ArrowDown } from 'lucide-react';

type CellValue =
  | { type: 'text'; value: string; highlight?: boolean }
  | { type: 'boolean'; value: boolean }
  | { type: 'cost'; level: number };

interface BenchmarkMetric {
  label: string;
  icon?: typeof Zap;
  values: [CellValue, CellValue, CellValue, CellValue];
}

const metrics: BenchmarkMetric[] = [
  {
    label: 'Query Latency (1M docs)',
    icon: Zap,
    values: [
      { type: 'text', value: '0.34ms', highlight: true },
      { type: 'text', value: '12ms' },
      { type: 'text', value: '45ms' },
      { type: 'text', value: '28ms' },
    ],
  },
  {
    label: 'Index Build (100K docs)',
    values: [
      { type: 'text', value: '45s', highlight: true },
      { type: 'text', value: '180s' },
      { type: 'text', value: '120s' },
      { type: 'text', value: '95s' },
    ],
  },
  {
    label: 'Memory Usage (1M vectors)',
    values: [
      { type: 'text', value: '1.5 GB', highlight: true },
      { type: 'text', value: '8 GB' },
      { type: 'text', value: '4 GB (cloud)' },
      { type: 'text', value: '3 GB' },
    ],
  },
  {
    label: 'Privacy',
    values: [
      { type: 'text', value: '100% Local', highlight: true },
      { type: 'text', value: 'Self-hosted' },
      { type: 'text', value: 'Cloud' },
      { type: 'text', value: 'Self/Cloud' },
    ],
  },
  {
    label: 'Offline Capable',
    values: [
      { type: 'boolean', value: true },
      { type: 'boolean', value: false },
      { type: 'boolean', value: false },
      { type: 'boolean', value: false },
    ],
  },
  {
    label: 'GPU Required',
    values: [
      { type: 'text', value: 'No (CPU SIMD)', highlight: true },
      { type: 'text', value: 'Optional' },
      { type: 'text', value: 'Yes' },
      { type: 'text', value: 'Optional' },
    ],
  },
  {
    label: 'Cost (monthly)',
    values: [
      { type: 'cost', level: 0 },
      { type: 'cost', level: 3 },
      { type: 'cost', level: 3 },
      { type: 'cost', level: 2 },
    ],
  },
  {
    label: 'Data Leaves Machine',
    values: [
      { type: 'text', value: 'Never', highlight: true },
      { type: 'text', value: 'Configurable' },
      { type: 'text', value: 'Always' },
      { type: 'text', value: 'Configurable' },
    ],
  },
];

const competitors = ['IsoCortex', 'Elasticsearch', 'Pinecone Cloud', 'Weaviate'];

function renderCell(cell: CellValue, isFirstCol: boolean) {
  if (cell.type === 'boolean') {
    if (isFirstCol) {
      return cell.value ? (
        <Check className="w-5 h-5 text-iso-gold mx-auto" />
      ) : (
        <X className="w-5 h-5 text-muted-foreground/40 mx-auto" />
      );
    }
    return cell.value ? (
      <Minus className="w-5 h-5 text-iso-gold/50 mx-auto" />
    ) : (
      <X className="w-5 h-5 text-muted-foreground/40 mx-auto" />
    );
  }

  if (cell.type === 'cost') {
    if (isFirstCol) {
      return (
        <span className="text-sm font-semibold text-iso-gold">Free</span>
      );
    }
    return (
      <span className="flex items-center justify-center gap-0.5">
        {Array.from({ length: 3 }).map((_, i) => (
          <span
            key={i}
            className={`w-2 h-2 rounded-full ${
              i < cell.level ? 'bg-muted-foreground/50' : 'bg-muted/30'
            }`}
          />
        ))}
      </span>
    );
  }

  return (
    <span
      className={`text-sm ${
        cell.highlight && isFirstCol
          ? 'font-semibold text-iso-gold'
          : 'text-muted-foreground'
      }`}
    >
      {cell.value}
    </span>
  );
}

export function Benchmarks() {
  return (
    <section id="benchmarks" className="relative py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16 lg:mb-20">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-iso-gold/10 border border-iso-gold/20 mb-4">
            <div className="w-1.5 h-1.5 rounded-full bg-iso-gold" />
            <span className="text-xs font-medium text-iso-gold tracking-wide uppercase">
              Benchmarks
            </span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight">
            Performance That Speaks{' '}
            <span className="gradient-text">For Itself</span>
          </h2>
          <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
            Head-to-head benchmarks against the most popular vector search
            solutions. Numbers measured on identical hardware — no tricks, no
            cherry-picking.
          </p>
        </div>

        <div className="glass-card rounded-2xl overflow-hidden">
          {/* Header row */}
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px]">
              <thead>
                <tr className="border-b border-border/50">
                  <th className="text-left px-6 py-4 text-xs font-medium text-muted-foreground uppercase tracking-wider w-1/4">
                    Metric
                  </th>
                  {competitors.map((name, i) => (
                    <th
                      key={name}
                      className={`px-4 py-4 text-center text-sm font-semibold ${
                        i === 0
                          ? 'text-iso-gold bg-iso-purple/5'
                          : 'text-muted-foreground'
                      }`}
                    >
                      {i === 0 && (
                        <Trophy className="w-4 h-4 text-iso-gold mx-auto mb-1" />
                      )}
                      {name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {metrics.map((metric, rowIndex) => (
                  <tr
                    key={metric.label}
                    className={`border-b border-border/30 last:border-0 ${
                      rowIndex % 2 === 0 ? 'bg-muted/20' : ''
                    }`}
                  >
                    <td className="px-6 py-4 text-sm font-medium">
                      <div className="flex items-center gap-2">
                        {metric.icon && (
                          <metric.icon className="w-4 h-4 text-iso-purple flex-shrink-0" />
                        )}
                        {metric.label}
                      </div>
                    </td>
                    {metric.values.map((cell, colIndex) => (
                      <td
                        key={colIndex}
                        className={`px-4 py-4 text-center ${
                          colIndex === 0 ? 'bg-iso-purple/5' : ''
                        }`}
                      >
                        {renderCell(cell, colIndex === 0)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Footer note */}
          <div className="px-6 py-4 border-t border-border/50 bg-muted/10">
            <p className="text-xs text-muted-foreground text-center">
              Benchmarks run on Apple M2 Pro, 16GB RAM, Ubuntu 22.04. Index
              size: 1M 384-dim vectors (all-MiniLM-L6-v2). Latency measured at
              95th percentile over 10K queries.
            </p>
          </div>
        </div>

        {/* Key wins callout */}
        <div className="mt-8 grid grid-cols-1 sm:grid-cols-3 gap-4">
          {[
            {
              label: '35× Faster Queries',
              sub: 'vs Elasticsearch for semantic search',
            },
            {
              label: '4× Less Memory',
              sub: 'vs Elasticsearch at 1M vectors',
            },
            {
              label: 'Zero Data Exposure',
              sub: 'your data never leaves the machine',
            },
          ].map((item) => (
            <div
              key={item.label}
              className="glass-card rounded-xl p-4 flex items-center gap-3 hover:border-iso-gold/30 transition-colors"
            >
              <div className="w-10 h-10 rounded-xl bg-iso-gold/10 border border-iso-gold/20 flex items-center justify-center flex-shrink-0">
                <ArrowDown className="w-5 h-5 text-iso-gold" />
              </div>
              <div>
                <p className="text-sm font-semibold text-iso-gold">
                  {item.label}
                </p>
                <p className="text-xs text-muted-foreground">{item.sub}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
