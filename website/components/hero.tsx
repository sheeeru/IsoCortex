'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ArrowRight, Shield, Zap, Database, Cpu, Lock, Wifi } from 'lucide-react';

export function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden hero-gradient">
      <div className="absolute inset-0">
        <img
          src="/hero-bg.png"
          alt=""
          className="w-full h-full object-cover opacity-30 animate-pulse-glow"
          aria-hidden="true"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-background/40 via-background/80 to-background" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-32 lg:py-40">
        <div className="flex flex-col items-center text-center">
          <Badge
            variant="outline"
            className="mb-6 px-4 py-1.5 text-xs font-medium tracking-wider uppercase border-iso-gold/30 text-iso-gold bg-iso-gold/5 animate-fade-in"
          >
            100% Local &middot; Zero Cloud Dependency &middot; Air-Gap Ready
          </Badge>

          <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight leading-[1.1] max-w-5xl animate-slide-up">
            <span className="block">Semantic Search</span>
            <span className="block gradient-text">That Never Leaves</span>
            <span className="block text-muted-foreground/80">Your Machine</span>
          </h1>

          <p className="mt-6 text-lg sm:text-xl text-muted-foreground max-w-2xl leading-relaxed animate-fade-in animation-delay-200" style={{ opacity: 0 }}>
            A high-performance neural information retrieval engine. Index 20+ file formats,
            search millions of documents with sub-millisecond latency — all running locally
            with zero network requests.
          </p>

          <div className="mt-10 flex flex-col sm:flex-row items-center gap-4 animate-fade-in animation-delay-400" style={{ opacity: 0 }}>
            <a href="#demo">
              <Button size="lg" className="gap-2 px-8 py-6 text-base font-semibold bg-iso-gold hover:bg-iso-gold-light text-background">
                Try Live Demo
                <ArrowRight className="w-4 h-4" />
              </Button>
            </a>
            <a href="#features">
              <Button size="lg" variant="outline" className="gap-2 px-8 py-6 text-base border-border/50 hover:bg-secondary/50">
                Explore Features
              </Button>
            </a>
          </div>

          <div className="mt-16 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 sm:gap-6 animate-fade-in animation-delay-600" style={{ opacity: 0 }}>
            {[
              { icon: Shield, label: 'GDPR & HIPAA' },
              { icon: Zap, label: 'Sub-ms Search' },
              { icon: Database, label: '20+ Formats' },
              { icon: Cpu, label: 'Fast Search' },
              { icon: Lock, label: 'Zero Cloud' },
              { icon: Wifi, label: 'Offline Ready' },
            ].map((item) => (
              <div
                key={item.label}
                className="glass-card rounded-xl p-4 flex flex-col items-center gap-2 text-center hover:border-iso-purple/40 transition-colors"
              >
                <item.icon className="w-5 h-5 text-iso-gold" />
                <span className="text-xs font-medium text-muted-foreground">{item.label}</span>
              </div>
            ))}
          </div>

          <div className="mt-12 animate-fade-in animation-delay-800" style={{ opacity: 0 }}>
            <div className="inline-flex items-center gap-3 glass-card rounded-full px-5 py-2.5">
              <div className="flex -space-x-1.5">
                {[...Array(3)].map((_, i) => (
                  <div
                    key={i}
                    className="w-6 h-6 rounded-full border-2 border-background bg-iso-purple/60 flex items-center justify-center"
                  >
                    <span className="text-[8px] font-bold text-iso-gold">
                      {['SQ', 'AK', 'JD'][i]}
                    </span>
                  </div>
                ))}
              </div>
              <div className="text-sm">
                <span className="text-iso-gold font-semibold">Open Source</span>
                <span className="text-muted-foreground"> &middot; Built for teams and organizations</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
        <div className="w-6 h-10 rounded-full border-2 border-muted-foreground/30 flex justify-center pt-2">
          <div className="w-1 h-2 bg-iso-gold rounded-full" />
        </div>
      </div>
    </section>
  );
}
