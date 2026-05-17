'use client';

import { Button } from '@/components/ui/button';
import { ArrowRight, Github, Star, Users, Zap } from 'lucide-react';

export function CTA() {
  return (
    <section className="relative py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="relative glass-card rounded-3xl p-8 sm:p-12 lg:p-16 overflow-hidden">
          <div className="absolute inset-0 hero-gradient opacity-50" />
          <div className="absolute top-0 right-0 w-96 h-96 bg-iso-purple/10 rounded-full blur-3xl" />
          <div className="absolute bottom-0 left-0 w-64 h-64 bg-iso-gold/10 rounded-full blur-3xl" />

          <div className="relative z-10 text-center">
            <div className="inline-flex items-center gap-4 mb-8">
              <div className="flex items-center gap-2 glass-card rounded-full px-4 py-2">
                <Star className="w-4 h-4 text-iso-gold" />
                <span className="text-sm font-semibold">Open Source</span>
                <span className="text-xs text-muted-foreground">MIT License</span>
              </div>
              <div className="flex items-center gap-2 glass-card rounded-full px-4 py-2">
                <Zap className="w-4 h-4 text-iso-gold" />
                <span className="text-sm font-semibold">v1.0</span>
                <span className="text-xs text-muted-foreground">Production Ready</span>
              </div>
            </div>

            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight mb-4">
              Ready to Take Control of{' '}
              <span className="gradient-text">Your Data?</span>
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-8">
              Stop sending your documents to the cloud. Start searching intelligently,
              privately, and instantly with IsoCortex.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button
                  size="lg"
                  className="gap-2 px-8 py-6 text-base bg-iso-gold hover:bg-iso-gold-light text-background font-semibold"
                >
                  <Github className="w-5 h-5" />
                  Star on GitHub
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </a>
              <a href="#demo">
                <Button
                  size="lg"
                  variant="outline"
                  className="gap-2 px-8 py-6 text-base border-border/50 hover:bg-secondary/50"
                >
                  <Zap className="w-4 h-4" />
                  Try Live Demo
                </Button>
              </a>
            </div>

            <div className="mt-12 flex items-center justify-center gap-8 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <Users className="w-4 h-4 text-iso-purple" />
                <span>Built for developers, by developers</span>
              </div>
              <div className="w-px h-4 bg-border" />
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4 text-iso-gold" />
                <span>Zero cloud dependency</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function Shield({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  );
}
