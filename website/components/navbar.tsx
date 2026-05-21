'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTrigger, SheetTitle } from '@/components/ui/sheet';
import { Menu, X, Brain, Github, Terminal } from 'lucide-react';

const navLinks = [
  { label: 'Features', href: '#features' },
  { label: 'Architecture', href: '#architecture' },
  { label: 'Demo', href: '#demo' },
  { label: 'Pricing', href: '#pricing' },
  { label: 'API', href: '#api' },
];

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? 'bg-background/80 backdrop-blur-xl border-b border-border/50 shadow-lg shadow-primary/5'
          : 'bg-transparent'
      }`}
    >
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 lg:h-20">
          <a href="#" className="flex items-center gap-2.5 group">
            <div className="relative w-9 h-9 flex items-center justify-center">
              <Brain className="w-8 h-8 text-iso-gold transition-transform group-hover:scale-110" />
              <div className="absolute inset-0 rounded-full bg-iso-purple/20 blur-lg group-hover:bg-iso-purple/30 transition-colors" />
            </div>
            <div className="flex flex-col">
              <span className="text-lg font-bold tracking-tight leading-none">
                <span className="text-iso-purple">Iso</span>
                <span className="text-iso-gold">Cortex</span>
              </span>
              <span className="text-[10px] text-muted-foreground tracking-widest uppercase leading-none mt-0.5">
                Neural Retrieval
              </span>
            </div>
          </a>

          <div className="hidden lg:flex items-center gap-1">
            {navLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="px-4 py-2 text-sm text-muted-foreground hover:text-foreground transition-colors rounded-lg hover:bg-secondary/50"
              >
                {link.label}
              </a>
            ))}
          </div>

          <div className="hidden lg:flex items-center gap-3">
            <a
              href="https://github.com/sheeeru/IsoCortex"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              <Github className="w-4 h-4" />
              <span>GitHub</span>
            </a>
            <a
              href="#demo"
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-background bg-iso-gold hover:bg-iso-gold-light rounded-lg transition-colors"
            >
              <Terminal className="w-4 h-4" />
              Try Live Demo
            </a>
          </div>

          <Sheet open={open} onOpenChange={setOpen}>
            <SheetTrigger asChild className="lg:hidden">
              <Button variant="ghost" size="icon" className="text-muted-foreground">
                <Menu className="w-5 h-5" />
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-80 bg-background/95 backdrop-blur-xl border-border/50 p-0">
              <SheetTitle className="sr-only">Navigation Menu</SheetTitle>
              <div className="flex flex-col h-full">
                <div className="flex items-center justify-between p-6 border-b border-border/50">
                  <span className="text-lg font-bold">
                    <span className="text-iso-purple">Iso</span>
                    <span className="text-iso-gold">Cortex</span>
                  </span>
                  <Button variant="ghost" size="icon" onClick={() => setOpen(false)}>
                    <X className="w-5 h-5" />
                  </Button>
                </div>
                <div className="flex-1 p-6 flex flex-col gap-2">
                  {navLinks.map((link) => (
                    <a
                      key={link.href}
                      href={link.href}
                      onClick={() => setOpen(false)}
                      className="px-4 py-3 text-base text-muted-foreground hover:text-foreground hover:bg-secondary/50 rounded-lg transition-colors"
                    >
                      {link.label}
                    </a>
                  ))}
                </div>
                <div className="p-6 border-t border-border/50 flex flex-col gap-3">
                  <a
                    href="https://github.com/sheeeru/IsoCortex"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-center gap-2 px-4 py-2.5 text-sm border border-border rounded-lg hover:bg-secondary/50 transition-colors"
                  >
                    <Github className="w-4 h-4" />
                    GitHub
                  </a>
                  <a
                    href="#demo"
                    onClick={() => setOpen(false)}
                    className="flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-medium text-background bg-iso-gold hover:bg-iso-gold-light rounded-lg transition-colors"
                  >
                    <Terminal className="w-4 h-4" />
                    Try Live Demo
                  </a>
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </nav>
    </header>
  );
}
