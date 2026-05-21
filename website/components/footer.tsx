'use client';

import { Brain, Github, Mail, Twitter } from 'lucide-react';

const footerLinks = {
  Product: [
    { label: 'Features', href: '#features' },
    { label: 'Architecture', href: '#architecture' },
    { label: 'Live Demo', href: '#demo' },
    { label: 'Pricing', href: '#pricing' },
    { label: 'API Reference', href: '#api' },
    { label: 'Changelog', href: '', disabled: true },
  ],
  Resources: [
    { label: 'Documentation', href: '', disabled: true },
    { label: 'GitHub Repository', href: 'https://github.com/sheeeru/IsoCortex' },
    { label: 'PyPI Package', href: '', disabled: true },
    { label: 'Docker Hub', href: '', disabled: true },
    { label: 'Contributing Guide', href: '', disabled: true },
    { label: 'SRS Document', href: '', disabled: true },
  ],
  Company: [
    { label: 'About', href: '', disabled: true },
    { label: 'Blog', href: '', disabled: true },
    { label: 'Contact', href: '', disabled: true },
    { label: 'Privacy Policy', href: '', disabled: true },
    { label: 'Terms of Service', href: '', disabled: true },
    { label: 'License (MIT)', href: 'https://github.com/sheeeru/IsoCortex/blob/main/LICENSE' },
  ],
};

export function Footer() {
  return (
    <footer className="border-t border-border/50 bg-secondary/20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 lg:py-16">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 lg:gap-12">
          <div className="col-span-2 md:col-span-1">
            <div className="flex items-center gap-2.5 mb-4">
              <Brain className="w-7 h-7 text-iso-gold" />
              <div className="flex flex-col">
                <span className="text-base font-bold tracking-tight leading-none">
                  <span className="text-iso-purple">Iso</span>
                  <span className="text-iso-gold">Cortex</span>
                </span>
                <span className="text-[9px] text-muted-foreground tracking-widest uppercase leading-none mt-0.5">
                  Neural Retrieval
                </span>
              </div>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed mb-4">
              High-performance, 100% local neural information retrieval engine. Your data never leaves your machine.
            </p>
            <div className="flex items-center gap-3">
              <a
                href="https://github.com/sheeeru/IsoCortex"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="IsoCortex on GitHub"
                className="w-9 h-9 rounded-lg bg-secondary/50 border border-border/50 flex items-center justify-center text-muted-foreground hover:text-iso-gold hover:border-iso-gold/30 transition-colors"
              >
                <Github className="w-4 h-4" />
              </a>
              <a
                href="#"
                aria-label="Follow IsoCortex on Twitter (coming soon)"
                className="w-9 h-9 rounded-lg bg-secondary/50 border border-border/50 flex items-center justify-center text-muted-foreground/40 cursor-not-allowed"
                aria-disabled="true"
                tabIndex={-1}
                onClick={(e: React.MouseEvent) => e.preventDefault()}
              >
                <Twitter className="w-4 h-4" />
              </a>
              <a
                href="mailto:contact@isocortex.dev"
                aria-label="Contact IsoCortex via email"
                className="w-9 h-9 rounded-lg bg-secondary/50 border border-border/50 flex items-center justify-center text-muted-foreground hover:text-iso-gold hover:border-iso-gold/30 transition-colors"
              >
                <Mail className="w-4 h-4" />
              </a>
            </div>
          </div>

          {Object.entries(footerLinks).map(([category, links]) => (
            <div key={category}>
              <h4 className="text-sm font-semibold mb-4">{category}</h4>
              <ul className="space-y-2.5">
                {links.map((link) => (
                  <li key={link.label}>
                    <a
                      href={'disabled' in link ? undefined : link.href}
                      className={`text-sm transition-colors ${'disabled' in link ? 'text-muted-foreground/40 cursor-not-allowed' : 'text-muted-foreground hover:text-foreground'}`}
                      {...('disabled' in link && { 'aria-disabled': true, tabIndex: -1, onClick: (e: React.MouseEvent) => e.preventDefault() })}
                    >
                      {link.label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="section-divider mt-12 mb-8" />

        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs text-muted-foreground">
            &copy; {new Date().getFullYear()} IsoCortex. Lead Architect & Developer: Shaheer Qureshi.
            All rights reserved.
          </p>
          <p className="text-xs text-muted-foreground">
            Core engine licensed under MIT. Pro/Enterprise features require a commercial license.
          </p>
        </div>
      </div>
    </footer>
  );
}
