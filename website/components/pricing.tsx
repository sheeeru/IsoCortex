import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Check, X, Star, Zap, Building2, Briefcase } from 'lucide-react';

const GITHUB = 'https://github.com/sheeeru/IsoCortex';

const tiers = [
  {
    name: 'Open Source',
    price: 'Free',
    period: '',
    description: 'For individuals and small teams who need private local search.',
    icon: Zap,
    features: [
      { text: 'Full ingestion pipeline (20+ formats)', included: true },
      { text: 'Single index (default)', included: true },
      { text: 'C++ HNSW engine with SIMD', included: true },
      { text: 'Sentence-aware chunking', included: true },
      { text: 'Incremental indexing', included: true },
      { text: 'Index export/import', included: true },
      { text: 'Web UI', included: false },
      { text: 'REST API server', included: false },
      { text: 'Multi-index support', included: false },
      { text: 'Authentication & RBAC', included: false },
      { text: 'Usage analytics', included: false },
      { text: 'Docker packaging', included: false },
      { text: 'Priority support', included: false },
    ],
    cta: 'Download on GitHub',
    ctaStyle: 'outline' as const,
    ctaHref: GITHUB,
    popular: false,
  },
  {
    name: 'Pro',
    price: '$19',
    period: '/month',
    description: 'For small teams who need a web interface and API access.',
    icon: Star,
    features: [
      { text: 'Everything in Open Source', included: true },
      { text: 'Web UI (Next.js)', included: true },
      { text: 'REST API server (20+ endpoints)', included: true },
      { text: 'Multi-index (up to 10)', included: true },
      { text: 'API keys (up to 3)', included: true },
      { text: 'Basic analytics dashboard', included: true },
      { text: 'Docker packaging', included: true },
      { text: 'Email support', included: true },
      { text: 'JWT authentication', included: false },
      { text: 'Role-based access control', included: false },
      { text: 'Rate limiting', included: false },
      { text: 'Custom embedding models', included: false },
      { text: 'SSO/LDAP integration', included: false },
      { text: 'SLA guarantee', included: false },
    ],
    cta: 'Start Free Trial',
    ctaStyle: 'default' as const,
    ctaHref: GITHUB,
    popular: true,
  },
  {
    name: 'Team',
    price: '$49',
    period: '/month',
    description: 'For growing teams needing authentication and analytics.',
    icon: Building2,
    features: [
      { text: 'Everything in Pro', included: true },
      { text: 'JWT authentication', included: true },
      { text: 'Unlimited API keys', included: true },
      { text: 'Role-based access control', included: true },
      { text: 'Rate limiting (per-key)', included: true },
      { text: 'Advanced analytics + export', included: true },
      { text: 'Unlimited indexes', included: true },
      { text: 'Priority support', included: true },
      { text: 'Custom embedding models', included: false },
      { text: 'SSO/LDAP integration', included: false },
      { text: 'Audit logging', included: false },
      { text: 'SLA guarantee', included: false },
      { text: 'Custom branding', included: false },
      { text: 'Compliance reports', included: false },
    ],
    cta: 'Start Free Trial',
    ctaStyle: 'outline' as const,
    ctaHref: GITHUB,
    popular: false,
  },
  {
    name: 'Enterprise',
    price: 'Custom',
    period: '',
    description: 'For organizations requiring compliance and custom deployment.',
    icon: Briefcase,
    features: [
      { text: 'Everything in Team', included: true },
      { text: 'SSO/LDAP integration', included: true },
      { text: 'Audit logging', included: true },
      { text: 'Custom embedding models', included: true },
      { text: 'SLA guarantee', included: true },
      { text: 'On-premise deployment support', included: true },
      { text: 'Custom branding', included: true },
      { text: 'Compliance reports', included: true },
      { text: 'Dedicated support engineer', included: true },
      { text: 'Industry-specific configurations', included: true },
      { text: 'Custom chunking rules', included: true },
      { text: 'Training & onboarding', included: true },
      { text: 'Data migration assistance', included: true },
      { text: 'Annual security review', included: true },
    ],
    cta: 'Contact Sales',
    ctaStyle: 'outline' as const,
    ctaHref: 'mailto:contact@isocortex.com',
    popular: false,
  },
];

export function Pricing() {
  return (
    <section id="pricing" className="relative py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-iso-gold/10 border border-iso-gold/20 mb-4">
            <div className="w-1.5 h-1.5 rounded-full bg-iso-gold" />
            <span className="text-xs font-medium text-iso-gold tracking-wide uppercase">Pricing</span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight">
            Simple, Transparent{' '}
            <span className="gradient-text">Pricing</span>
          </h2>
          <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
            Start free with the open-source edition. Upgrade when you need a web interface,
            team features, or enterprise compliance.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-5">
          {tiers.map((tier) => (
            <div
              key={tier.name}
              className={`relative rounded-2xl p-6 flex flex-col ${
                tier.popular
                  ? 'glass-card border-iso-gold/40 shadow-lg shadow-iso-gold/5'
                  : 'glass-card hover:border-iso-purple/40'
              } transition-all duration-300`}
            >
              {tier.popular && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                  <Badge className="bg-iso-gold text-background font-semibold text-xs px-3 py-0.5">
                    Most Popular
                  </Badge>
                </div>
              )}

              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-iso-purple/10 border border-iso-purple/20 flex items-center justify-center">
                  <tier.icon className="w-5 h-5 text-iso-gold" />
                </div>
                <div>
                  <h3 className="font-semibold">{tier.name}</h3>
                  <p className="text-xs text-muted-foreground">{tier.description}</p>
                </div>
              </div>

              <div className="mb-6">
                <div className="flex items-baseline gap-1">
                  <span className="text-3xl font-bold">{tier.price}</span>
                  {tier.period && <span className="text-sm text-muted-foreground">{tier.period}</span>}
                </div>
              </div>

              <a
                href={tier.ctaHref}
                target={tier.ctaHref.startsWith('http') ? '_blank' : undefined}
                rel={tier.ctaHref.startsWith('http') ? 'noopener noreferrer' : undefined}
                className="w-full mb-6"
              >
                <Button
                  className={`w-full ${
                    tier.ctaStyle === 'default'
                      ? 'bg-iso-gold hover:bg-iso-gold-light text-background font-semibold'
                      : 'border-border/50 hover:bg-secondary/50 hover:border-iso-purple/40'
                  }`}
                  variant={tier.ctaStyle}
                >
                  {tier.cta}
                </Button>
              </a>

              <ul className="space-y-2.5 flex-1">
                {tier.features.map((feature) => (
                  <li key={feature.text} className="flex items-start gap-2.5">
                    {feature.included ? (
                      <Check className="w-4 h-4 text-iso-gold flex-shrink-0 mt-0.5" />
                    ) : (
                      <X className="w-4 h-4 text-muted-foreground/40 flex-shrink-0 mt-0.5" />
                    )}
                    <span
                      className={`text-xs leading-relaxed ${
                        feature.included ? 'text-muted-foreground' : 'text-muted-foreground/40'
                      }`}
                    >
                      {feature.text}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-12 text-center">
          <p className="text-sm text-muted-foreground">
            All tiers include 100% local processing. Zero cloud dependencies. Your data never leaves your machine.
          </p>
          <p className="text-xs text-muted-foreground/60 mt-2">
            Core engine is MIT-licensed. Pro/Team/Enterprise features require a license key.
          </p>
        </div>
      </div>
    </section>
  );
}
