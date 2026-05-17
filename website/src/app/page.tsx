'use client';

import { Navbar } from '@/components/isocortex/navbar';
import { Hero } from '@/components/isocortex/hero';
import { Features } from '@/components/isocortex/features';
import { HowItWorks } from '@/components/isocortex/architecture';
import { Demo } from '@/components/isocortex/demo';
import { Pricing } from '@/components/isocortex/pricing';
import { ApiPreview } from '@/components/isocortex/api-preview';
import { CTA } from '@/components/isocortex/cta';
import { Footer } from '@/components/isocortex/footer';

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1">
        <Hero />
        <div className="section-divider mx-auto max-w-5xl" />
        <Features />
        <div className="section-divider mx-auto max-w-5xl" />
        <HowItWorks />
        <div className="section-divider mx-auto max-w-5xl" />
        <Demo />
        <div className="section-divider mx-auto max-w-5xl" />
        <Pricing />
        <div className="section-divider mx-auto max-w-5xl" />
        <ApiPreview />
        <CTA />
      </main>
      <Footer />
    </div>
  );
}
