'use client';

import { Navbar } from '../../components/navbar';
import { Hero } from '../../components/hero';
import { Features } from '../../components/features';
import { HowItWorks } from '../../components/architecture';
import { Demo } from '../../components/demo';
import { Benchmarks } from '../../components/benchmarks';
import { Comparison } from '../../components/comparison';
import { Testimonials } from '../../components/testimonials';
import { Pricing } from '../../components/pricing';
import { ApiPreview } from '../../components/api-preview';
import { CTA } from '../../components/cta';
import { Footer } from '../../components/footer';

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
        <Benchmarks />
        <div className="section-divider mx-auto max-w-5xl" />
        <Comparison />
        <div className="section-divider mx-auto max-w-5xl" />
        <Testimonials />
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
