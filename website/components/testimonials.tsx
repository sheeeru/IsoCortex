import { Star, Quote } from 'lucide-react';

const testimonials = [
  {
    quote:
      'We migrated our entire knowledge base from Pinecone to IsoCortex last quarter. The latency improvement alone justified the switch — queries that took 45ms now return in under a millisecond. More importantly, our legal team finally signed off because zero data ever leaves our infrastructure.',
    name: 'Marta Kowalski',
    role: 'CTO',
    company: 'ClinicFlow Health Systems',
    stars: 5,
  },
  {
    quote:
      'The SIMD acceleration is not marketing fluff — I benchmarked it myself. On my M2 MacBook, vector search throughput is 6x faster than the pure Python HNSW libraries I was using before. The C++17 engine is genuinely impressive.',
    name: 'Raj Chakrabarti',
    role: 'Senior ML Engineer',
    company: 'DeepSet AI',
    stars: 5,
  },
  {
    quote:
      'Docker deployment in literally two minutes. No cluster setup, no Java runtime headaches like Elasticsearch, no API key rotation. Just one command and the API server was up. Our DevOps team was genuinely surprised.',
    name: 'Takeshi Yamamoto',
    role: 'DevOps Lead',
    company: 'Nexus Infrastructure',
    stars: 5,
  },
  {
    quote:
      'In my line of work, air-gapped search is non-negotiable. IsoCortex works completely offline after the initial model pull. I indexed our entire vulnerability database and it runs on a machine that has never touched the internet. Nothing else comes close.',
    name: 'Dr. Elena Voronova',
    role: 'Principal Security Researcher',
    company: 'CertiK Labs',
    stars: 5,
  },
  {
    quote:
      'We searched for months for a search solution that works entirely offline. IsoCortex indexes our entire case file archive in minutes and returns results instantly. Our attorneys can finally find any document without relying on cloud services.',
    name: 'Jordan Mitchell',
    role: 'Head of IT',
    company: 'Canopy Legal Group',
    stars: 4,
  },
  {
    quote:
      'I indexed over 120,000 research papers from arXiv in under a minute. The incremental indexing means I only reprocess new uploads each morning. My literature review workflow went from hours of manual searching to a single semantic query.',
    name: 'Dr. Ananya Deshmukh',
    role: 'Research Data Scientist',
    company: 'Allen Institute for AI',
    stars: 5,
  },
];

export function Testimonials() {
  return (
    <section id="testimonials" className="relative py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16 lg:mb-20">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-iso-purple/10 border border-iso-purple/20 mb-4">
            <div className="w-1.5 h-1.5 rounded-full bg-iso-purple" />
            <span className="text-xs font-medium text-iso-purple tracking-wide uppercase">
              Testimonials
            </span>
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight">
            Trusted by Organizations That{' '}
            <span className="gradient-text">Value Privacy</span>
          </h2>
          <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
            From healthcare to defense, organizations choose IsoCortex when data
            sovereignty is not optional. Hear what they have to say.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {testimonials.map((testimonial, index) => (
            <div
              key={testimonial.name}
              className="group glass-card rounded-2xl p-6 hover:border-iso-purple/40 transition-all duration-300 hover:-translate-y-0.5"
              style={{ animationDelay: `${index * 0.05}s` }}
            >
              <div className="flex items-center gap-1 mb-4">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Star
                    key={i}
                    className={`w-4 h-4 ${
                      i < testimonial.stars
                        ? 'fill-iso-gold text-iso-gold'
                        : 'fill-muted text-muted/30'
                    }`}
                  />
                ))}
              </div>

              <div className="relative mb-5">
                <Quote className="w-5 h-5 text-iso-purple/20 mb-2" />
                <p className="text-sm text-muted-foreground leading-relaxed italic">
                  &ldquo;{testimonial.quote}&rdquo;
                </p>
              </div>

              <div className="flex items-center gap-3 pt-4 border-t border-border/50">
                <div className="w-10 h-10 rounded-full bg-iso-purple/10 border border-iso-purple/20 flex items-center justify-center">
                  <span className="text-sm font-semibold text-iso-gold">
                    {testimonial.name
                      .split(' ')
                      .map((n) => n[0])
                      .join('')}
                  </span>
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-semibold truncate">
                    {testimonial.name}
                  </p>
                  <p className="text-xs text-muted-foreground truncate">
                    {testimonial.role} at {testimonial.company}
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
