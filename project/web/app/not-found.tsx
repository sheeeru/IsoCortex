import Link from 'next/link';

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-8" style={{ backgroundColor: '#0F0A1A' }}>
      <div
        className="rounded-xl border p-12 text-center max-w-md"
        style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}
      >
        <p className="text-6xl font-bold mb-4" style={{ color: '#311B5B' }}>404</p>
        <h1 className="text-xl font-semibold text-white mb-2">Page Not Found</h1>
        <p className="text-sm text-gray-400 mb-6">
          The page you&apos;re looking for doesn&apos;t exist or has been moved.
        </p>
        <Link
          href="/dashboard"
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium text-white transition-colors"
          style={{ backgroundColor: '#311B5B' }}
        >
          Go to Dashboard
        </Link>
      </div>
    </div>
  );
}
