'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import { ApiError } from '@/lib/api';
import { Spinner } from '@/components/ui/Spinner';
import { useToast } from '@/components/ui/Toast';

export default function LoginPage() {
  const router = useRouter();
  const { login } = useAuth();
  const { addToast } = useToast();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await login(username, password);
      addToast('Welcome back!', 'success');
      router.push('/dashboard');
    } catch (err) {
      if (err instanceof ApiError) {
        if (err.status === 404 || err.detail?.includes('setup')) {
          router.push('/setup');
          return;
        }
        setError(err.detail);
      } else {
        setError('Failed to connect to server. Is the API running?');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="min-h-screen flex items-center justify-center px-4"
      style={{ backgroundColor: '#0F0A1A' }}
    >
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <div
            className="w-16 h-16 mx-auto mb-4 rounded-2xl flex items-center justify-center font-bold text-white text-2xl shadow-lg"
            style={{ backgroundColor: '#311B5B', boxShadow: '0 0 40px rgba(49, 27, 91, 0.4)' }}
          >
            IC
          </div>
          <h1 className="text-2xl font-bold text-white">
            Iso<span style={{ color: '#C59B47' }}>Cortex</span>
          </h1>
          <p className="text-gray-500 mt-1 text-sm">Semantic Search Engine</p>
        </div>

        {/* Login Card */}
        <div
          className="rounded-2xl p-8 border"
          style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}
        >
          <h2 className="text-xl font-semibold text-white mb-6">Sign In</h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div className="px-4 py-3 rounded-lg text-sm text-red-400" style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                {error}
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-400 mb-1.5">Username</label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                className="w-full px-4 py-2.5 rounded-lg text-white text-sm placeholder-gray-500"
                style={{ backgroundColor: '#0F0A1A', border: '1px solid #2D1F45' }}
                placeholder="Enter your username"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-400 mb-1.5">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="w-full px-4 py-2.5 rounded-lg text-white text-sm placeholder-gray-500"
                style={{ backgroundColor: '#0F0A1A', border: '1px solid #2D1F45' }}
                placeholder="Enter your password"
              />
            </div>

            <button
              type="submit"
              disabled={loading || !username || !password}
              className="w-full py-2.5 rounded-lg text-sm font-semibold text-white transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              style={{
                backgroundColor: '#311B5B',
                boxShadow: loading ? 'none' : '0 0 20px rgba(49, 27, 91, 0.4)',
              }}
              onMouseEnter={(e) => {
                if (!loading) e.currentTarget.style.backgroundColor = '#4A2D82';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = '#311B5B';
              }}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <Spinner size="sm" /> Signing in...
                </span>
              ) : (
                'Sign In'
              )}
            </button>
          </form>

          <p className="text-center text-xs text-gray-500 mt-6">
            First time?{' '}
            <button
              onClick={() => router.push('/setup')}
              className="hover:underline"
              style={{ color: '#C59B47' }}
            >
              Run initial setup
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}
