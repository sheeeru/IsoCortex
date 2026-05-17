'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { ApiError } from '@/lib/api';
import { authApi, setTokens } from '@/lib/api';
import { Spinner } from '@/components/ui/Spinner';
import { useToast } from '@/components/ui/Toast';
import { UserPlusIcon } from '@heroicons/react/24/outline';

export default function SetupPage() {
  const router = useRouter();
  const { addToast } = useToast();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    setLoading(true);
    try {
      const response = await authApi.setup({
        username,
        password,
        email: email || undefined,
      });
      setTokens(response.access_token, response.refresh_token);
      addToast('Admin account created successfully!', 'success');
      router.push('/dashboard');
    } catch (err) {
      if (err instanceof ApiError) {
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
            className="w-16 h-16 mx-auto mb-4 rounded-2xl flex items-center justify-center shadow-lg"
            style={{ backgroundColor: '#311B5B', boxShadow: '0 0 40px rgba(49, 27, 91, 0.4)' }}
          >
            <UserPlusIcon className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-white">Initial Setup</h1>
          <p className="text-gray-500 mt-1 text-sm">
            Create your admin account to get started
          </p>
        </div>

        {/* Setup Card */}
        <div
          className="rounded-2xl p-8 border"
          style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}
        >
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div className="px-4 py-3 rounded-lg text-sm text-red-400" style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                {error}
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-400 mb-1.5">Admin Username</label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                className="w-full px-4 py-2.5 rounded-lg text-white text-sm placeholder-gray-500"
                style={{ backgroundColor: '#0F0A1A', border: '1px solid #2D1F45' }}
                placeholder="admin"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-400 mb-1.5">Email (optional)</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-2.5 rounded-lg text-white text-sm placeholder-gray-500"
                style={{ backgroundColor: '#0F0A1A', border: '1px solid #2D1F45' }}
                placeholder="admin@example.com"
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
                placeholder="Minimum 8 characters"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-400 mb-1.5">Confirm Password</label>
              <input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                className="w-full px-4 py-2.5 rounded-lg text-white text-sm placeholder-gray-500"
                style={{ backgroundColor: '#0F0A1A', border: '1px solid #2D1F45' }}
                placeholder="Repeat your password"
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
                  <Spinner size="sm" /> Creating admin account...
                </span>
              ) : (
                'Create Admin Account'
              )}
            </button>
          </form>

          <p className="text-center text-xs text-gray-500 mt-6">
            Already have an account?{' '}
            <button
              onClick={() => router.push('/login')}
              className="hover:underline"
              style={{ color: '#C59B47' }}
            >
              Sign in
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}
