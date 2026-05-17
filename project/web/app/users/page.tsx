'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { authApi, ApiError } from '@/lib/api';
import { User, CreateUserRequest } from '@/lib/types';
import { PageSpinner, Spinner } from '@/components/ui/Spinner';
import { useToast } from '@/components/ui/Toast';
import { useAuth } from '@/lib/auth';
import { formatDate } from '@/lib/utils';
import {
  PlusIcon,
  TrashIcon,
  XMarkIcon,
  UsersIcon,
  ShieldCheckIcon,
  UserIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';

export default function UsersPage() {
  const { isAdmin } = useAuth();
  const { addToast } = useToast();
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [creating, setCreating] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  const fetchUsers = useCallback(async () => {
    setLoading(true);
    try {
      const data = await authApi.listUsers() as any;
      if (Array.isArray(data)) setUsers(data);
      else if (data && Array.isArray(data.users)) setUsers(data.users);
      else setUsers([]);
    } catch {
      addToast('Failed to load users', 'error');
    } finally {
      setLoading(false);
    }
  }, [addToast]);

  useEffect(() => {
    if (isAdmin) fetchUsers();
  }, [isAdmin, fetchUsers]);

  if (!isAdmin) {
    return (
      <div className="rounded-xl border p-16 text-center" style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}>
        <ExclamationTriangleIcon className="w-10 h-10 text-amber-500 mx-auto mb-4" />
        <h3 className="text-gray-300 text-lg font-medium">Access Denied</h3>
        <p className="text-gray-500 text-sm mt-2">You need admin privileges to manage users</p>
      </div>
    );
  }

  const handleCreate = async (data: CreateUserRequest) => {
    setCreating(true);
    try {
      await authApi.createUser(data);
      addToast(`User "${data.username}" created`, 'success');
      setShowCreate(false);
      fetchUsers();
    } catch (err) {
      if (err instanceof ApiError) addToast(err.detail, 'error');
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      await authApi.deleteUser(deleteTarget);
      addToast('User deleted', 'success');
      setDeleteTarget(null);
      fetchUsers();
    } catch (err) {
      if (err instanceof ApiError) addToast(err.detail, 'error');
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-gray-400">{users.length} user{users.length !== 1 ? 's' : ''}</p>
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-white"
          style={{ backgroundColor: '#311B5B' }}
        >
          <PlusIcon className="w-4 h-4" />
          Add User
        </button>
      </div>

      {loading ? (
        <PageSpinner />
      ) : users.length === 0 ? (
        <div className="rounded-xl border p-12 text-center" style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}>
          <UsersIcon className="w-10 h-10 text-gray-600 mx-auto mb-3" />
          <p className="text-gray-400 text-sm">No users found</p>
        </div>
      ) : (
        <div className="rounded-xl border overflow-hidden" style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b" style={{ borderColor: '#2D1F45' }}>
                  <th className="text-left text-xs font-medium text-gray-500 px-5 py-3">User</th>
                  <th className="text-left text-xs font-medium text-gray-500 px-5 py-3 hidden sm:table-cell">Email</th>
                  <th className="text-left text-xs font-medium text-gray-500 px-5 py-3">Role</th>
                  <th className="text-left text-xs font-medium text-gray-500 px-5 py-3 hidden md:table-cell">Status</th>
                  <th className="text-left text-xs font-medium text-gray-500 px-5 py-3 hidden lg:table-cell">Created</th>
                  <th className="text-right text-xs font-medium text-gray-500 px-5 py-3">Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.map((user) => (
                  <tr
                    key={user.id || user.username || user.email || Math.random()}
                    className="border-b hover:bg-white/[0.02] transition-colors"
                    style={{ borderColor: '#2D1F45' }}
                  >
                    <td className="px-5 py-3">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold"
                          style={{ backgroundColor: user.role === 'admin' ? 'rgba(197, 155, 71, 0.2)' : 'rgba(49, 27, 91, 0.5)', color: user.role === 'admin' ? '#C59B47' : '#8B7EC8' }}
                        >
                          {user.role === 'admin' ? (
                            <ShieldCheckIcon className="w-4 h-4" />
                          ) : (
                            <UserIcon className="w-4 h-4" />
                          )}
                        </div>
                        <span className="text-sm font-medium text-white">{user.username}</span>
                      </div>
                    </td>
                    <td className="px-5 py-3 hidden sm:table-cell">
                      <span className="text-sm text-gray-400">{user.email || '—'}</span>
                    </td>
                    <td className="px-5 py-3">
                      <span
                        className="px-2 py-0.5 rounded-full text-xs font-semibold capitalize"
                        style={{
                          backgroundColor: user.role === 'admin' ? 'rgba(197, 155, 71, 0.15)' : 'rgba(49, 27, 91, 0.4)',
                          color: user.role === 'admin' ? '#C59B47' : '#8B7EC8',
                        }}
                      >
                        {user.role}
                      </span>
                    </td>
                    <td className="px-5 py-3 hidden md:table-cell">
                      <div className="flex items-center gap-1.5">
                        <div
                          className="w-2 h-2 rounded-full"
                          style={{ backgroundColor: user.is_active ? '#10B981' : '#6B7280' }}
                        />
                        <span className="text-sm text-gray-400">{user.is_active ? 'Active' : 'Inactive'}</span>
                      </div>
                    </td>
                    <td className="px-5 py-3 hidden lg:table-cell">
                      <span className="text-xs text-gray-500">{formatDate(user.created_at)}</span>
                    </td>
                    <td className="px-5 py-3 text-right">
                      <button
                        onClick={() => setDeleteTarget(user.id)}
                        className="p-1.5 rounded text-gray-400 hover:text-red-400 hover:bg-red-500/10"
                      >
                        <TrashIcon className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Create Modal */}
      {showCreate && (
        <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
          <div className="absolute inset-0 bg-black/60" onClick={() => setShowCreate(false)} />
          <div className="relative w-full max-w-md rounded-2xl border p-6" style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-white">Create User</h2>
              <button onClick={() => setShowCreate(false)} className="p-1 rounded text-gray-400 hover:text-white hover:bg-white/10">
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>
            <CreateUserForm
              onSubmit={handleCreate}
              loading={creating}
              onCancel={() => setShowCreate(false)}
            />
          </div>
        </div>
      )}

      {/* Delete Modal */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
          <div className="absolute inset-0 bg-black/60" onClick={() => setDeleteTarget(null)} />
          <div className="relative w-full max-w-sm rounded-2xl border p-6" style={{ backgroundColor: '#1A1228', borderColor: '#2D1F45' }}>
            <h2 className="text-lg font-semibold text-white mb-2">Delete User</h2>
            <p className="text-sm text-gray-400 mb-6">This action cannot be undone.</p>
            <div className="flex gap-3">
              <button onClick={() => setDeleteTarget(null)} disabled={deleting} className="flex-1 py-2.5 rounded-lg text-sm font-medium text-gray-300 border" style={{ borderColor: '#2D1F45' }}>
                Cancel
              </button>
              <button onClick={handleDelete} disabled={deleting} className="flex-1 py-2.5 rounded-lg text-sm font-semibold text-white" style={{ backgroundColor: '#991B1B' }}>
                {deleting ? <Spinner size="sm" /> : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function CreateUserForm({
  onSubmit,
  loading,
  onCancel,
}: {
  onSubmit: (data: CreateUserRequest) => void;
  loading: boolean;
  onCancel: () => void;
}) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
  const [role, setRole] = useState<'admin' | 'user'>('user');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({ username, password, email: email || undefined, role });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-400 mb-1">Username</label>
        <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} required className="w-full px-4 py-2.5 rounded-lg text-white text-sm placeholder-gray-500" style={{ backgroundColor: '#0F0A1A', border: '1px solid #2D1F45' }} placeholder="username" />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-400 mb-1">Password</label>
        <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8} className="w-full px-4 py-2.5 rounded-lg text-white text-sm placeholder-gray-500" style={{ backgroundColor: '#0F0A1A', border: '1px solid #2D1F45' }} placeholder="Min 8 characters" />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-400 mb-1">Email (optional)</label>
        <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} className="w-full px-4 py-2.5 rounded-lg text-white text-sm placeholder-gray-500" style={{ backgroundColor: '#0F0A1A', border: '1px solid #2D1F45' }} placeholder="user@example.com" />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-400 mb-1">Role</label>
        <select value={role} onChange={(e) => setRole(e.target.value as 'admin' | 'user')} className="w-full px-4 py-2.5 rounded-lg text-white text-sm appearance-none" style={{ backgroundColor: '#0F0A1A', border: '1px solid #2D1F45' }}>
          <option value="user">User</option>
          <option value="admin">Admin</option>
        </select>
      </div>
      <div className="flex gap-3 pt-2">
        <button type="button" onClick={onCancel} className="flex-1 py-2.5 rounded-lg text-sm font-medium text-gray-300 border" style={{ borderColor: '#2D1F45' }}>Cancel</button>
        <button type="submit" disabled={loading || !username || !password} className="flex-1 py-2.5 rounded-lg text-sm font-semibold text-white disabled:opacity-50" style={{ backgroundColor: '#311B5B' }}>
          {loading ? <Spinner size="sm" /> : 'Create User'}
        </button>
      </div>
    </form>
  );
}
