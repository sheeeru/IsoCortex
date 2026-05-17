'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useAuth } from '@/lib/auth';
import {
  Bars3Icon,
  ArrowRightOnRectangleIcon,
  UserCircleIcon,
} from '@heroicons/react/24/outline';

interface HeaderProps {
  onMenuClick: () => void;
  title?: string;
}

export default function Header({ onMenuClick, title }: HeaderProps) {
  const { user, logout } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <header
      className="sticky top-0 z-30 h-16 flex items-center justify-between px-4 sm:px-6 border-b border-purple-900/30"
      style={{ backgroundColor: 'rgba(15, 10, 26, 0.85)', backdropFilter: 'blur(12px)' }}
    >
      <div className="flex items-center gap-3">
        <button
          onClick={onMenuClick}
          className="lg:hidden p-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
        >
          <Bars3Icon className="w-5 h-5" />
        </button>
        {title && <h1 className="text-lg font-semibold text-white">{title}</h1>}
      </div>

      <div className="relative" ref={menuRef}>
        <button
          onClick={() => setMenuOpen(!menuOpen)}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-gray-300 hover:text-white hover:bg-white/5 transition-colors"
        >
          <UserCircleIcon className="w-6 h-6" />
          <span className="hidden sm:block text-sm font-medium">{user?.username || 'User'}</span>
        </button>

        {menuOpen && (
          <div
            className="absolute right-0 mt-2 w-56 rounded-xl shadow-2xl border border-purple-900/50 py-1 z-50"
            style={{ backgroundColor: '#1A1228' }}
          >
            <div className="px-4 py-3 border-b border-purple-900/50">
              <p className="text-sm font-medium text-white">{user?.username}</p>
              <p className="text-xs text-gray-400">{user?.email || user?.role}</p>
              <span
                className="inline-block mt-1 px-2 py-0.5 text-xs font-medium rounded-full"
                style={{
                  backgroundColor: user?.role === 'admin' ? 'rgba(197, 155, 71, 0.2)' : 'rgba(49, 27, 91, 0.5)',
                  color: user?.role === 'admin' ? '#C59B47' : '#8B7EC8',
                }}
              >
                {user?.role?.toUpperCase()}
              </span>
            </div>
            <button
              onClick={() => {
                setMenuOpen(false);
                logout();
              }}
              className="flex items-center gap-2 w-full px-4 py-2.5 text-sm text-red-400 hover:bg-red-500/10 transition-colors"
            >
              <ArrowRightOnRectangleIcon className="w-4 h-4" />
              Sign Out
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
