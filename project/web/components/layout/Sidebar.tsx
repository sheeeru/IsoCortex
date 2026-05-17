'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import { cn } from '@/lib/utils';
import {
  HomeIcon,
  MagnifyingGlassIcon,
  CircleStackIcon,
  DocumentTextIcon,
  UsersIcon,
  ChartBarIcon,
  Cog6ToothIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
  { name: 'Search', href: '/search', icon: MagnifyingGlassIcon },
  { name: 'Indexes', href: '/indexes', icon: CircleStackIcon },
  { name: 'Documents', href: '/documents', icon: DocumentTextIcon },
  { name: 'Users', href: '/users', icon: UsersIcon, adminOnly: true },
  { name: 'Analytics', href: '/analytics', icon: ChartBarIcon, adminOnly: true },
  { name: 'Settings', href: '/settings', icon: Cog6ToothIcon },
];

interface SidebarProps {
  open: boolean;
  onClose: () => void;
}

export default function Sidebar({ open, onClose }: SidebarProps) {
  const pathname = usePathname();
  const { isAdmin } = useAuth();

  const filteredNav = navigation.filter(
    (item) => !item.adminOnly || isAdmin
  );

  return (
    <>
      {/* Mobile overlay */}
      {open && (
        <div
          className="fixed inset-0 bg-black/60 z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          'fixed top-0 left-0 z-50 h-full w-64 flex flex-col transition-transform duration-300 lg:translate-x-0 lg:static lg:z-auto',
          open ? 'translate-x-0' : '-translate-x-full'
        )}
        style={{ backgroundColor: '#1A1228' }}
      >
        {/* Logo */}
        <div className="flex items-center justify-between h-16 px-6 border-b border-purple-900/50">
          <Link href="/dashboard" className="flex items-center gap-3">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center font-bold text-white text-sm"
              style={{ backgroundColor: '#311B5B' }}
            >
              IC
            </div>
            <span className="font-semibold text-lg text-white">
              Iso<span style={{ color: '#C59B47' }}>Cortex</span>
            </span>
          </Link>
          <button
            onClick={onClose}
            className="lg:hidden p-1 rounded-md text-gray-400 hover:text-white hover:bg-white/10"
          >
            <XMarkIcon className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {filteredNav.map((item) => {
            const isActive =
              pathname === item.href || pathname.startsWith(item.href + '/');
            return (
              <Link
                key={item.name}
                href={item.href}
                onClick={onClose}
                className={cn(
                  'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150',
                  isActive
                    ? 'text-white shadow-lg'
                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                )}
                style={
                  isActive
                    ? { backgroundColor: '#311B5B', boxShadow: '0 0 20px rgba(49, 27, 91, 0.5)' }
                    : {}
                }
              >
                <item.icon className="w-5 h-5 flex-shrink-0" />
                {item.name}
                {isActive && (
                  <div
                    className="ml-auto w-1.5 h-1.5 rounded-full"
                    style={{ backgroundColor: '#C59B47' }}
                  />
                )}
              </Link>
            );
          })}
        </nav>

        {/* Version info */}
        <div className="px-6 py-3 border-t border-purple-900/50">
          <p className="text-xs text-gray-500">IsoCortex Web UI v1.0</p>
        </div>
      </aside>
    </>
  );
}
