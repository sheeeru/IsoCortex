'use client';

import React from 'react';

export function Spinner({ size = 'md' }: { size?: 'sm' | 'md' | 'lg' }) {
  const sizes = { sm: 'w-4 h-4', md: 'w-6 h-6', lg: 'w-10 h-10' };
  return (
    <div
      className={`${sizes[size]} border-2 rounded-full animate-spin`}
      style={{ borderColor: '#3D2A5C', borderTopColor: '#C59B47' }}
    />
  );
}

export function PageSpinner() {
  return (
    <div className="flex items-center justify-center py-20">
      <Spinner size="lg" />
    </div>
  );
}

export function InlineSpinner() {
  return (
    <span className="inline-block">
      <Spinner size="sm" />
    </span>
  );
}
