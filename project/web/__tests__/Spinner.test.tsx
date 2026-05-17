/**
 * IsoCortex — Spinner Component Tests
 * ======================================
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import { Spinner, PageSpinner, InlineSpinner } from '@/components/ui/Spinner';

describe('Spinner', () => {
  it('renders with default medium size', () => {
    const { container } = render(<Spinner />);
    const el = container.firstChild as HTMLElement;
    expect(el).toBeInTheDocument();
    expect(el.className).toContain('w-6');
    expect(el.className).toContain('h-6');
    expect(el.className).toContain('animate-spin');
    expect(el.className).toContain('rounded-full');
  });

  it('renders small spinner', () => {
    const { container } = render(<Spinner size="sm" />);
    const el = container.firstChild as HTMLElement;
    expect(el.className).toContain('w-4');
    expect(el.className).toContain('h-4');
  });

  it('renders large spinner', () => {
    const { container } = render(<Spinner size="lg" />);
    const el = container.firstChild as HTMLElement;
    expect(el.className).toContain('w-10');
    expect(el.className).toContain('h-10');
  });
});

describe('PageSpinner', () => {
  it('renders centered spinner with padding', () => {
    const { container } = render(<PageSpinner />);
    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper.className).toContain('flex');
    expect(wrapper.className).toContain('items-center');
    expect(wrapper.className).toContain('justify-center');
    // Should contain the spinner child
    expect(wrapper.querySelector('.animate-spin')).toBeInTheDocument();
  });
});

describe('InlineSpinner', () => {
  it('renders inline spinner', () => {
    const { container } = render(<InlineSpinner />);
    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper.tagName).toBe('SPAN');
    expect(wrapper.className).toContain('inline-block');
    expect(wrapper.querySelector('.animate-spin')).toBeInTheDocument();
  });
});
