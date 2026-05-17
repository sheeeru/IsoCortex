/**
 * IsoCortex — Utility Functions Tests
 * =====================================
 * Tests for pure utility functions in lib/utils.ts.
 */

import { formatBytes, formatNumber, formatUptime, formatDate, timeAgo, cn, truncate } from '@/lib/utils';

describe('formatBytes', () => {
  it('returns "0 B" for zero bytes', () => {
    expect(formatBytes(0)).toBe('0 B');
  });

  it('formats bytes correctly', () => {
    expect(formatBytes(500)).toBe('500 B');
    expect(formatBytes(1024)).toBe('1 KB');
    expect(formatBytes(1536)).toBe('1.5 KB');
    expect(formatBytes(1048576)).toBe('1 MB');
    expect(formatBytes(1073741824)).toBe('1 GB');
    expect(formatBytes(1099511627776)).toBe('1 TB');
  });

  it('respects decimal parameter', () => {
    expect(formatBytes(1536, 0)).toBe('2 KB');
    expect(formatBytes(1234567, 3)).toBe('1.193 MB');
  });
});

describe('formatNumber', () => {
  it('formats small numbers with locale', () => {
    expect(formatNumber(42)).toBe('42');
    expect(formatNumber(999)).toBe('999');
  });

  it('formats thousands', () => {
    expect(formatNumber(1000)).toBe('1.0K');
    expect(formatNumber(5500)).toBe('5.5K');
    expect(formatNumber(999999)).toBe('1000.0K');
  });

  it('formats millions', () => {
    expect(formatNumber(1000000)).toBe('1.0M');
    expect(formatNumber(2500000)).toBe('2.5M');
  });
});

describe('formatUptime', () => {
  it('formats seconds only', () => {
    expect(formatUptime(45)).toBe('0m');
  });

  it('formats minutes and seconds', () => {
    expect(formatUptime(125)).toBe('2m');
  });

  it('formats hours and minutes', () => {
    expect(formatUptime(3725)).toBe('1h 2m');
  });

  it('formats days, hours, and minutes', () => {
    expect(formatUptime(90125)).toBe('1d 1h 2m');
  });

  it('handles zero', () => {
    expect(formatUptime(0)).toBe('0m');
  });
});

describe('formatDate', () => {
  it('formats ISO date string', () => {
    const result = formatDate('2024-03-15T10:30:00Z');
    // Check it contains expected parts (locale-dependent)
    expect(result).toContain('2024');
  });

  it('handles various date formats', () => {
    const result = formatDate('2025-01-01T00:00:00Z');
    expect(result).toContain('2025');
  });
});

describe('timeAgo', () => {
  it('returns "just now" for recent times', () => {
    const now = new Date().toISOString();
    expect(timeAgo(now)).toBe('just now');
  });

  it('returns minutes ago', () => {
    const fiveMinAgo = new Date(Date.now() - 5 * 60 * 1000).toISOString();
    expect(timeAgo(fiveMinAgo)).toBe('5m ago');
  });

  it('returns hours ago', () => {
    const threeHoursAgo = new Date(Date.now() - 3 * 3600 * 1000).toISOString();
    expect(timeAgo(threeHoursAgo)).toBe('3h ago');
  });

  it('returns days ago', () => {
    const twoDaysAgo = new Date(Date.now() - 2 * 86400 * 1000).toISOString();
    expect(timeAgo(twoDaysAgo)).toBe('2d ago');
  });
});

describe('cn', () => {
  it('joins class names', () => {
    expect(cn('a', 'b', 'c')).toBe('a b c');
  });

  it('filters falsy values', () => {
    expect(cn('a', false, null, undefined, 'b')).toBe('a b');
  });

  it('handles empty input', () => {
    expect(cn()).toBe('');
    expect(cn(false, null)).toBe('');
  });
});

describe('truncate', () => {
  it('returns string as-is when within limit', () => {
    expect(truncate('hello', 10)).toBe('hello');
  });

  it('truncates long strings', () => {
    expect(truncate('hello world', 5)).toBe('hello...');
  });

  it('handles exact length', () => {
    expect(truncate('hello', 5)).toBe('hello');
  });

  it('handles empty string', () => {
    expect(truncate('', 10)).toBe('');
  });
});
