/**
 * IsoCortex — API Client Tests
 * ==============================
 * Tests for the API client module (lib/api.ts).
 */

import { ApiError, authApi, indexesApi, searchApi, healthApi } from '@/lib/api';
import { setTokens, clearTokens } from '@/lib/api';

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

// Reset localStorage and fetch between tests
beforeEach(() => {
  jest.clearAllMocks();
  clearTokens();
  (window.localStorage.getItem as jest.Mock).mockReturnValue(null);
});

describe('ApiError', () => {
  it('creates error with status and detail', () => {
    const err = new ApiError(404, 'Not found');
    expect(err).toBeInstanceOf(Error);
    expect(err.status).toBe(404);
    expect(err.detail).toBe('Not found');
    expect(err.name).toBe('ApiError');
    expect(err.message).toBe('Not found');
  });
});

describe('authApi', () => {
  it('setup sends POST to /auth/setup', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 201,
      json: async () => ({
        access_token: 'at_123',
        refresh_token: 'rt_456',
        token_type: 'bearer',
        expires_in: 86400,
      }),
    });

    const result = await authApi.setup({
      username: 'admin',
      password: 'SecurePassword123!',
      email: 'admin@test.com',
    });

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url, opts] = mockFetch.mock.calls[0];
    expect(url).toContain('/api/v1/auth/setup');
    expect(opts.method).toBe('POST');
    expect(result.access_token).toBe('at_123');
    expect(result.refresh_token).toBe('rt_456');
  });

  it('login sends POST to /auth/login', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        access_token: 'at_token',
        refresh_token: 'rt_token',
        token_type: 'bearer',
        expires_in: 86400,
        user: {
          id: 'u1',
          username: 'admin',
          role: 'admin',
          is_active: true,
          created_at: '2024-01-01',
        },
      }),
    });

    const result = await authApi.login({
      username: 'admin',
      password: 'password123',
    });

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url] = mockFetch.mock.calls[0];
    expect(url).toContain('/api/v1/auth/login');
    expect(result.access_token).toBe('at_token');
  });

  it('login throws ApiError on 401', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      json: async () => ({
        detail: 'Invalid credentials',
      }),
    });

    await expect(authApi.login({
      username: 'admin',
      password: 'wrong',
    })).rejects.toThrow(ApiError);
  });
});

describe('indexesApi', () => {
  it('list sends GET to /indexes with auth header', async () => {
    setTokens('my_access_token', 'my_refresh_token');

    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => [
        {
          name: 'test-index',
          dimension: 384,
          metric: 'cosine',
          document_count: 100,
          vector_count: 5000,
          size_bytes: 1024000,
          status: 'ready',
          created_at: '2024-01-01',
          updated_at: '2024-01-02',
          config: { dimension: 384, metric: 'cosine' },
        },
      ],
    });

    const result = await indexesApi.list();

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url, opts] = mockFetch.mock.calls[0];
    expect(url).toContain('/api/v1/indexes');
    expect(opts.headers['Authorization']).toBe('Bearer my_access_token');
    expect(result).toHaveLength(1);
    expect(result[0].name).toBe('test-index');
  });

  it('create sends POST with body', async () => {
    setTokens('at', 'rt');
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 202,
      json: async () => ({ job_id: 'job-1', message: 'Index creation started' }),
    });

    const result = await indexesApi.create({
      name: 'new-index',
      dimension: 384,
      metric: 'cosine',
    });

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url, opts] = mockFetch.mock.calls[0];
    expect(url).toContain('/api/v1/indexes');
    expect(opts.method).toBe('POST');
    expect(result.job_id).toBe('job-1');
  });
});

describe('searchApi', () => {
  it('search sends POST with query', async () => {
    setTokens('at', 'rt');
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        results: [
          {
            id: 'r1',
            content: 'Test result',
            score: 0.95,
            metadata: {},
            chunk_index: 0,
          },
        ],
        query: 'test query',
        total_results: 1,
        latency_ms: 5.2,
      }),
    });

    const result = await searchApi.search('my-index', {
      query: 'test query',
      top_k: 10,
    });

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url, opts] = mockFetch.mock.calls[0];
    expect(url).toContain('/api/v1/indexes/my-index/search');
    expect(opts.method).toBe('POST');
    expect(result.results).toHaveLength(1);
    expect(result.results[0].score).toBe(0.95);
    expect(result.latency_ms).toBe(5.2);
  });
});

describe('healthApi', () => {
  it('check fetches /health without auth', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        status: 'healthy',
        version: '1.0.0',
        uptime: 3600,
        components: {
          database: 'ok',
          search_engine: 'ok',
          embedding: 'ok',
        },
      }),
    });

    const result = await healthApi.check();

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url] = mockFetch.mock.calls[0];
    expect(url).toContain('/health');
    expect(result.status).toBe('healthy');
  });
});
