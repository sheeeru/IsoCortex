// IsoCortex API client with JWT handling

import {
  LoginRequest,
  SetupRequest,
  AuthResponse,
  User,
  CreateUserRequest,
  Index,
  CreateIndexRequest,
  UpdateIndexRequest,
  Document,
  PaginatedDocuments,
  SearchRequest,
  SearchResponse,
  Job,
  JobCreatedResponse,
  SystemStats,
  RateLimitStatus,
  HealthResponse,
} from './types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8900';
const REQUEST_TIMEOUT_MS = 30000;

class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
    this.name = 'ApiError';
  }
}

function getToken(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem('access_token');
}

function getRefreshToken(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem('refresh_token');
}

function setTokens(access: string, refresh: string) {
  localStorage.setItem('access_token', access);
  localStorage.setItem('refresh_token', refresh);
}

function clearTokens() {
  localStorage.removeItem('access_token');
  localStorage.removeItem('refresh_token');
}

// Prevent concurrent refresh requests
let refreshPromise: Promise<boolean> | null = null;

async function refreshAccessToken(): Promise<boolean> {
  // If a refresh is already in flight, reuse it
  if (refreshPromise) return refreshPromise;

  const refresh = getRefreshToken();
  if (!refresh) return false;

  refreshPromise = (async () => {
    try {
      const res = await fetch(`${API_URL}/api/v1/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refresh }),
      });

      if (!res.ok) {
        clearTokens();
        return false;
      }

      const data: AuthResponse = await res.json();
      setTokens(data.access_token, data.refresh_token);
      return true;
    } catch {
      clearTokens();
      return false;
    } finally {
      refreshPromise = null;
    }
  })();

  return refreshPromise;
}

async function apiFetch<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    ...((options.headers as Record<string, string>) || {}),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  if (!headers['Content-Type'] && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }

  // Add timeout via AbortController
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const mergedOptions: RequestInit = {
    ...options,
    headers,
    signal: controller.signal,
  };

  let res = await fetch(`${API_URL}${endpoint}`, mergedOptions);
  clearTimeout(timeoutId);

  // Try refresh on 401
  if (res.status === 401 && token) {
    const refreshed = await refreshAccessToken();
    if (refreshed) {
      const newToken = getToken();
      if (newToken) {
        headers['Authorization'] = `Bearer ${newToken}`;
      }
      const retryController = new AbortController();
      const retryTimeout = setTimeout(() => retryController.abort(), REQUEST_TIMEOUT_MS);
      res = await fetch(`${API_URL}${endpoint}`, {
        ...options,
        headers,
        signal: retryController.signal,
      });
      clearTimeout(retryTimeout);
    } else {
      // Redirect to login
      if (typeof window !== 'undefined' && !window.location.pathname.includes('/login')) {
        window.location.href = '/login';
      }
      throw new ApiError(401, 'Session expired. Please log in again.');
    }
  }

  if (!res.ok) {
    let detail = `Request failed with status ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || body.message || detail;
    } catch {
      // ignore
    }
    throw new ApiError(res.status, detail);
  }

  // Handle 204 No Content
  if (res.status === 204) {
    return undefined as unknown as T;
  }

  return res.json();
}

// Auth API
export const authApi = {
  setup: (data: SetupRequest) =>
    apiFetch<AuthResponse>('/api/v1/auth/setup', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  login: (data: LoginRequest) =>
    apiFetch<AuthResponse>('/api/v1/auth/login', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  me: () => apiFetch<User>('/api/v1/auth/me'),

  // Admin user management
  listUsers: () => apiFetch<User[]>('/api/v1/auth/users/'),

  createUser: (data: CreateUserRequest) =>
    apiFetch<User>('/api/v1/auth/users/', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  deleteUser: (userId: string) =>
    apiFetch<void>(`/api/v1/auth/users/${userId}`, {
      method: 'DELETE',
    }),
};

// Indexes API
export const indexesApi = {
  list: () => apiFetch<Index[]>('/api/v1/indexes'),

  get: (name: string) => apiFetch<Index>(`/api/v1/indexes/${encodeURIComponent(name)}`),

  create: (data: CreateIndexRequest) =>
    apiFetch<JobCreatedResponse>('/api/v1/indexes', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  update: (name: string, data: UpdateIndexRequest) =>
    apiFetch<Index>(`/api/v1/indexes/${encodeURIComponent(name)}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  delete: (name: string) =>
    apiFetch<JobCreatedResponse>(`/api/v1/indexes/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    }),

  exportIndex: (name: string) =>
    apiFetch<JobCreatedResponse>(`/api/v1/indexes/${encodeURIComponent(name)}/export`, {
      method: 'POST',
    }),

  importIndex: (formData: FormData) =>
    apiFetch<JobCreatedResponse>('/api/v1/indexes/import', {
      method: 'POST',
      body: formData,
    }),
};

// Search API
export const searchApi = {
  search: (indexName: string, data: SearchRequest) =>
    apiFetch<SearchResponse>(`/api/v1/indexes/${encodeURIComponent(indexName)}/search`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  batchSearch: (indexName: string, queries: SearchRequest[]) =>
    apiFetch<SearchResponse[]>(`/api/v1/indexes/${encodeURIComponent(indexName)}/search/batch`, {
      method: 'POST',
      body: JSON.stringify({ queries }),
    }),
};

// Documents API
export const documentsApi = {
  list: (indexName: string, page = 1, pageSize = 20) =>
    apiFetch<PaginatedDocuments>(
      `/api/v1/indexes/${encodeURIComponent(indexName)}/documents?page=${page}&page_size=${pageSize}`
    ),

  get: (indexName: string, docId: string) =>
    apiFetch<Document>(
      `/api/v1/indexes/${encodeURIComponent(indexName)}/documents/${encodeURIComponent(docId)}`
    ),

  delete: (indexName: string, docId: string) =>
    apiFetch<void>(
      `/api/v1/indexes/${encodeURIComponent(indexName)}/documents/${encodeURIComponent(docId)}`,
      { method: 'DELETE' }
    ),
};

// Jobs API
export const jobsApi = {
  get: (jobId: string) => apiFetch<Job>(`/api/v1/jobs/${encodeURIComponent(jobId)}`),

  stream: (jobId: string): EventSource => {
    // EventSource doesn't support custom headers, so pass token as query param
    const token = getToken();
    const url = token
      ? `${API_URL}/api/v1/jobs/${encodeURIComponent(jobId)}/stream?token=${encodeURIComponent(token)}`
      : `${API_URL}/api/v1/jobs/${encodeURIComponent(jobId)}/stream`;
    return new EventSource(url);
  },
};

// Admin API
export const adminApi = {
  getStats: () => apiFetch<SystemStats>('/api/v1/admin/stats'),

  getRateLimits: () => apiFetch<RateLimitStatus>('/api/v1/admin/rate-limits'),
};

// Health API (no auth)
export const healthApi = {
  check: () => fetch(`${API_URL}/health`).then(res => res.json() as Promise<HealthResponse>),
};

export { ApiError, setTokens, clearTokens, getToken };
export default apiFetch;
