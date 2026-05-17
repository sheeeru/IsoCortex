// IsoCortex API TypeScript interfaces

// Auth
export interface LoginRequest {
  username: string;
  password: string;
}

export interface SetupRequest {
  username: string;
  password: string;
  email?: string;
}

export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface User {
  id: string;
  username: string;
  email?: string;
  role: 'admin' | 'user';
  is_active: boolean;
  created_at: string;
  last_login?: string;
}

export interface CreateUserRequest {
  username: string;
  password: string;
  email?: string;
  role?: 'admin' | 'user';
}

// Indexes
export interface IndexConfig {
  dimension: number;
  metric: 'cosine' | 'l2' | 'ip';
  M?: number;
  ef_construction?: number;
  ef_search?: number;
}

export interface Index {
  name: string;
  dimension: number;
  metric: string;
  document_count: number;
  vector_count: number;
  size_bytes: number;
  status: 'ready' | 'building' | 'error';
  created_at: string;
  updated_at: string;
  config: IndexConfig;
}

export interface CreateIndexRequest {
  name: string;
  dimension: number;
  metric?: 'cosine' | 'l2' | 'ip';
  M?: number;
  ef_construction?: number;
}

export interface UpdateIndexRequest {
  ef_search?: number;
}

// Documents
export interface Document {
  id: string;
  index_name: string;
  content: string;
  metadata: Record<string, unknown>;
  vector?: number[];
  chunk_index: number;
  source?: string;
  created_at: string;
  updated_at: string;
}

export interface PaginatedDocuments {
  documents: Document[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

// Search
export interface SearchRequest {
  query: string;
  top_k?: number;
  filters?: Record<string, unknown>;
  include_metadata?: boolean;
}

export interface SearchResult {
  id: string;
  content: string;
  score: number;
  metadata: Record<string, unknown>;
  chunk_index: number;
  source?: string;
}

export interface SearchResponse {
  results: SearchResult[];
  query: string;
  total_results: number;
  latency_ms: number;
}

// Jobs
export interface Job {
  id: string;
  type: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error?: string;
}

export interface JobCreatedResponse {
  job_id: string;
  message: string;
}

// Admin / Analytics
export interface SystemStats {
  total_indexes: number;
  total_vectors: number;
  total_documents: number;
  total_users: number;
  total_searches: number;
  uptime_seconds: number;
  memory_usage_bytes: number;
  disk_usage_bytes: number;
}

export interface RateLimitStatus {
  enabled: boolean;
  requests_per_minute: number;
  burst_size: number;
  current_usage: {
    requests: number;
    remaining: number;
    reset_at: string;
  };
}

export interface SearchStats {
  total_queries: number;
  avg_latency_ms: number;
  top_queries: { query: string; count: number }[];
  queries_over_time: { timestamp: string; count: number }[];
}

// Health
export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  uptime: number;
  components: {
    database: string;
    search_engine: string;
    embedding: string;
  };
}
