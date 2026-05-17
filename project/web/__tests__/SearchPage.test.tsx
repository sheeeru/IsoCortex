/**
 * IsoCortex — Search Page Tests
 * ===============================
 * Tests for the search page component.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import SearchPage from '@/app/search/page';
import { ToastProvider } from '@/components/ui/Toast';

// Mock the API module
jest.mock('@/lib/api', () => ({
  indexesApi: {
    list: jest.fn(),
  },
  searchApi: {
    search: jest.fn(),
  },
  ApiError: class extends Error {
    status: number;
    detail: string;
    constructor(status: number, detail: string) {
      super(detail);
      this.status = status;
      this.detail = detail;
      this.name = 'ApiError';
    }
  },
}));

// Need to import after mock to get references
const { indexesApi, searchApi } = jest.requireMock('@/lib/api');

function renderWithProvider(ui: React.ReactElement) {
  return render(<ToastProvider>{ui}</ToastProvider>);
}

describe('SearchPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('shows loading spinner while fetching indexes', () => {
    // Promise that never resolves to keep loading state
    indexesApi.list.mockReturnValue(new Promise(() => {}));

    renderWithProvider(<SearchPage />);
    expect(screen.getByText('Searching...') || document.querySelector('.animate-spin')).toBeTruthy();
  });

  it('renders search form after indexes load', async () => {
    indexesApi.list.mockResolvedValue([
      {
        name: 'test-index',
        dimension: 384,
        metric: 'cosine',
        document_count: 50,
        vector_count: 2500,
        size_bytes: 500000,
        status: 'ready',
        created_at: '2024-01-01',
        updated_at: '2024-01-02',
        config: { dimension: 384, metric: 'cosine' },
      },
    ]);

    await act(async () => {
      renderWithProvider(<SearchPage />);
    });

    await waitFor(() => {
      expect(screen.getByPlaceholderText('Enter your semantic search query...')).toBeInTheDocument();
    });

    expect(screen.getByText('Search Query')).toBeInTheDocument();
    expect(screen.getByText('Index')).toBeInTheDocument();
    expect(screen.getByText('Search')).toBeInTheDocument();
  });

  it('shows empty state when no indexes loaded', async () => {
    indexesApi.list.mockResolvedValue([]);

    await act(async () => {
      renderWithProvider(<SearchPage />);
    });

    await waitFor(() => {
      expect(screen.getByText('Semantic Search')).toBeInTheDocument();
    });

    expect(screen.getByText(/Enter a natural language query/)).toBeInTheDocument();
  });

  it('displays search results after successful search', async () => {
    indexesApi.list.mockResolvedValue([
      {
        name: 'test-index',
        dimension: 384,
        metric: 'cosine',
        document_count: 10,
        vector_count: 100,
        size_bytes: 200000,
        status: 'ready',
        created_at: '2024-01-01',
        updated_at: '2024-01-02',
        config: { dimension: 384, metric: 'cosine' },
      },
    ]);

    searchApi.search.mockResolvedValue({
      results: [
        {
          id: 'r1',
          content: 'This is a test result document with relevant content about neural search.',
          score: 0.92,
          metadata: { source: 'test.pdf' },
          chunk_index: 0,
          source: 'test.pdf',
        },
      ],
      query: 'neural search',
      total_results: 1,
      latency_ms: 3.5,
    });

    await act(async () => {
      renderWithProvider(<SearchPage />);
    });

    await waitFor(() => {
      expect(screen.getByPlaceholderText('Enter your semantic search query...')).toBeInTheDocument();
    });

    // Enter query
    const queryInput = screen.getByPlaceholderText('Enter your semantic search query...');
    await act(async () => {
      fireEvent.change(queryInput, { target: { value: 'neural search' } });
    });

    // Select index
    const select = screen.getByDisplayValue('Select an index...');
    await act(async () => {
      fireEvent.change(select, { target: { value: 'test-index' } });
    });

    // Submit search
    await act(async () => {
      fireEvent.click(screen.getByText('Search'));
    });

    await waitFor(() => {
      expect(screen.getByText('This is a test result document with relevant content about neural search.')).toBeInTheDocument();
    });

    expect(screen.getByText('92% match')).toBeInTheDocument();
    expect(screen.getByText(/3.5ms/)).toBeInTheDocument();
    expect(screen.getByText('#1')).toBeInTheDocument();
  });
});
