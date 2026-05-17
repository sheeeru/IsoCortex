/**
 * IsoCortex — Indexes Page Tests
 * ================================
 * Tests for the indexes list page component.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import IndexesPage from '@/app/indexes/page';
import { ToastProvider } from '@/components/ui/Toast';

// Mock Next.js Link
jest.mock('next/link', () => {
  return function MockLink({ children, href, ...props }: { children: React.ReactNode; href: string; [key: string]: unknown }) {
    return <a href={href} {...props}>{children}</a>;
  };
});

// Mock the API module
jest.mock('@/lib/api', () => ({
  indexesApi: {
    list: jest.fn(),
    create: jest.fn(),
    delete: jest.fn(),
  },
  jobsApi: {
    get: jest.fn(),
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

const { indexesApi, jobsApi } = jest.requireMock('@/lib/api');

function renderWithProvider(ui: React.ReactElement) {
  return render(<ToastProvider>{ui}</ToastProvider>);
}

describe('IndexesPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('shows loading spinner while fetching indexes', () => {
    indexesApi.list.mockReturnValue(new Promise(() => {}));

    renderWithProvider(<IndexesPage />);
    expect(document.querySelector('.animate-spin')).toBeTruthy();
  });

  it('renders empty state when no indexes exist', async () => {
    indexesApi.list.mockResolvedValue([]);

    await act(async () => {
      renderWithProvider(<IndexesPage />);
    });

    await waitFor(() => {
      expect(screen.getByText('No indexes yet')).toBeInTheDocument();
    });

    expect(screen.getByText('Create your first semantic index to get started')).toBeInTheDocument();
    expect(screen.getByText('Create Index')).toBeInTheDocument();
  });

  it('renders index cards when indexes exist', async () => {
    indexesApi.list.mockResolvedValue([
      {
        name: 'docs-index',
        dimension: 384,
        metric: 'cosine',
        document_count: 150,
        vector_count: 7500,
        size_bytes: 1500000,
        status: 'ready',
        created_at: '2024-01-01',
        updated_at: '2024-06-15T10:30:00Z',
        config: { dimension: 384, metric: 'cosine' },
      },
      {
        name: 'code-index',
        dimension: 768,
        metric: 'l2',
        document_count: 300,
        vector_count: 15000,
        size_bytes: 3000000,
        status: 'building',
        created_at: '2024-02-01',
        updated_at: '2024-06-14T08:00:00Z',
        config: { dimension: 768, metric: 'l2' },
      },
    ]);

    await act(async () => {
      renderWithProvider(<IndexesPage />);
    });

    await waitFor(() => {
      expect(screen.getByText('docs-index')).toBeInTheDocument();
    });

    expect(screen.getByText('code-index')).toBeInTheDocument();
    expect(screen.getByText('2 indexes found')).toBeInTheDocument();

    // Check status indicators
    expect(screen.getByText('ready')).toBeInTheDocument();
    expect(screen.getByText('building')).toBeInTheDocument();

    // Check that index cards link to detail pages
    const links = screen.getAllByRole('link');
    expect(links.some(link => link.getAttribute('href') === '/indexes/docs-index')).toBe(true);
    expect(links.some(link => link.getAttribute('href') === '/indexes/code-index')).toBe(true);
  });

  it('opens create modal when Create Index is clicked', async () => {
    indexesApi.list.mockResolvedValue([]);

    await act(async () => {
      renderWithProvider(<IndexesPage />);
    });

    await waitFor(() => {
      expect(screen.getByText('No indexes yet')).toBeInTheDocument();
    });

    await act(async () => {
      fireEvent.click(screen.getByText('Create Index'));
    });

    expect(screen.getByText('Create New Index')).toBeInTheDocument();
    expect(screen.getByText('Index Name')).toBeInTheDocument();
    expect(screen.getByText('Dimensions')).toBeInTheDocument();
    expect(screen.getByText('Distance Metric')).toBeInTheDocument();
    expect(screen.getByText('Cancel')).toBeInTheDocument();
  });

  it('creates index and shows job toast on success', async () => {
    indexesApi.list.mockResolvedValue([]);
    indexesApi.create.mockResolvedValue({
      job_id: 'job-123',
      message: 'Index creation started',
    });
    jobsApi.get.mockResolvedValue({
      id: 'job-123',
      type: 'index_create',
      status: 'completed',
      progress: 100,
      message: 'Done',
      created_at: '2024-01-01',
    });

    await act(async () => {
      renderWithProvider(<IndexesPage />);
    });

    await waitFor(() => {
      expect(screen.getByText('No indexes yet')).toBeInTheDocument();
    });

    // Open create modal
    await act(async () => {
      fireEvent.click(screen.getByText('Create Index'));
    });

    // Fill in the form
    const nameInput = screen.getByPlaceholderText('my-index');
    await act(async () => {
      fireEvent.change(nameInput, { target: { value: 'test-idx' } });
    });

    // Submit
    await act(async () => {
      fireEvent.click(screen.getByText('Create Index'));
    });

    // Wait for API call
    await waitFor(() => {
      expect(indexesApi.create).toHaveBeenCalledWith({
        name: 'test-idx',
        dimension: 768,
        metric: 'cosine',
        ef_construction: 128,
        M: 16,
      });
    });
  });
});
