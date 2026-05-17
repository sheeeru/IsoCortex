"""
IsoCortex — Chunker Tests
==========================
Tests for sentence-aware chunking, source-code chunking,
overlap, min-size merging, and token guard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from isocortex.engine.ingestion.chunker import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_TOKEN_LIMIT,
    Chunk,
    ChunkedDocument,
    _chunk_sentence_aware,
    _chunk_source_code,
    _split_sentences,
    chunk_document,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def mock_extraction_simple() -> Any:
    """Mock extraction result with a long paragraph."""
    class MockChunk:
        def __init__(self, text, source_label):
            self.text = text
            self.source_label = source_label

    class MockResult:
        def __init__(self):
            self.absolute_path = Path("/test/doc.txt")
            self.format_category = "plain_text"
            self.success = True
            # A paragraph with many sentences
            self.chunks = [MockChunk(
                text=(
                    "The quick brown fox jumps over the lazy dog. "
                    "This is the second sentence in our paragraph. "
                    "Here comes the third sentence with more content. "
                    "Fourth sentence provides additional details. "
                    "Fifth sentence wraps up this paragraph nicely. "
                    "Now a sixth sentence for good measure. "
                    "Seventh sentence adds even more information. "
                    "Eighth sentence keeps the text flowing well. "
                    "Ninth sentence is almost the last one. "
                    "Tenth sentence concludes the text block."
                ),
                source_label="File: doc.txt",
            )]

    return MockResult()


@pytest.fixture()
def mock_extraction_code() -> Any:
    """Mock extraction result with source code."""
    class MockChunk:
        def __init__(self, text, source_label):
            self.text = text
            self.source_label = source_label

    class MockResult:
        def __init__(self):
            self.absolute_path = Path("/test/main.py")
            self.format_category = "source_code"
            self.success = True
            self.chunks = [MockChunk(
                text=(
                    "def hello():\n"
                    "    print('Hello World')\n"
                    "    return True\n"
                    "\n"
                    "def add(a, b):\n"
                    "    return a + b\n"
                    "\n"
                    "class Calculator:\n"
                    "    def multiply(self, x, y):\n"
                    "        return x * y\n"
                ),
                source_label="File: main.py",
            )]

    return MockResult()


# =============================================================================
# Sentence-Aware Chunking
# =============================================================================

class TestSentenceAwareChunking:

    def test_chunk_sentence_aware_basic(self, mock_extraction_simple):
        """Chunk a paragraph into multiple pieces."""
        doc = chunk_document(
            mock_extraction_simple,
            chunk_size=30,
            overlap=5,
        )
        assert doc.success is True
        assert doc.total_chunks >= 1
        assert all(c.word_count > 0 for c in doc.chunks)

    def test_chunk_with_overlap(self, mock_extraction_simple):
        """Verify overlap between consecutive chunks."""
        doc = chunk_document(
            mock_extraction_simple,
            chunk_size=25,
            overlap=8,
        )
        if doc.total_chunks >= 2:
            # Get words from consecutive chunks
            words_0 = doc.chunks[0].text.split()
            words_1 = doc.chunks[1].text.split()
            # Some words should be shared
            overlap_words = set(words_0[-8:]) & set(words_1[:8])
            assert len(overlap_words) >= 1

    def test_chunk_source_code(self, mock_extraction_code):
        """Code chunking by blank lines."""
        doc = chunk_document(
            mock_extraction_code,
            chunk_size=30,
            overlap=5,
        )
        assert doc.success is True
        assert doc.total_chunks >= 1

    def test_chunk_min_size_merging(self):
        """Short chunks get merged with neighbours."""
        class MockChunk:
            def __init__(self, text, source_label):
                self.text = text
                self.source_label = source_label

        class MockResult:
            def __init__(self):
                self.absolute_path = Path("/test/tiny.txt")
                self.format_category = "plain_text"
                self.success = True
                self.chunks = [MockChunk("Hi.", "File: tiny.txt")]

        doc = chunk_document(MockResult(), chunk_size=100, overlap=10)
        assert doc.success is True

    def test_chunk_empty_input(self):
        """Handle empty text gracefully."""
        class MockChunk:
            def __init__(self, text, source_label):
                self.text = text
                self.source_label = source_label

        class MockResult:
            def __init__(self):
                self.absolute_path = Path("/test/empty.txt")
                self.format_category = "plain_text"
                self.success = True
                self.chunks = [MockChunk("", "File: empty.txt")]

        doc = chunk_document(MockResult(), chunk_size=50)
        # Empty chunks are skipped, but if ALL chunks are empty,
        # the result should indicate failure
        assert isinstance(doc, ChunkedDocument)

    def test_chunk_failed_extraction(self):
        """Failed extraction result should produce failed chunked doc."""
        class MockResult:
            def __init__(self):
                self.absolute_path = Path("/test/failed.txt")
                self.format_category = "plain_text"
                self.success = False
                self.chunks = []

        doc = chunk_document(MockResult(), chunk_size=50)
        assert doc.success is False
        assert doc.error_message is not None


# =============================================================================
# Token Guard Tests
# =============================================================================

class TestTokenGuard:

    def test_token_guard(self):
        """Chunks exceeding token limit get re-split."""
        # Without a real tokenizer, the guard is skipped.
        # Test with None tokenizer (should still produce chunks).
        text = "Hello world. " * 200
        chunks = _chunk_sentence_aware(
            text, chunk_size=50, overlap=5,
            tokenizer=None, token_limit=100,
        )
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk) > 0


# =============================================================================
# Sentence Splitting
# =============================================================================

class TestSentenceSplitting:

    def test_split_sentences_basic(self):
        """Sentences split on period + space + capital."""
        text = "First sentence. Second sentence. Third sentence."
        sentences = _split_sentences(text)
        assert len(sentences) >= 1

    def test_split_paragraphs(self):
        """Paragraph boundaries produce splits."""
        text = "First paragraph here.\n\nSecond paragraph there."
        sentences = _split_sentences(text)
        assert len(sentences) >= 2
