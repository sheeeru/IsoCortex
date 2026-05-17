"""
IsoCortex — Scanner Tests
==========================
Tests for scan_directory, recursive scanning, ignore patterns,
file size limits, and format category mapping.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from isocortex.engine.ingestion.scanner import (
    ScannedFile,
    scan_directory,
    get_format_category,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def docs_dir(tmp_path: Path) -> Path:
    """Create a test directory with various files."""
    docs = tmp_path / "docs"
    docs.mkdir()

    (docs / "readme.md").write_text("# README\nSome content here.", encoding="utf-8")
    (docs / "notes.txt").write_text("Notes content", encoding="utf-8")
    (docs / "data.json").write_text('{"key": "value"}', encoding="utf-8")

    # Nested directory
    nested = docs / "subdir"
    nested.mkdir()
    (nested / "code.py").write_text("print('hello')", encoding="utf-8")

    # Unsupported file
    (docs / "image.xyz").write_text("binary", encoding="utf-8")

    return docs


@pytest.fixture()
def empty_dir(tmp_path: Path) -> Path:
    """Create an empty directory."""
    d = tmp_path / "empty"
    d.mkdir()
    return d


# =============================================================================
# Scan Tests
# =============================================================================

class TestScanDirectory:

    def test_scan_single_file(self, tmp_path: Path):
        """Scan a directory with one file."""
        d = tmp_path / "single"
        d.mkdir()
        (d / "file.txt").write_text("hello", encoding="utf-8")

        result = scan_directory(d)
        assert len(result.files) == 1
        assert result.files[0].extension == ".txt"
        assert result.files[0].format_category == "plain_text"
        assert result.summary.total_accepted == 1

    def test_scan_recursive(self, docs_dir: Path):
        """Scan nested directories."""
        result = scan_directory(docs_dir)
        extensions = {f.extension for f in result.files}
        assert ".md" in extensions
        assert ".txt" in extensions
        assert ".json" in extensions
        assert ".py" in extensions  # in subdir
        # .xyz should be skipped
        assert ".xyz" not in extensions

    def test_scan_ignore_patterns(self, tmp_path: Path):
        """Test .gitignore-like patterns."""
        root = tmp_path / "project"
        root.mkdir()
        (root / "keep.md").write_text("keep", encoding="utf-8")

        # These should be ignored
        git = root / ".git"
        git.mkdir()
        (git / "config").write_text("git config", encoding="utf-8")

        venv = root / "venv"
        venv.mkdir()
        (venv / "script.py").write_text("print()", encoding="utf-8")

        result = scan_directory(root)
        assert len(result.files) == 1
        assert result.files[0].extension == ".md"

    def test_scan_file_size_limit(self, tmp_path: Path):
        """Skip oversized files."""
        d = tmp_path / "bigfiles"
        d.mkdir()
        (d / "small.txt").write_text("small", encoding="utf-8")
        # Create a ~2MB file
        (d / "big.txt").write_bytes(b"x" * (2 * 1024 * 1024))

        result = scan_directory(d, max_file_size_mb=1.0)
        assert len(result.files) == 1
        assert result.files[0].absolute_path.name == "small.txt"

    def test_scan_empty_directory(self, empty_dir: Path):
        """Handle empty dir."""
        result = scan_directory(empty_dir)
        assert len(result.files) == 0
        assert result.summary.total_accepted == 0

    def test_scan_nonexistent_directory(self, tmp_path: Path):
        """Raise FileNotFoundError for missing dir."""
        with pytest.raises(FileNotFoundError):
            scan_directory(tmp_path / "does_not_exist")


# =============================================================================
# Format Category Mapping
# =============================================================================

class TestFormatCategory:

    def test_get_format_category(self):
        """Verify extension to category mapping."""
        assert get_format_category("report.pdf") == "pdf"
        assert get_format_category("script.py") == "source_code"
        assert get_format_category("data.xlsx") == "spreadsheet"
        assert get_format_category("notes.md") == "plain_text"
        assert get_format_category("deck.pptx") == "presentation"
        assert get_format_category("email.eml") == "email"
        assert get_format_category("page.html") == "web"
        assert get_format_category("table.csv") == "data"
        assert get_format_category("photo.png") == "image"
        assert get_format_category("unknown.xyz") is None
        assert get_format_category("noext") is None

    @pytest.mark.parametrize("ext,expected", [
        (".txt", "plain_text"),
        (".py", "source_code"),
        (".pdf", "pdf"),
        (".docx", "word"),
        (".pptx", "presentation"),
        (".xlsx", "spreadsheet"),
        (".csv", "data"),
        (".json", "data"),
        (".eml", "email"),
        (".html", "web"),
        (".png", "image"),
        (".jpg", "image"),
    ])
    def test_all_supported_extensions(self, ext: str, expected: str):
        """Verify all supported extensions map correctly."""
        assert get_format_category(f"file{ext}") == expected
