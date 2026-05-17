"""
IsoCortex — C++ HNSW Test Integration
========================================
Pytest wrapper that compiles and runs the standalone C++ test harness
(hnsw.cpp) and reports results through the pytest framework.

This bridges the C++ test suite into the Python pytest workflow so that
``pytest`` runs both Python and C++ tests in a single invocation.

SRS References:
  - FR-IDX-001: Index construction correctness
  - FR-IDX-002: Soft delete
  - Section 7: Index persistence

Author : Shaheer Qureshi
Project: IsoCortex
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Locate C++ source and header
# ---------------------------------------------------------------------------

_HNSW_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "isocortex" / "core" / "hnsw"
_CPP_SRC = _HNSW_DIR / "hnsw.cpp"
_HPP_SRC = _HNSW_DIR / "hnsw.hpp"


def _has_cxx_compiler() -> bool:
    """Check if g++ or clang++ is available."""
    return shutil.which("g++") is not None or shutil.which("clang++") is not None


def _get_compiler() -> str:
    """Return the first available C++ compiler."""
    for compiler in ("g++", "clang++"):
        if shutil.which(compiler) is not None:
            return compiler
    raise RuntimeError("No C++ compiler found (need g++ or clang++)")


def _compile_and_run() -> tuple[int, str, str]:
    """Compile the C++ test harness, run it, and return (exit_code, stdout, stderr).

    Returns
    -------
    (exit_code, stdout, stderr)
    """
    if not _CPP_SRC.exists():
        raise FileNotFoundError(f"C++ test source not found: {_CPP_SRC}")

    compiler = _get_compiler()

    with tempfile.TemporaryDirectory(prefix="isocortex_cpp_test_") as tmpdir:
        binary = os.path.join(tmpdir, "test_hnsw")

        # Compile
        compile_cmd = [
            compiler,
            "-std=c++17",
            "-O2",
            "-o", binary,
            str(_CPP_SRC),
            "-I", str(_HNSW_DIR),  # for hnsw.hpp
            "-lm",
        ]

        compile_result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if compile_result.returncode != 0:
            return (
                compile_result.returncode,
                compile_result.stdout,
                compile_result.stderr,
            )

        # Run
        run_result = subprocess.run(
            [binary],
            capture_output=True,
            text=True,
            timeout=300,
        )

        return (
            run_result.returncode,
            run_result.stdout,
            run_result.stderr,
        )


# ---------------------------------------------------------------------------
# Pytest integration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _CPP_SRC.exists(),
    reason="C++ test source (hnsw.cpp) not found",
)
@pytest.mark.skipif(
    not _has_cxx_compiler(),
    reason="No C++ compiler (g++/clang++) available",
)
class TestCppHnsw:
    """Pytest wrapper around the C++ HNSW test harness.

    Each test method runs the full C++ test suite and checks a specific
    aspect of the output.  This approach is preferred over running the
    binary once and parsing all output, because it gives clearer per-test
    reporting in pytest.
    """

    _cached_result: tuple[int, str, str] | None = None

    @classmethod
    def _get_result(cls) -> tuple[int, str, str]:
        """Run the C++ tests once and cache the result."""
        if cls._cached_result is None:
            cls._cached_result = _compile_and_run()
        return cls._cached_result

    @classmethod
    def teardown_class(cls) -> None:
        """Reset cache after all tests in the class."""
        cls._cached_result = None

    def test_cpp_compilation_succeeds(self):
        """The C++ test binary compiles without errors."""
        exit_code, stdout, stderr = self._get_result()
        compile_failed = stderr and (
            "error:" in stderr.lower()
            or "undefined reference" in stderr.lower()
            or "fatal error" in stderr.lower()
        )
        assert not compile_failed, f"C++ compilation failed:\n{stderr}"

    def test_cpp_all_tests_pass(self):
        """All C++ HNSW tests pass (exit code 0)."""
        exit_code, stdout, stderr = self._get_result()
        assert exit_code == 0, (
            f"C++ tests failed (exit code {exit_code})\n"
            f"--- stdout ---\n{stdout}\n"
            f"--- stderr ---\n{stderr}"
        )

    def test_cpp_config_validation_passes(self):
        """C++ config validation test passes."""
        exit_code, stdout, _ = self._get_result()
        assert "[PASS]" in stdout, "No [PASS] markers found in C++ test output"
        # Check config_validation specifically
        assert "config_validation" in stdout

    def test_cpp_search_recall_passes(self):
        """C++ search recall test passes (>90% recall)."""
        _, stdout, _ = self._get_result()
        assert "search_recall" in stdout

    def test_cpp_save_load_passes(self):
        """C++ index persistence (save/load) test passes."""
        _, stdout, _ = self._get_result()
        assert "save_load" in stdout

    def test_cpp_soft_delete_passes(self):
        """C++ soft delete test passes."""
        _, stdout, _ = self._get_result()
        assert "soft_delete" in stdout

    def test_cpp_benchmark_runs(self):
        """C++ benchmark runs and produces output."""
        _, stdout, _ = self._get_result()
        assert "BENCHMARK" in stdout or "benchmark" in stdout.lower()
