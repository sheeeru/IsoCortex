"""
IsoCortex — ingestion package
==============================
Exposes the public API of the ingestion pipeline.

Uses relative imports so this package works correctly whether it is
imported as 'ingestion.scanner' from the project root, or as a
standalone package — without any sys.path manipulation.
"""

__version__ = "0.1.0"
__author__  = "Shaheer Qureshi"

from .scanner import (
    ScannedFile,
    ScanResult,
    ScanSummary,
    scan_directory,
    iter_scan_directory,
    is_supported,
    get_format_category,
    get_supported_extensions,
)

from .extractor import (
    ExtractedChunk,
    ExtractionResult,
    extract,
    extract_batch,
)

__all__ = [
    "__version__",
    "__author__",
    # scanner
    "ScannedFile",
    "ScanResult",
    "ScanSummary",
    "scan_directory",
    "iter_scan_directory",
    "is_supported",
    "get_format_category",
    "get_supported_extensions",
    # extractor
    "ExtractedChunk",
    "ExtractionResult",
    "extract",
    "extract_batch",
]