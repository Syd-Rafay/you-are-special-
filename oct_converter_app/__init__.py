"""OCT Converter Application Layer.

This package provides the application/orchestration layer for OCT file conversion.
It wraps the existing oct_converter library to provide:
- Format detection
- Reader factory/registry
- Processing pipeline
- Multiple output exporters (DICOM, NPY, images, metadata)
- CLI interface

The existing oct_converter library remains unchanged and is used as the low-level
extraction engine.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .detector import FormatDetector
from .factory import ReaderFactory
from .models import OCTStudy
from .pipeline import ProcessingPipeline

__all__ = [
    "FormatDetector",
    "ReaderFactory", 
    "OCTStudy",
    "ProcessingPipeline",
    "__version__",
]
