"""OCT Converter Service Layer.

This package provides a high-level service API for OCT file conversion,
built on top of the oct_converter_app application layer.

Example usage:
    >>> from oct_converter_service import ConversionService, ConversionRequest
    >>> service = ConversionService()
    >>> request = ConversionRequest(
    ...     input_path="scan.fds",
    ...     output_dir="./output",
    ...     outputs=["dicom", "npy", "images", "metadata"]
    ... )
    >>> result = service.convert(request)
    >>> print(result.success)
"""

from oct_converter_service.service import ConversionService
from oct_converter_service.requests import ConversionRequest
from oct_converter_service.results import ConversionResult
from oct_converter_service.config import ConversionConfig
from oct_converter_service.batch import BatchConversionService
from oct_converter_service.errors import (
    ConversionServiceError,
    InvalidRequestError,
    InputNotFoundError,
    UnsupportedFormatError,
    ConversionFailedError,
    OutputError,
    OverwriteNotAllowedError,
)

__all__ = [
    # Core service classes
    "ConversionService",
    "ConversionRequest",
    "ConversionResult",
    "ConversionConfig",
    "BatchConversionService",
    # Exceptions
    "ConversionServiceError",
    "InvalidRequestError",
    "InputNotFoundError",
    "UnsupportedFormatError",
    "ConversionFailedError",
    "OutputError",
    "OverwriteNotAllowedError",
]
