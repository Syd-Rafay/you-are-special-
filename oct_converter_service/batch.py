"""Batch conversion service for processing multiple OCT files."""

from typing import Iterator

from oct_converter_service.service import ConversionService
from oct_converter_service.requests import ConversionRequest
from oct_converter_service.results import ConversionResult
from oct_converter_service.errors import ConversionServiceError


class BatchConversionService:
    """Service for batch processing of multiple OCT files.

    This service wraps the single-file ConversionService to provide
    batch processing capabilities with configurable error handling.

    Example usage:
        >>> from oct_converter_service import (
        ...     ConversionService,
        ...     ConversionRequest,
        ...     BatchConversionService,
        ... )
        >>> service = ConversionService()
        >>> batch = BatchConversionService(service)
        >>> requests = [
        ...     ConversionRequest(input_path="scan1.fds", output_dir="./out"),
        ...     ConversionRequest(input_path="scan2.fds", output_dir="./out"),
        ... ]
        >>> results = batch.convert_batch(requests, continue_on_error=True)
        >>> for result in results:
        ...     print(f"{result.input_path}: {'success' if result.success else 'failed'}")
    """

    def __init__(self, service: ConversionService | None = None):
        """Initialize the batch conversion service.

        Args:
            service: Optional ConversionService instance.
                     Creates a new one with defaults if not provided.
        """
        self._service = service or ConversionService()

    @property
    def service(self) -> ConversionService:
        """Get the underlying single-file conversion service."""
        return self._service

    def convert_batch(
        self,
        requests: list[ConversionRequest] | Iterator[ConversionRequest],
        continue_on_error: bool = True,
    ) -> list[ConversionResult]:
        """Convert multiple OCT files in batch.

        Processes each request sequentially, with configurable error handling.

        Args:
            requests: List or iterator of conversion requests.
            continue_on_error: Whether to continue processing remaining
                               files if one fails. If False, stops on
                               first error and re-raises the exception.

        Returns:
            List of ConversionResult objects, one per input request,
            in the same order as the input requests.

        Raises:
            ConversionServiceError: If continue_on_error is False and
                                    a conversion fails.
        """
        results = []

        # Convert iterator to list if needed for consistent behavior
        if not isinstance(requests, list):
            requests = list(requests)

        for request in requests:
            try:
                result = self._service.convert(request)
                results.append(result)
            except ConversionServiceError as e:
                if continue_on_error:
                    # Create a failed result to maintain ordering
                    from datetime import datetime

                    failed_result = ConversionResult(
                        success=False,
                        input_path=Path(request.input_path),
                        output_dir=Path(request.output_dir),
                        requested_outputs=request.outputs.copy(),
                        failures=[str(e)],
                        started_at=datetime.now(),
                        completed_at=datetime.now(),
                    )
                    results.append(failed_result)
                else:
                    # Re-raise to stop processing
                    raise

        return results

    def convert_batch_generator(
        self,
        requests: Iterator[ConversionRequest],
        continue_on_error: bool = True,
    ) -> Iterator[ConversionResult]:
        """Convert multiple OCT files using a generator.

        This is a memory-efficient alternative to convert_batch that
        yields results as they are produced, rather than collecting
        them all in memory.

        Args:
            requests: Iterator of conversion requests.
            continue_on_error: Whether to continue processing remaining
                               files if one fails.

        Yields:
            ConversionResult objects, one per input request.
        """
        for request in requests:
            try:
                result = self._service.convert(request)
                yield result
            except ConversionServiceError as e:
                if continue_on_error:
                    from datetime import datetime
                    from pathlib import Path

                    yield ConversionResult(
                        success=False,
                        input_path=Path(request.input_path),
                        output_dir=Path(request.output_dir),
                        requested_outputs=request.outputs.copy(),
                        failures=[str(e)],
                        started_at=datetime.now(),
                        completed_at=datetime.now(),
                    )
                else:
                    raise


# Import Path here to avoid circular imports
from pathlib import Path
