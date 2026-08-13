"""Main conversion service for OCT file processing."""

from datetime import datetime
from pathlib import Path

from oct_converter_app.pipeline import ProcessingPipeline, ValidationError as AppValidationError
from oct_converter_app.registry import ExporterNotFoundError
from oct_converter_app.exporters.base import ExportError

from oct_converter_service.config import ConversionConfig
from oct_converter_service.requests import ConversionRequest
from oct_converter_service.results import ConversionResult
from oct_converter_service.errors import (
    ConversionServiceError,
    InvalidRequestError,
    InputNotFoundError,
    UnsupportedFormatError,
    ConversionFailedError,
    OutputError,
    OverwriteNotAllowedError,
)


class ConversionService:
    """High-level service for OCT file conversion.

    This service provides a clean API for converting OCT files,
    delegating to the application layer pipeline for actual processing.

    Example usage:
        >>> from oct_converter_service import ConversionService, ConversionRequest
        >>> service = ConversionService()
        >>> request = ConversionRequest(
        ...     input_path="scan.fds",
        ...     output_dir="./output",
        ...     outputs=["dicom", "npy", "images", "metadata"]
        ... )
        >>> result = service.convert(request)
        >>> if result.success:
        ...     print(f"Generated {len(result.generated_files)} files")
    """

    def __init__(self, config: ConversionConfig | None = None):
        """Initialize the conversion service.

        Args:
            config: Optional configuration for the service.
                    Uses defaults if not provided.
        """
        self.config = config or ConversionConfig()
        self._pipeline = ProcessingPipeline(
            compute_hash=self.config.compute_hash,
        )

    def convert(self, request: ConversionRequest) -> ConversionResult:
        """Convert an OCT file according to the request.

        This method validates the request, processes the file through
        the application layer pipeline, and returns structured results.

        Args:
            request: The conversion request with all parameters.

        Returns:
            ConversionResult with details of the operation.

        Raises:
            InvalidRequestError: If the request is invalid.
            InputNotFoundError: If the input file does not exist.
            UnsupportedFormatError: If format is not supported.
            ConversionFailedError: If conversion fails.
            OutputError: If output writing fails.
            OverwriteNotAllowedError: If overwrite is disabled and files exist.
        """
        started_at = datetime.now()

        # Validate request (raises ValueError which we wrap)
        try:
            # Force validation by accessing the validated fields
            _ = request.input_path
            _ = request.output_dir
            _ = request.outputs
        except ValueError as e:
            raise InvalidRequestError(str(e)) from e

        # Check input exists
        input_path = Path(request.input_path)
        if not input_path.exists():
            raise InputNotFoundError(f"Input file not found: {input_path}")
        if not input_path.is_file():
            raise InvalidRequestError(f"Input path is not a file: {input_path}")

        # Ensure output directory can be created
        output_dir = Path(request.output_dir)
        if output_dir.exists() and not output_dir.is_dir():
            raise OutputError(f"Output path exists but is not a directory: {output_dir}")

        # Process through the application layer pipeline
        try:
            study = self._pipeline.process(
                input_path=input_path,
                output_dir=output_dir,
                outputs=request.outputs,
                exporter_options=request.exporter_options,
                validate=request.validate,
                continue_on_warning=request.continue_on_warning,
            )

            # Get generated files - check what was actually exported
            generated_files = []
            skipped_outputs = []
            
            # Track which outputs were requested vs what was generated
            # The pipeline stores created files in the study's internal state
            if hasattr(study, '_exported_files'):
                generated_files = list(study._exported_files)
            
            # Determine skipped outputs by checking what exists
            for output_name in request.outputs:
                # Check if any file matching this output pattern was created
                found = False
                for f in generated_files:
                    if output_name in f.name.lower() or f.suffix[1:] == output_name:
                        found = True
                        break
                if not found:
                    skipped_outputs.append(output_name)

            completed_at = datetime.now()

            return ConversionResult.from_study(
                study=study,
                success=len(generated_files) > 0 or len(skipped_outputs) == len(request.outputs),
                generated_files=generated_files,
                skipped_outputs=skipped_outputs,
                failures=[],
                started_at=started_at,
                completed_at=completed_at,
            )

        except FileNotFoundError as e:
            raise InputNotFoundError(str(e)) from e
        except ExporterNotFoundError as e:
            raise UnsupportedFormatError(str(e)) from e
        except AppValidationError as e:
            raise ConversionFailedError(f"Validation failed: {e}") from e
        except ExportError as e:
            raise OutputError(str(e)) from e
        except Exception as e:
            # Wrap any other exception as a conversion failure
            raise ConversionFailedError(f"Conversion failed: {e}") from e
