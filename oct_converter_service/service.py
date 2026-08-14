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

        # Validate request schema
        try:
            request.validate_request()
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

        # Resolve configuration precedence: explicit request > service config > default
        overwrite = request.overwrite if request.overwrite is not None else self.config.overwrite
        validate = request.validate if request.validate is not None else self.config.validate
        continue_on_warning = (
            request.continue_on_warning
            if request.continue_on_warning is not None
            else self.config.continue_on_warning
        )
        compute_hash = (
            request.compute_hash
            if request.compute_hash is not None
            else self.config.compute_hash
        )

        # Build per-exporter options with overwrite propagation
        merged_exporter_options: dict[str, dict] = {}
        for out_format in request.outputs:
            opts = {"overwrite": overwrite}
            if out_format in request.exporter_options:
                opts.update(request.exporter_options[out_format])
            merged_exporter_options[out_format] = opts

        # Process through the application layer pipeline
        try:
            pipeline_result = self._pipeline.process_with_outputs(
                input_path=input_path,
                output_dir=output_dir,
                outputs=request.outputs,
                exporter_options=merged_exporter_options,
                validate=validate,
                continue_on_warning=continue_on_warning,
                compute_hash=compute_hash,
            )

            study = pipeline_result.study
            generated_files = pipeline_result.created_files

            # Determine skipped outputs by inspecting generated files against requested formats
            skipped_outputs = []
            format_extensions = {
                "dicom": {".dcm"},
                "npy": {".npy"},
                "images": {".png", ".jpg", ".jpeg", ".tiff"},
                "metadata": {".json"},
            }

            for output_name in request.outputs:
                valid_exts = format_extensions.get(output_name, set())
                found = any(f.suffix.lower() in valid_exts for f in generated_files)
                if not found:
                    skipped_outputs.append(output_name)

            completed_at = datetime.now()

            # Success requires generating files and having no skipped requested outputs
            success = len(generated_files) > 0 and len(skipped_outputs) == 0

            failures = []
            if not success:
                if len(generated_files) == 0:
                    failures.append("No output files were generated for the request.")
                for skipped_out in skipped_outputs:
                    failures.append(
                        f"Requested output format '{skipped_out}' was not produced."
                    )

            return ConversionResult.from_study(
                study=study,
                success=success,
                generated_files=generated_files,
                requested_outputs=list(request.outputs),
                output_dir=output_dir,
                skipped_outputs=skipped_outputs,
                failures=failures,
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
            err_msg = str(e).lower()
            if "overwrite" in err_msg or "already exists" in err_msg:
                raise OverwriteNotAllowedError(str(e)) from e
            raise OutputError(str(e)) from e
        except Exception as e:
            # Wrap any other exception as a conversion failure
            raise ConversionFailedError(f"Conversion failed: {e}") from e
