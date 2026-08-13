"""Processing pipeline for OCT file conversion.

Orchestrates the complete workflow from input file to multiple outputs.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from oct_converter_app.detector import FormatDetector, detect_format
from oct_converter_app.exporters.base import ExportError
from oct_converter_app.factory import ReaderFactory
from oct_converter_app.models import OCTStudy, Provenance
from oct_converter_app.registry import ExportRegistry, ExporterNotFoundError
from oct_converter_app.validation import OctValidator, ValidationResult


class ExtractionError(RuntimeError):
    """Raised when data extraction fails."""

    pass


class ProcessingPipeline:
    """Main processing pipeline for OCT file conversion.

    Orchestrates the complete workflow:
    1. Validate input file
    2. Detect format
    3. Create reader
    4. Extract OCT volume and fundus image
    5. Collect metadata
    6. Validate extracted data
    7. Route to exporters
    8. Return results

    The pipeline extracts data ONCE and shares it across in-memory exporters
    (NPY, images, metadata). Note: The DICOM exporter is currently an exception
    to single-pass processing because it delegates to create_dicom_from_oct(),
    which re-parses the source file to execute the validated DICOM metadata extraction pipeline.

    Attributes:
        compute_hash: Whether to compute SHA-256 hash of source file.
        reader_version: Version string to record in provenance.
    """

    def __init__(
        self,
        compute_hash: bool = False,
        reader_version: str | None = None,
    ):
        """Initialize processing pipeline.

        Args:
            compute_hash: Whether to compute file hash (adds I/O overhead).
            reader_version: Version string for provenance tracking.
        """
        self.compute_hash = compute_hash
        self.reader_version = reader_version or "oct_converter_app"

    def process(
        self,
        input_path: Path | str,
        output_dir: Path | str,
        outputs: list[str] | None = None,
        exporter_options: dict[str, dict] | None = None,
        validate: bool = True,
        continue_on_warning: bool = True,
    ) -> OCTStudy:
        """Process an OCT file and export to specified formats.

        This is the main entry point for batch processing.

        Args:
            input_path: Path to input OCT file.
            output_dir: Directory for output files.
            outputs: List of output formats to generate.
                     Options: 'dicom', 'npy', 'images', 'metadata'
                     Default: ['metadata'] (just extract and validate)
            exporter_options: Per-exporter configuration options.
                              Dict mapping exporter name to options dict.
            validate: Whether to validate extracted data before export (default: True).
                      Setting validate=False explicitly bypasses safety validation checks
                      (e.g., empty volumes or missing data) and attempts export regardless of data validity.
            continue_on_warning: Whether to continue if validation has warnings.

        Returns:
            OCTStudy containing extracted data and processing results.

        Raises:
            FileNotFoundError: If input file does not exist.
            UnsupportedFormatError: If file format is not supported.
            ReaderCreationError: If reader cannot be created.
            ExtractionError: If data extraction fails.
            ValidationError: If validation fails (when validate=True).
            ExportError: If export fails.
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        outputs = outputs or ["metadata"]
        exporter_options = exporter_options or {}

        # Step 1: Validate input
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if not input_path.is_file():
            raise ValueError(f"Input path is not a file: {input_path}")

        # Step 2: Detect format
        try:
            format_name = detect_format(input_path)
        except Exception as e:
            raise type(e)(f"Format detection failed: {e}") from e

        # Step 3: Create reader
        reader = ReaderFactory.create(format_name, input_path)

        # Step 4: Extract data
        oct_volume = None
        fundus = None
        raw_metadata = {}
        warnings = []

        try:
            # Try to extract OCT volume
            try:
                oct_volume = reader.read_oct_volume()
            except Exception as e:
                warnings.append(f"OCT volume extraction failed: {e}")

            # Try to extract fundus image
            try:
                fundus = reader.read_fundus_image()
            except Exception as e:
                warnings.append(f"Fundus image extraction failed: {e}")

            # Extract raw metadata
            try:
                raw_metadata = reader.read_all_metadata() or {}
            except Exception as e:
                warnings.append(f"Metadata extraction failed: {e}")

        except Exception as e:
            raise ExtractionError(f"Data extraction failed: {e}") from e

        # Step 5: Build study object
        provenance = Provenance.create(
            source_path=input_path,
            source_format=format_name,
            compute_hash=self.compute_hash,
            reader_version=self.reader_version,
        )

        study = OCTStudy(
            source_path=input_path,
            source_format=format_name,
            oct_volume=oct_volume,
            fundus=fundus,
            metadata={},  # Common metadata extracted below
            raw_metadata=raw_metadata,
            warnings=warnings,
            provenance=provenance,
        )

        # Extract common metadata fields
        study.metadata = self._extract_common_metadata(study)

        # Step 6: Validate
        if validate:
            validation_result = OctValidator.validate_study(oct_volume, fundus)
            study.warnings.extend(validation_result.warnings)

            if not validation_result.is_valid and validation_result.errors:
                errors_str = "; ".join(validation_result.errors)
                raise ValidationError(f"Validation failed: {errors_str}")

            if validation_result.warnings and not continue_on_warning:
                warnings_str = "; ".join(validation_result.warnings)
                raise ValidationError(f"Validation warnings: {warnings_str}")

        # Step 7: Export
        created_files = []
        for output_name in outputs:
            try:
                exporter = ExportRegistry.get(output_name)

                # Check if exporter supports available data
                if not exporter.supports_oct(study) and not exporter.supports_fundus(study):
                    study.add_warning(
                        f"Exporter '{output_name}' skipped: no compatible data available"
                    )
                    continue

                options = exporter_options.get(output_name, {})
                files = exporter.export(study, output_dir, options)
                created_files.extend(files)

            except ExporterNotFoundError:
                raise
            except ExportError as e:
                raise
            except Exception as e:
                raise ExportError(f"Export '{output_name}' failed: {e}") from e

        return study

    def _extract_common_metadata(self, study: OCTStudy) -> dict[str, Any]:
        """Extract common metadata fields from study.

        Args:
            study: The study to extract metadata from.

        Returns:
            Dictionary with common metadata fields.
        """
        metadata = {}

        # From OCT volume
        if study.oct_volume:
            if study.oct_volume.patient_id:
                metadata["patient_id"] = study.oct_volume.patient_id
            if study.oct_volume.laterality:
                metadata["laterality"] = study.oct_volume.laterality
            if study.oct_volume.acquisition_date:
                metadata["acquisition_date"] = study.oct_volume.acquisition_date.isoformat()
            if study.oct_volume.first_name or study.oct_volume.surname:
                name_parts = []
                if study.oct_volume.first_name:
                    name_parts.append(study.oct_volume.first_name)
                if study.oct_volume.surname:
                    name_parts.append(study.oct_volume.surname)
                metadata["patient_name"] = " ".join(name_parts)
            if study.oct_volume.sex:
                metadata["sex"] = study.oct_volume.sex
            if study.oct_volume.DOB:
                metadata["date_of_birth"] = study.oct_volume.DOB

        # From fundus (fill gaps)
        if study.fundus:
            if not metadata.get("patient_id") and study.fundus.patient_id:
                metadata["patient_id"] = study.fundus.patient_id
            if not metadata.get("laterality") and study.fundus.laterality:
                metadata["laterality"] = study.fundus.laterality

        return metadata

    def load(self, input_path: Path | str) -> OCTStudy:
        """Load and extract data from an OCT file without exporting.

        This is useful for programmatic access to extracted data.

        Args:
            input_path: Path to input OCT file.

        Returns:
            OCTStudy containing extracted data.
        """
        return self.process(
            input_path=input_path,
            output_dir=Path.cwd() / "_temp",  # Dummy, won't be used
            outputs=[],  # No exports
            validate=True,
        )


class ValidationError(ValueError):
    """Raised when validation fails."""

    pass


# Convenience function
def extract_oct(input_path: Path | str) -> OCTStudy:
    """Extract OCT data from a file.

    Convenience wrapper around ProcessingPipeline.load().

    Args:
        input_path: Path to OCT file.

    Returns:
        OCTStudy containing extracted data.

    Examples:
        >>> study = extract_oct("scan.fds")
        >>> print(study.volume_dimensions)
        >>> print(study.patient_id)
    """
    return ProcessingPipeline().load(input_path)
