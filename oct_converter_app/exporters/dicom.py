"""DICOM exporter wrapping existing oct_converter DICOM functionality.

This exporter delegates to the existing create_dicom_from_oct function
to preserve validated DICOM generation logic.

Performance Note:
Unlike NpyExporter, ImageExporter, and MetadataExporter which operate
from the in-memory OCTStudy object, DicomExporter is an exception to
single-pass extraction. It re-parses the source file via create_dicom_from_oct()
to apply validated format-specific DICOM metadata mappings.
"""

from __future__ import annotations

from pathlib import Path

from oct_converter.dicom import create_dicom_from_oct

from oct_converter_app.exporters.base import BaseExporter, ExportError
from oct_converter_app.models import OCTStudy


class DicomExporter(BaseExporter):
    """Exporter for DICOM format.

    This exporter wraps the existing create_dicom_from_oct function from
    oct_converter.dicom to preserve all validated DICOM generation logic.

    Note: The current implementation requires the original source file path
    because the underlying create_dicom_from_oct re-reads the file to extract
    data and apply format-specific DICOM metadata mappings. This preserves
    the validated DICOM output but means DicomExporter incurs a second file
    parse rather than operating exclusively from the in-memory OCTStudy object.

    Attributes:
        name: Exporter name ('dicom').
    """

    name = "dicom"

    def __init__(
        self,
        rows: int = 1024,
        cols: int = 512,
        interlaced: bool = False,
        diskbuffered: bool = False,
        extract_scan_repeats: bool = False,
        scalex: float = 0.01,
        slice_thickness: float = 0.05,
    ):
        """Initialize DICOM exporter with conversion parameters.

        Args:
            rows: Number of rows (for .img files).
            cols: Number of columns (for .img files).
            interlaced: Whether data is interlaced (for .img files).
            diskbuffered: Use disk buffering (for Bioptigen .OCT).
            extract_scan_repeats: Extract repeated scans (for .e2e).
            scalex: X scale in mm (for .e2e).
            slice_thickness: Z scale in mm (for .e2e).
        """
        self.rows = rows
        self.cols = cols
        self.interlaced = interlaced
        self.diskbuffered = diskbuffered
        self.extract_scan_repeats = extract_scan_repeats
        self.scalex = scalex
        self.slice_thickness = slice_thickness

    def export(
        self, study: OCTStudy, output_dir: Path | str, options: dict | None = None
    ) -> list[Path]:
        """Export study to DICOM format.

        This method calls the existing create_dicom_from_oct function which
        re-reads the source file. The study object is used for validation
        but the actual DICOM generation uses the original file.

        Args:
            study: The OCTStudy (used for validation, not data extraction).
            output_dir: Directory to write DICOM files.
            options: Optional overrides for conversion parameters.
                     Keys: 'rows', 'cols', 'interlaced', 'diskbuffered',
                           'extract_scan_repeats', 'scalex', 'slice_thickness'

        Returns:
            List of paths to created DICOM files.

        Raises:
            ExportError: If DICOM conversion fails.
        """
        output_path = self._ensure_output_dir(output_dir)

        # Apply any option overrides
        kwargs = {
            "rows": self.rows,
            "cols": self.cols,
            "interlaced": self.interlaced,
            "diskbuffered": self.diskbuffered,
            "extract_scan_repeats": self.extract_scan_repeats,
            "scalex": self.scalex,
            "slice_thickness": self.slice_thickness,
        }
        if options:
            kwargs.update(options)

        try:
            # Delegate to existing validated DICOM conversion
            # This re-reads the source file but preserves correct behavior
            dicom_paths = create_dicom_from_oct(
                input_file=str(study.source_path),
                output_dir=str(output_path),
                **kwargs,
            )
            return [Path(p) for p in dicom_paths]
        except Exception as e:
            raise ExportError(
                f"DICOM export failed for {study.source_path}: {e}"
            ) from e

    def supports_oct(self, study: OCTStudy) -> bool:
        """Check if DICOM export is possible for this study.

        DICOM export requires the source file to exist (since it re-reads).

        Args:
            study: The study to check.

        Returns:
            True if source file exists.
        """
        return study.source_path.exists()
