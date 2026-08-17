"""DICOM exporter for OCT studies into Ophthalmic Tomography Image Storage objects.

Operates directly on the in-memory OCTStudy object without re-parsing source files.
"""

from __future__ import annotations

from pathlib import Path

from oct_converter.dicom import write_ophthalmic_tomography_dicom_from_study
from oct_converter_app.exporters.base import BaseExporter, ExportError, sanitize_path_component
from oct_converter_app.models import OCTStudy


class DicomExporter(BaseExporter):
    """Exporter for DICOM format (Ophthalmic Tomography Image Storage IOD).

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

        Operates directly from the in-memory OCTStudy object.

        Args:
            study: The OCTStudy containing extracted volume data.
            output_dir: Directory to write DICOM files.
            options: Optional configuration.

        Returns:
            List of paths to created DICOM files.

        Raises:
            ExportError: If DICOM conversion fails.
        """
        output_path = self._ensure_output_dir(output_dir)
        options_copy = dict(options) if options else {}
        overwrite = options_copy.pop("overwrite", True)

        if not self.supports_oct(study):
            raise ExportError("No OCT volume data available to export to DICOM")

        filename_stem = options_copy.get("oct_filename")
        if filename_stem:
            filename_stem = sanitize_path_component(filename_stem, default="oct")
            if filename_stem.endswith(".dcm"):
                filename_stem = filename_stem[:-4]
        else:
            patient_id = sanitize_path_component(study.patient_id, default="unknown")
            filename_stem = f"{patient_id}_oct"

        output_file = output_path / f"{filename_stem}.dcm"

        if not overwrite and output_file.exists():
            raise ExportError(
                f"File already exists and overwrite is disabled: {output_file}"
            )

        try:
            created_file = write_ophthalmic_tomography_dicom_from_study(study, output_file)
            return [created_file]
        except Exception as e:
            raise ExportError(f"DICOM export failed: {e}") from e

    def supports_oct(self, study: OCTStudy) -> bool:
        """Check if DICOM export is possible for this study.

        Args:
            study: The study to check.

        Returns:
            True if valid OCT volume data is present.
        """
        return (
            study.oct_volume is not None
            and study.oct_volume.volume is not None
            and len(study.oct_volume.volume) > 0
        )

