"""Base exporter interface.

This module defines the abstract base class for all output exporters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from oct_converter_app.models import OCTStudy


class ExportError(RuntimeError):
    """Raised when an export operation fails."""

    pass


class BaseExporter(ABC):
    """Abstract base class for all OCT data exporters.

    Exporters are responsible for converting extracted OCT study data
    into specific output formats (DICOM, NPY, images, metadata JSON, etc.).

    Subclasses must implement the export() method.

    Attributes:
        name: Human-readable name for this exporter format.
    """

    name: str = "base"

    @abstractmethod
    def export(self, study: OCTStudy, output_dir: Path | str, options: dict | None = None) -> list[Path]:
        """Export study data to the specified format.

        Args:
            study: The OCTStudy containing extracted data.
            output_dir: Directory to write output files.
            options: Optional exporter-specific configuration.

        Returns:
            List of paths to created files.

        Raises:
            ExportError: If the export operation fails.
        """
        pass

    def supports_oct(self, study: OCTStudy) -> bool:
        """Check if this exporter can handle the given study's OCT data.

        Default implementation returns True if OCT volume exists.
        Subclasses may override for more specific checks.

        Args:
            study: The study to check.

        Returns:
            True if OCT export is supported.
        """
        return study.oct_volume is not None

    def supports_fundus(self, study: OCTStudy) -> bool:
        """Check if this exporter can handle the given study's fundus data.

        Default implementation returns True if fundus image exists.
        Subclasses may override for more specific checks.

        Args:
            study: The study to check.

        Returns:
            True if fundus export is supported.
        """
        return study.fundus is not None

    def _ensure_output_dir(self, output_dir: Path | str) -> Path:
        """Ensure output directory exists.

        Args:
            output_dir: Path to output directory.

        Returns:
            Path object for the output directory.

        Raises:
            ExportError: If directory cannot be created.
        """
        output_path = Path(output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ExportError(f"Cannot create output directory {output_path}: {e}") from e
        return output_path
