"""Base exporter interface.

This module defines the abstract base class for all output exporters.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

from oct_converter_app.models import OCTStudy


class ExportError(RuntimeError):
    """Raised when an export operation fails."""

    pass


def sanitize_path_component(name: str | None, default: str = "unknown") -> str:
    """Sanitize a string for safe use as a single filesystem path component (filename or directory component).

    Prevents directory traversal attacks (e.g. '../', '\', absolute paths, control characters)
    while preserving valid patient IDs and filenames.

    Args:
        name: Raw string to sanitize (e.g. patient ID or custom filename).
        default: Fallback string if name is empty, None, or fully invalid (default: 'unknown').

    Returns:
        Sanitized string containing no path separators, control characters, or relative traversal segments.
    """
    if name is None:
        return default

    s = str(name)
    # Remove null bytes and control characters (ASCII 0-31, 127)
    s = re.sub(r"[\x00-\x1f\x7f]", "", s)
    # Replace path separators and reserved characters with underscores
    s = re.sub(r'[\\/:*?"<>|]', "_", s)
    # Strip leading/trailing whitespace and dots
    s = s.strip(" .")
    # Remove any remaining leading dots
    while s.startswith("."):
        s = s.lstrip(".")
    s = s.strip(" .")

    if not s:
        return default

    return s



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
