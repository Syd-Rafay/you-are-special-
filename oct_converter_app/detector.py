"""Format detection for OCT files.

This module provides format detection based on file extension.
For .OCT files, special handling is required to disambiguate between
Bioptigen (.OCT) and Optovue (.OCT) formats.
"""

from __future__ import annotations

from pathlib import Path

from construct import StreamError, StringError

from oct_converter.exceptions import InvalidOCTReaderError
from oct_converter.readers import BOCT


# Supported extensions (lowercase)
SUPPORTED_EXTENSIONS = {".fds", ".fda", ".e2e", ".img", ".oct", ".dcm"}

# Format name mapping
EXTENSION_TO_FORMAT = {
    ".fds": "fds",
    ".fda": "fda",
    ".e2e": "e2e",
    ".img": "img",
    ".dcm": "dcm",
}


class UnsupportedFormatError(ValueError):
    """Raised when an unsupported file format is detected."""

    pass


def detect_format(filepath: Path | str) -> str:
    """Detect the OCT format from a file path.

    Uses file extension for detection. For .OCT files, attempts to
    disambiguate between Bioptigen and Optovue formats by trying
    to validate with the Bioptigen reader.

    Args:
        filepath: Path to the OCT file.

    Returns:
        Canonical format identifier (e.g., 'fds', 'fda', 'boct', 'poct').

    Raises:
        UnsupportedFormatError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.

    Examples:
        >>> detect_format("scan.fds")
        'fds'
        >>> detect_format("scan.OCT")  # Bioptigen
        'boct'
        >>> detect_format("scan.oct")  # Optovue
        'poct'
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if not filepath.is_file():
        raise ValueError(f"Path is not a file: {filepath}")

    ext = filepath.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported OCT format: {ext}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Special handling for .OCT files - disambiguate Bioptigen vs Optovue
    if ext == ".oct":
        try:
            # Try to validate as Bioptigen
            BOCT(str(filepath))._validate(filepath)
            return "boct"
        except (InvalidOCTReaderError, StreamError, StringError, UnicodeDecodeError):
            # If Bioptigen validation fails, treat as Optovue
            return "poct"

    # Standard extension-based detection
    return EXTENSION_TO_FORMAT[ext]


class FormatDetector:
    """Format detector class for consistency with factory pattern.

    This class wraps the detect_format function for use in the pipeline.
    """

    @staticmethod
    def detect(filepath: Path | str) -> str:
        """Detect the format of an OCT file.

        Args:
            filepath: Path to the OCT file.

        Returns:
            Canonical format identifier.

        Raises:
            UnsupportedFormatError: If the format is not supported.
        """
        return detect_format(filepath)

    @staticmethod
    def supported_formats() -> set[str]:
        """Return the set of supported format identifiers.

        Returns:
            Set of format names (without leading dot).
        """
        return {fmt for fmt in EXTENSION_TO_FORMAT.values()} | {"boct", "poct"}
