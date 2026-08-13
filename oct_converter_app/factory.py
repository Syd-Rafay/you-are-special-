"""Reader factory for OCT file formats.

This module provides a registry-based factory pattern for instantiating
the appropriate reader class based on the detected format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Type

from oct_converter.readers import BOCT, Dicom, E2E, FDA, FDS, IMG, POCT


class ReaderCreationError(ValueError):
    """Raised when a reader cannot be created."""

    pass


class ReaderFactory:
    """Factory for creating OCT readers based on format name.

    The factory maintains a registry of format names to reader classes.
    New formats can be registered dynamically.

    Attributes:
        _registry: Class-level dictionary mapping format names to reader classes.

    Examples:
        >>> reader = ReaderFactory.create("fds", "scan.fds")
        >>> reader = ReaderFactory.create("fda", Path("scan.fda"))
    """

    # Default registry with all supported readers
    _registry: dict[str, Type] = {
        "fds": FDS,
        "fda": FDA,
        "e2e": E2E,
        "img": IMG,
        "boct": BOCT,
        "poct": POCT,
        "dcm": Dicom,
    }

    @classmethod
    def create(cls, format_name: str, filepath: Path | str) -> object:
        """Create a reader instance for the specified format.

        Args:
            format_name: Canonical format identifier (e.g., 'fds', 'fda').
            filepath: Path to the OCT file.

        Returns:
            Reader instance for the specified format.

        Raises:
            ReaderCreationError: If the format is not registered.
        """
        filepath = Path(filepath)
        reader_class = cls._registry.get(format_name.lower())

        if reader_class is None:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ReaderCreationError(
                f"Unknown format: '{format_name}'. "
                f"Available formats: {available}"
            )

        try:
            return reader_class(str(filepath))
        except Exception as e:
            raise ReaderCreationError(
                f"Failed to create {format_name} reader for {filepath}: {e}"
            ) from e

    @classmethod
    def register(cls, format_name: str, reader_class: Type) -> None:
        """Register a new reader class for a format.

        This allows extending support for new formats without modifying
        the factory code.

        Args:
            format_name: Canonical format identifier.
            reader_class: Reader class that accepts a filepath string.

        Examples:
            >>> from my_custom_reader import CustomReader
            >>> ReaderFactory.register("custom", CustomReader)
        """
        cls._registry[format_name.lower()] = reader_class

    @classmethod
    def unregister(cls, format_name: str) -> bool:
        """Unregister a format from the factory.

        Args:
            format_name: Format name to remove.

        Returns:
            True if the format was registered and removed, False otherwise.
        """
        return cls._registry.pop(format_name.lower(), None) is not None

    @classmethod
    def supported_formats(cls) -> list[str]:
        """Return list of supported format identifiers.

        Returns:
            Sorted list of format names.
        """
        return sorted(cls._registry.keys())

    @classmethod
    def get_reader_class(cls, format_name: str) -> Type:
        """Get the reader class for a format without instantiating.

        Args:
            format_name: Canonical format identifier.

        Returns:
            Reader class for the format.

        Raises:
            ReaderCreationError: If the format is not registered.
        """
        reader_class = cls._registry.get(format_name.lower())
        if reader_class is None:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ReaderCreationError(
                f"Unknown format: '{format_name}'. "
                f"Available formats: {available}"
            )
        return reader_class
