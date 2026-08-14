"""Registry for OCT data exporters.

Provides a central registry for looking up exporters by name.
"""

from __future__ import annotations

from typing import Type

from oct_converter_app.exporters.base import BaseExporter
from oct_converter_app.exporters.dicom import DicomExporter
from oct_converter_app.exporters.images import ImageExporter
from oct_converter_app.exporters.metadata import MetadataExporter
from oct_converter_app.exporters.npy import NpyExporter
from oct_converter_app.exporters.zarr import ZarrExporter


class ExporterNotFoundError(KeyError):
    """Raised when a requested exporter is not found."""

    pass


class ExportRegistry:
    """Registry for OCT data exporters.

    Provides lookup of exporters by name and allows registration of
    custom exporters.

    Attributes:
        _exporters: Class-level dictionary mapping names to exporter classes.
    """

    # Default registry with built-in exporters
    _exporters: dict[str, type[BaseExporter]] = {
        "dicom": DicomExporter,
        "npy": NpyExporter,
        "images": ImageExporter,
        "metadata": MetadataExporter,
        "zarr": ZarrExporter,
    }

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseExporter:
        """Get an exporter instance by name.

        Args:
            name: Exporter name (case-insensitive).
            **kwargs: Arguments to pass to exporter constructor.

        Returns:
            Exporter instance.

        Raises:
            ExporterNotFoundError: If the exporter is not registered.
        """
        exporter_class = cls._exporters.get(name.lower())
        if exporter_class is None:
            available = ", ".join(sorted(cls._exporters.keys()))
            raise ExporterNotFoundError(
                f"Unknown exporter: '{name}'. Available exporters: {available}"
            )
        return exporter_class(**kwargs)

    @classmethod
    def register(cls, name: str, exporter_class: Type[BaseExporter]) -> None:
        """Register a new exporter class.

        Args:
            name: Name to register under (will be lowercased).
            exporter_class: Exporter class (must inherit from BaseExporter).

        Examples:
            >>> class MyExporter(BaseExporter):
            ...     name = "custom"
            ...     def export(self, study, output_dir, options=None):
            ...         ...
            >>> ExportRegistry.register("custom", MyExporter)
        """
        if not issubclass(exporter_class, BaseExporter):
            raise TypeError(f"{exporter_class} must inherit from BaseExporter")
        cls._exporters[name.lower()] = exporter_class

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister an exporter.

        Args:
            name: Exporter name to remove.

        Returns:
            True if the exporter was registered and removed.
        """
        return cls._exporters.pop(name.lower(), None) is not None

    @classmethod
    def list_exporters(cls) -> list[str]:
        """List all registered exporter names.

        Returns:
            Sorted list of exporter names.
        """
        return sorted(cls._exporters.keys())

    @classmethod
    def supports(cls, name: str) -> bool:
        """Check if an exporter is registered.

        Args:
            name: Exporter name to check.

        Returns:
            True if the exporter is registered.
        """
        return name.lower() in cls._exporters
