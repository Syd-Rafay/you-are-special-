"""Metadata JSON exporter for OCT studies.

Exports study metadata as a structured JSON file.
"""

from __future__ import annotations

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any

from oct_converter_app.exporters.base import BaseExporter, ExportError
from oct_converter_app.models import OCTStudy


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        if isinstance(obj, (Path,)):
            return str(obj)
        return super().default(obj)


class MetadataExporter(BaseExporter):
    """Exporter for metadata in JSON format.

    Exports common metadata fields and preserves raw vendor-specific
    metadata for future use. Does NOT include pixel arrays.

    Attributes:
        name: Exporter name ('metadata').
        indent: JSON indentation level (default: 2).
        include_raw: Whether to include raw vendor metadata (default: True).
    """

    name = "metadata"

    def __init__(self, indent: int = 2, include_raw: bool = True):
        """Initialize metadata exporter.

        Args:
            indent: JSON indentation level. Use None for compact output.
            include_raw: Whether to include raw vendor metadata.
        """
        self.indent = indent
        self.include_raw = include_raw

    def export(
        self, study: OCTStudy, output_dir: Path | str, options: dict | None = None
    ) -> list[Path]:
        """Export study metadata as JSON.

        Creates a single JSON file containing all available metadata.

        Args:
            study: The OCTStudy containing extracted data.
            output_dir: Directory to write JSON file.
            options: Optional configuration.
                     Keys: 'filename' to override default,
                           'indent' to override instance indent,
                           'include_raw' to override instance include_raw.

        Returns:
            List with single path to created JSON file.

        Raises:
            ExportError: If export fails.
        """
        output_path = self._ensure_output_dir(output_dir)

        # Get option overrides
        indent = options.get("indent", self.indent) if options else self.indent
        include_raw = options.get("include_raw", self.include_raw) if options else self.include_raw

        filename = options.get("filename") if options else None
        if not filename:
            patient_id = study.patient_id or "unknown"
            filename = f"{patient_id}_metadata.json"

        filepath = output_path / filename

        try:
            metadata = self._build_metadata(study, include_raw)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=indent, cls=NumpyEncoder, ensure_ascii=False)

            return [filepath]

        except Exception as e:
            raise ExportError(f"Failed to export metadata to JSON: {e}") from e

    def _build_metadata(self, study: OCTStudy, include_raw: bool = True) -> dict[str, Any]:
        """Build metadata dictionary from study.

        Args:
            study: The OCTStudy to extract metadata from.
            include_raw: Whether to include raw vendor metadata.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        metadata: dict[str, Any] = {
            "source": {
                "path": str(study.source_path),
                "format": study.source_format,
            },
            "provenance": {},
            "patient": {},
            "oct": {},
            "fundus": {},
        }

        # Add provenance if available
        if study.provenance:
            metadata["provenance"] = {
                "processing_timestamp": study.provenance.processing_timestamp.isoformat(),
                "file_hash": study.provenance.file_hash,
                "reader_version": study.provenance.reader_version,
            }

        # Patient info (from OCT or fundus)
        if study.patient_id:
            metadata["patient"]["patient_id"] = study.patient_id
        if study.laterality:
            metadata["patient"]["laterality"] = study.laterality
        if study.acquisition_date:
            metadata["patient"]["acquisition_date"] = study.acquisition_date.isoformat()

        # OCT volume metadata
        if study.oct_volume:
            oct_vol = study.oct_volume
            metadata["oct"]["present"] = True
            metadata["oct"]["num_slices"] = oct_vol.num_slices

            if oct_vol.volume:
                h, w = oct_vol.volume[0].shape
                metadata["oct"]["dimensions"] = {
                    "num_slices": oct_vol.num_slices,
                    "height": h,
                    "width": w,
                }

            if oct_vol.pixel_spacing:
                metadata["oct"]["pixel_spacing_mm"] = oct_vol.pixel_spacing

            if oct_vol.volume_id:
                metadata["oct"]["volume_id"] = oct_vol.volume_id

            if oct_vol.first_name or oct_vol.surname:
                name_parts = []
                if oct_vol.first_name:
                    name_parts.append(oct_vol.first_name)
                if oct_vol.surname:
                    name_parts.append(oct_vol.surname)
                metadata["patient"]["name"] = " ".join(name_parts)

            if oct_vol.sex:
                metadata["patient"]["sex"] = oct_vol.sex

            if oct_vol.DOB:
                metadata["patient"]["date_of_birth"] = oct_vol.DOB

            if oct_vol.contours:
                metadata["oct"]["has_contours"] = True
                metadata["oct"]["contour_count"] = len(oct_vol.contours)

            # Common metadata fields
            if study.metadata:
                metadata["common_metadata"] = study.metadata

        else:
            metadata["oct"]["present"] = False

        # Fundus metadata
        if study.fundus:
            fundus = study.fundus
            metadata["fundus"]["present"] = True

            if fundus.image.size > 0:
                shape = fundus.image.shape
                metadata["fundus"]["dimensions"] = {
                    "height": int(shape[0]),
                    "width": int(shape[1]),
                }
                if len(shape) > 2:
                    metadata["fundus"]["channels"] = int(shape[2])

            if fundus.pixel_spacing:
                metadata["fundus"]["pixel_spacing_mm"] = fundus.pixel_spacing

            if fundus.image_id:
                metadata["fundus"]["image_id"] = fundus.image_id

            if fundus.DOB:
                metadata["patient"]["date_of_birth"] = fundus.DOB

            if fundus.acquisition_date:
                metadata["fundus"]["acquisition_date"] = fundus.acquisition_date

        else:
            metadata["fundus"]["present"] = False

        # Raw vendor metadata (preserved as-is)
        if include_raw and study.raw_metadata:
            metadata["raw_metadata"] = study.raw_metadata

        # Warnings from processing
        if study.warnings:
            metadata["warnings"] = study.warnings

        # Capabilities summary
        if study.capabilities:
            metadata["capabilities"] = {
                "has_oct_volume": study.capabilities.has_oct_volume,
                "has_fundus": study.capabilities.has_fundus,
                "has_pixel_spacing": study.capabilities.has_pixel_spacing,
                "has_contours": study.capabilities.has_contours,
            }

        return metadata

    def supports_oct(self, study: OCTStudy) -> bool:
        """Metadata export is always possible if we have any data."""
        return study.oct_volume is not None or study.fundus is not None

    def supports_fundus(self, study: OCTStudy) -> bool:
        """Metadata export is always possible if we have any data."""
        return study.oct_volume is not None or study.fundus is not None
