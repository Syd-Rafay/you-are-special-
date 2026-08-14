"""Zarr v3 exporter for OCT volumes and fundus images.

Exports extracted data as Zarr v3 stores for scalable numerical access.
Uses selected OME-NGFF-inspired metadata conventions but does not claim full NGFF compliance.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any

import zarr

from oct_converter_app.exporters.base import BaseExporter, ExportError, sanitize_path_component
from oct_converter_app.models import OCTStudy


class ZarrExporter(BaseExporter):
    """Exporter for Zarr v3 format.

    Exports OCT volumes as 3D arrays (z, y, x) and optionally fundus images
    as 2D/3D arrays in a Zarr v3 store.

    The exported arrays preserve the original numeric data without
    normalization or conversion. Uses Zstandard compression with B-scan
    chunking strategy (1, height, width) for efficient B-scan access.

    Attributes:
        name: Exporter name ('zarr').
    """

    name = "zarr"

    def export(
        self, study: OCTStudy, output_dir: Path | str, options: dict | None = None
    ) -> list[Path]:
        """Export study data as a Zarr v3 store.

        Creates a Zarr store with the OCT volume as the primary 'volume' array.
        If fundus is available and cleanly supported, includes it as 'fundus'.

        Args:
            study: The OCTStudy containing extracted data.
            output_dir: Directory to write Zarr store.
            options: Optional configuration.
                     Keys: 'overwrite' (bool, default True),
                           'oct_filename' (str, stem name for Zarr store),
                           'fundus_filename' (str, optional separate fundus array name).

        Returns:
            List containing path to created Zarr store directory.

        Raises:
            ExportError: If export fails.
        """
        output_path = self._ensure_output_dir(output_dir)
        overwrite = options.get("overwrite", True) if options else True

        # Determine Zarr store name
        filename_stem = options.get("oct_filename") if options else None
        if filename_stem:
            filename_stem = sanitize_path_component(filename_stem, default="oct")
            # Remove .zarr suffix if present to avoid double extension
            if filename_stem.endswith(".zarr"):
                filename_stem = filename_stem[:-5]
        else:
            patient_id = sanitize_path_component(study.patient_id, default="unknown")
            filename_stem = f"{patient_id}_oct"

        zarr_store_name = f"{filename_stem}.zarr"
        zarr_store_path = output_path / zarr_store_name

        # Check for existing store
        if not overwrite and zarr_store_path.exists():
            raise ExportError(f"Zarr store already exists and overwrite is disabled: {zarr_store_path}")

        created_files = []

        try:
            # Prepare metadata attributes
            attributes = self._build_metadata_attributes(study)

            # Check if store exists and remove if overwrite is enabled
            if zarr_store_path.exists() and overwrite:
                import shutil
                shutil.rmtree(zarr_store_path)

            created_zarr_store = False

            # Create Zarr store and write OCT volume
            if study.oct_volume is not None and study.oct_volume.volume:
                # Stack B-scans into 3D array: (z, y, x)
                volume_array = np.stack(study.oct_volume.volume, axis=0)

                # Determine chunking: (1, height, width) for B-scan access pattern
                num_bscans, height, width = volume_array.shape
                chunks = (1, height, width)

                # Create Zarr v3 group first
                root_group = zarr.open_group(
                    store=str(zarr_store_path),
                    mode="w",
                    zarr_format=3,
                )

                # Add study-level attributes to group
                for key, value in attributes.items():
                    root_group.attrs[key] = value

                # Create volume array within the group
                volume_arr = root_group.create_array(
                    name="volume",
                    shape=volume_array.shape,
                    chunks=chunks,
                    dtype=volume_array.dtype,
                    compressors=[zarr.codecs.ZstdCodec()],
                    dimension_names=("z", "y", "x"),
                )

                # Write data
                volume_arr[:] = volume_array

                created_files.append(zarr_store_path)
                created_zarr_store = True

            # Optionally add fundus as secondary array if available
            if study.fundus is not None and study.fundus.image.size > 0:
                fundus_image = np.asarray(study.fundus.image)

                # If we already created the store, open it and add fundus array
                if created_zarr_store:
                    root = zarr.open_group(store=str(zarr_store_path), mode="r+", zarr_format=3)
                else:
                    # Create new group if no OCT volume was written
                    root = zarr.open_group(store=str(zarr_store_path), mode="w", zarr_format=3)
                    # Add study-level attributes to group
                    for key, value in attributes.items():
                        root.attrs[key] = value

                # Add fundus array with appropriate dimension names
                if fundus_image.ndim == 2:
                    fundus_dims = ("y", "x")
                elif fundus_image.ndim == 3:
                    fundus_dims = ("y", "x", "c")
                else:
                    fundus_dims = tuple(f"dim_{i}" for i in range(fundus_image.ndim))

                fundus_arr = root.create_array(
                    name="fundus",
                    shape=fundus_image.shape,
                    chunks=fundus_image.shape,  # Single chunk for fundus
                    dtype=fundus_image.dtype,
                    compressors=[zarr.codecs.ZstdCodec()],
                    dimension_names=fundus_dims,
                )
                fundus_arr[:] = fundus_image

                # If we only wrote fundus (no OCT), ensure store path is tracked
                if zarr_store_path not in created_files:
                    created_files.append(zarr_store_path)

        except ExportError:
            raise
        except Exception as e:
            raise ExportError(f"Failed to export to Zarr: {e}") from e

        if not created_files:
            raise ExportError("No data available to export (no OCT volume or fundus)")

        return created_files

    def _build_metadata_attributes(self, study: OCTStudy) -> dict[str, Any]:
        """Build metadata attributes for Zarr store.

        Includes OME-NGFF-inspired axes information, physical spacing where available,
        source format, and provenance information.

        Args:
            study: The OCTStudy to extract metadata from.

        Returns:
            Dictionary of metadata attributes.
        """
        attributes: dict[str, Any] = {}

        # OME-NGFF-inspired axes specification
        if study.oct_volume is not None and study.oct_volume.volume:
            vol = study.oct_volume.volume
            if isinstance(vol, list) and len(vol) > 0:
                num_bscans = len(vol)
                # Get spatial dimensions from first B-scan
                first_scan = vol[0]
                if hasattr(first_scan, 'shape'):
                    height, width = first_scan.shape[:2]
                else:
                    height, width = None, None

                axes = [
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ]
                attributes["axes"] = axes

                # Physical spacing from OCT volume if available
                pixel_spacing = getattr(study.oct_volume, 'pixel_spacing', None)
                if pixel_spacing is not None:
                    # pixel_spacing typically contains (axial, lateral) or similar
                    # Map to z, y, x spacing conservatively
                    spacing = {}
                    if isinstance(pixel_spacing, (list, tuple)) and len(pixel_spacing) >= 2:
                        # Assume (lateral_y, lateral_x) or similar
                        spacing["y"] = float(pixel_spacing[0])
                        spacing["x"] = float(pixel_spacing[1])
                        if len(pixel_spacing) >= 3:
                            spacing["z"] = float(pixel_spacing[2])
                    elif isinstance(pixel_spacing, (int, float)):
                        # Uniform spacing
                        spacing["x"] = float(pixel_spacing)
                        spacing["y"] = float(pixel_spacing)
                        spacing["z"] = float(pixel_spacing)

                    if spacing:
                        attributes["scale"] = spacing

        # Source format
        attributes["source_format"] = study.source_format

        # Provenance information
        if study.provenance is not None:
            provenance_attrs = {}
            if study.provenance.source_path:
                provenance_attrs["source_path"] = str(study.provenance.source_path)
            provenance_attrs["source_format"] = study.provenance.source_format
            if study.provenance.processing_timestamp:
                provenance_attrs["processing_timestamp"] = study.provenance.processing_timestamp.isoformat()
            if study.provenance.file_hash:
                provenance_attrs["file_hash"] = study.provenance.file_hash
            if study.provenance.reader_version:
                provenance_attrs["reader_version"] = study.provenance.reader_version
            attributes["provenance"] = provenance_attrs

        # Vendor/device information
        if study.vendor_device is not None:
            vendor_attrs = {}
            if study.vendor_device.manufacturer:
                vendor_attrs["manufacturer"] = study.vendor_device.manufacturer
            if study.vendor_device.model:
                vendor_attrs["model"] = study.vendor_device.model
            if study.vendor_device.software_version:
                vendor_attrs["software_version"] = study.vendor_device.software_version
            if vendor_attrs:
                attributes["vendor_device"] = vendor_attrs

        return attributes

    def supports_oct(self, study: OCTStudy) -> bool:
        """Check if OCT volume export is possible."""
        return (
            study.oct_volume is not None
            and len(study.oct_volume.volume) > 0
        )

    def supports_fundus(self, study: OCTStudy) -> bool:
        """Check if fundus image export is possible."""
        return study.fundus is not None and study.fundus.image.size > 0
