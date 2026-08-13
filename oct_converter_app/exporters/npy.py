"""NumPy array exporter for OCT volumes and fundus images.

Exports extracted data as .npy files for numerical analysis.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from oct_converter_app.exporters.base import BaseExporter, ExportError
from oct_converter_app.models import OCTStudy


class NpyExporter(BaseExporter):
    """Exporter for NumPy array format.

    Exports OCT volumes as 3D arrays (num_slices, height, width) and
    fundus images as 2D/3D arrays.

    The exported arrays preserve the original numeric data without
    normalization or conversion to 8-bit.

    Attributes:
        name: Exporter name ('npy').
    """

    name = "npy"

    def export(
        self, study: OCTStudy, output_dir: Path | str, options: dict | None = None
    ) -> list[Path]:
        """Export study data as NumPy .npy files.

        Creates separate files for OCT volume and fundus image if available.

        Args:
            study: The OCTStudy containing extracted data.
            output_dir: Directory to write .npy files.
            options: Optional configuration.
                     Keys: 'oct_filename', 'fundus_filename' to override defaults.

        Returns:
            List of paths to created .npy files.

        Raises:
            ExportError: If export fails.
        """
        output_path = self._ensure_output_dir(output_dir)
        created_files = []

        # Export OCT volume
        if study.oct_volume is not None and study.oct_volume.volume:
            try:
                filename = options.get("oct_filename") if options else None
                if not filename:
                    patient_id = study.patient_id or "unknown"
                    filename = f"{patient_id}_oct.npy"

                filepath = output_path / filename

                # Stack B-scans into 3D array
                # Shape: (num_slices, height, width)
                volume_array = np.stack(study.oct_volume.volume, axis=0)

                np.save(filepath, volume_array)
                created_files.append(filepath)

            except Exception as e:
                raise ExportError(f"Failed to export OCT volume to NPY: {e}") from e

        # Export fundus image
        if study.fundus is not None and study.fundus.image.size > 0:
            try:
                filename = options.get("fundus_filename") if options else None
                if not filename:
                    patient_id = study.patient_id or "unknown"
                    filename = f"{patient_id}_fundus.npy"

                filepath = output_path / filename

                np.save(filepath, study.fundus.image)
                created_files.append(filepath)

            except Exception as e:
                raise ExportError(f"Failed to export fundus image to NPY: {e}") from e

        if not created_files:
            raise ExportError("No data available to export (no OCT volume or fundus)")

        return created_files

    def supports_oct(self, study: OCTStudy) -> bool:
        """Check if OCT volume export is possible."""
        return (
            study.oct_volume is not None
            and len(study.oct_volume.volume) > 0
        )

    def supports_fundus(self, study: OCTStudy) -> bool:
        """Check if fundus image export is possible."""
        return study.fundus is not None and study.fundus.image.size > 0
