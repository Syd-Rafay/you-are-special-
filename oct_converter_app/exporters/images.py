"""Image exporter for OCT B-scans and fundus images.

Exports extracted data as PNG/JPEG image files.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from oct_converter_app.exporters.base import BaseExporter, ExportError, sanitize_path_component
from oct_converter_app.models import OCTStudy


class ImageExporter(BaseExporter):
    """Exporter for image formats (PNG, JPEG, TIFF).

    Exports OCT B-scans as individual image files and fundus images
    as single files. For OCT volumes, each B-scan is saved separately.

    Note: This exporter converts to 8-bit for display purposes.
    The original numeric data should be exported via NpyExporter
    if quantitative analysis is needed.

    Attributes:
        name: Exporter name ('images').
        format: Image format ('png', 'jpg', 'tiff', etc.).
    """

    name = "images"

    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}

    def __init__(self, format: str = "png"):
        """Initialize image exporter.

        Args:
            format: Output image format (default: 'png').
                    Must be a supported extension.
        """
        fmt = format.lower()
        if not fmt.startswith("."):
            fmt = f".{fmt}"

        if fmt not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format: {format}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        self.format = fmt

    def export(
        self, study: OCTStudy, output_dir: Path | str, options: dict | None = None
    ) -> list[Path]:
        """Export study data as image files.

        Creates one file per B-scan for OCT volumes, and one file
        for fundus images.

        Args:
            study: The OCTStudy containing extracted data.
            output_dir: Directory to write image files.
            options: Optional configuration.
                     Keys: 'prefix' to override default filename prefix,
                           'format' to override instance format,
                           'overwrite' (bool, default True) whether to overwrite existing files.

        Returns:
            List of paths to created image files.

        Raises:
            ExportError: If export fails.
        """
        output_path = self._ensure_output_dir(output_dir)
        created_files = []
        overwrite = options.get("overwrite", True) if options else True

        # Get format override from options
        img_format = options.get("format", self.format) if options else self.format
        if not img_format.startswith("."):
            img_format = f".{img_format}"

        # Export OCT B-scans
        if study.oct_volume is not None and study.oct_volume.volume:
            try:
                prefix = options.get("prefix") if options else None
                if prefix:
                    prefix = sanitize_path_component(prefix, default="bscan")
                else:
                    prefix = sanitize_path_component(study.patient_id, default="bscan")

                # Create subdirectory for B-scans
                bscan_dir = output_path / f"{prefix}_bscans"
                bscan_dir.mkdir(parents=True, exist_ok=True)

                num_digits = len(str(len(study.oct_volume.volume)))

                if not overwrite:
                    first_file = bscan_dir / f"{prefix}_{0:0{num_digits}d}{img_format}"
                    if first_file.exists():
                        raise ExportError(f"File already exists and overwrite is disabled: {first_file}")

                for i, scan in enumerate(study.oct_volume.volume):
                    filename = f"{prefix}_{i:0{num_digits}d}{img_format}"
                    filepath = bscan_dir / filename

                    if not overwrite and filepath.exists():
                        raise ExportError(f"File already exists and overwrite is disabled: {filepath}")

                    # Convert to 8-bit for display
                    if scan.dtype != np.uint8:
                        # Normalize to 0-255 range
                        scan_min = scan.min()
                        scan_max = scan.max()
                        if scan_max > scan_min:
                            scan_8bit = ((scan - scan_min) / (scan_max - scan_min) * 255).astype(
                                np.uint8
                            )
                        else:
                            scan_8bit = np.zeros_like(scan, dtype=np.uint8)
                    else:
                        scan_8bit = scan

                    # OpenCV expects BGR, but grayscale has no channel order
                    if len(scan_8bit.shape) == 2:
                        cv2.imwrite(str(filepath), scan_8bit)
                    else:
                        # RGB to BGR conversion for color
                        scan_bgr = cv2.cvtColor(scan_8bit, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(filepath), scan_bgr)

                    created_files.append(filepath)

            except ExportError:
                raise
            except Exception as e:
                raise ExportError(f"Failed to export OCT B-scans: {e}") from e

        # Export fundus image
        if study.fundus is not None and study.fundus.image.size > 0:
            try:
                prefix = options.get("prefix") if options else None
                if prefix:
                    prefix = sanitize_path_component(prefix, default="fundus")
                else:
                    prefix = sanitize_path_component(study.patient_id, default="fundus")

                filename = f"{prefix}_fundus{img_format}"
                filepath = output_path / filename

                if not overwrite and filepath.exists():
                    raise ExportError(f"File already exists and overwrite is disabled: {filepath}")

                image = study.fundus.image

                # Convert to 8-bit if needed
                if image.dtype != np.uint8:
                    img_min = image.min()
                    img_max = image.max()
                    if img_max > img_min:
                        image_8bit = ((image - img_min) / (img_max - img_min) * 255).astype(
                            np.uint8
                        )
                    else:
                        image_8bit = np.zeros_like(image, dtype=np.uint8)
                else:
                    image_8bit = image

                # Convert RGB to BGR for OpenCV
                if len(image_8bit.shape) == 3 and image_8bit.shape[2] == 3:
                    image_bgr = cv2.cvtColor(image_8bit, cv2.COLOR_RGB2BGR)
                else:
                    image_bgr = image_8bit

                cv2.imwrite(str(filepath), image_bgr)
                created_files.append(filepath)

            except ExportError:
                raise
            except Exception as e:
                raise ExportError(f"Failed to export fundus image: {e}") from e

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
