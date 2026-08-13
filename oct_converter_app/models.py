"""Data models for OCT processing results.

This module defines lightweight container classes for holding the results
of OCT file processing. These models wrap the existing OCTVolumeWithMetaData
and FundusImageWithMetaData classes rather than duplicating their data.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from oct_converter.image_types import FundusImageWithMetaData, OCTVolumeWithMetaData


@dataclass
class Provenance:
    """Provenance information for a processed study.

    Attributes:
        source_path: Original file path.
        source_format: Detected format identifier.
        processing_timestamp: When the file was processed.
        file_hash: SHA-256 hash of the source file (if computed).
        reader_version: Version of the reader/package used.
    """

    source_path: Path
    source_format: str
    processing_timestamp: datetime = field(default_factory=datetime.now)
    file_hash: str | None = None
    reader_version: str | None = None

    @classmethod
    def create(
        cls,
        source_path: Path | str,
        source_format: str,
        compute_hash: bool = False,
        reader_version: str | None = None,
    ) -> "Provenance":
        """Create provenance information for a file.

        Args:
            source_path: Path to the source file.
            source_format: Format identifier.
            compute_hash: Whether to compute SHA-256 hash (adds I/O).
            reader_version: Version string to record.

        Returns:
            Provenance instance.
        """
        source_path = Path(source_path)

        file_hash = None
        if compute_hash and source_path.exists():
            sha256 = hashlib.sha256()
            with open(source_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            file_hash = sha256.hexdigest()

        return cls(
            source_path=source_path,
            source_format=source_format,
            file_hash=file_hash,
            reader_version=reader_version,
        )


@dataclass
class Capabilities:
    """Capabilities report for a processed study.

    Indicates which data modalities are available from the source file.

    Attributes:
        has_oct_volume: Whether OCT volume data is available.
        has_fundus: Whether fundus image is available.
        has_metadata: Whether metadata is available.
        has_pixel_spacing: Whether pixel spacing calibration is available.
        has_contours: Whether segmentation contours are available.
        num_bscans: Number of B-scans (0 if no OCT).
        fundus_shape: Shape of fundus image tuple (H, W, C) or None.
    """

    has_oct_volume: bool = False
    has_fundus: bool = False
    has_metadata: bool = False
    has_pixel_spacing: bool = False
    has_contours: bool = False
    num_bscans: int = 0
    fundus_shape: tuple[int, ...] | None = None

    @classmethod
    def from_study(cls, study: "OCTStudy") -> "Capabilities":
        """Derive capabilities from an OCTStudy.

        Args:
            study: The study to analyze.

        Returns:
            Capabilities instance.
        """
        has_oct = study.oct_volume is not None and len(study.oct_volume.volume) > 0
        has_fundus = study.fundus is not None and study.fundus.image.size > 0

        fundus_shape = None
        if study.fundus is not None:
            fundus_shape = tuple(study.fundus.image.shape)

        num_bscans = 0
        if study.oct_volume is not None:
            num_bscans = study.oct_volume.num_slices

        has_pixel_spacing = False
        if study.oct_volume is not None and study.oct_volume.pixel_spacing is not None:
            has_pixel_spacing = True

        has_contours = False
        if study.oct_volume is not None and study.oct_volume.contours is not None:
            has_contours = True

        return cls(
            has_oct_volume=has_oct,
            has_fundus=has_fundus,
            has_metadata=bool(study.metadata),
            has_pixel_spacing=has_pixel_spacing,
            has_contours=has_contours,
            num_bscans=num_bscans,
            fundus_shape=fundus_shape,
        )


@dataclass
class OCTStudy:
    """Container for processed OCT study results.

    This is a lightweight wrapper that holds references to the existing
    OCTVolumeWithMetaData and FundusImageWithMetaData objects returned
    by the oct_converter readers. It does NOT duplicate the pixel data.

    Attributes:
        source_path: Original file path.
        source_format: Detected format identifier (e.g., 'fds', 'fda').
        oct_volume: OCT volume with metadata (may be None).
        fundus: Fundus image with metadata (may be None).
        metadata: Common extracted metadata dictionary.
        raw_metadata: Raw vendor-specific metadata (preserved as-is).
        warnings: List of warning messages from processing.
        provenance: Source and processing information.
        capabilities: Report of available data modalities.
    """

    source_path: Path
    source_format: str
    oct_volume: OCTVolumeWithMetaData | None = None
    fundus: FundusImageWithMetaData | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: Provenance | None = None
    capabilities: Capabilities | None = None

    def __post_init__(self):
        """Compute capabilities after initialization."""
        if self.capabilities is None:
            self.capabilities = Capabilities.from_study(self)

    @property
    def patient_id(self) -> str | None:
        """Get patient ID from OCT volume or fundus."""
        if self.oct_volume and self.oct_volume.patient_id:
            return self.oct_volume.patient_id
        if self.fundus and self.fundus.patient_id:
            return self.fundus.patient_id
        return None

    @property
    def laterality(self) -> str | None:
        """Get laterality from OCT volume or fundus."""
        if self.oct_volume and self.oct_volume.laterality:
            return self.oct_volume.laterality
        if self.fundus and self.fundus.laterality:
            return self.fundus.laterality
        return None

    @property
    def acquisition_date(self) -> datetime | None:
        """Get acquisition date from OCT volume or fundus."""
        if self.oct_volume and self.oct_volume.acquisition_date:
            return self.oct_volume.acquisition_date
        # Fundus stores as string, try to parse
        if self.fundus and self.fundus.acquisition_date:
            try:
                return datetime.fromisoformat(self.fundus.acquisition_date)
            except (ValueError, TypeError):
                pass
        return None

    @property
    def volume_dimensions(self) -> tuple[int, int, int] | None:
        """Get OCT volume dimensions (num_slices, height, width)."""
        if self.oct_volume is None or not self.oct_volume.volume:
            return None
        h, w = self.oct_volume.volume[0].shape
        return (self.oct_volume.num_slices, h, w)

    @property
    def fundus_dimensions(self) -> tuple[int, int] | None:
        """Get fundus image dimensions (height, width)."""
        if self.fundus is None:
            return None
        shape = self.fundus.image.shape
        if len(shape) >= 2:
            return (shape[0], shape[1])
        return None

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def has_errors(self) -> bool:
        """Check if study has any extraction errors."""
        return self.oct_volume is None and self.fundus is None
