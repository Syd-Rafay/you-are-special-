"""Data models for OCT processing results.

This module defines lightweight container classes for holding the results
of OCT file processing. These models wrap the existing OCTVolumeWithMetaData
and FundusImageWithMetaData classes rather than duplicating their data.
"""

from __future__ import annotations

import hashlib
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from oct_converter.image_types import FundusImageWithMetaData, OCTVolumeWithMetaData


@dataclass(frozen=True)
class VendorDevice:
    """Vendor/device identification information.

    Attributes:
        manufacturer: Device manufacturer name (e.g., "Topcon", "Heidelberg").
        model: Device model name/number.
        software_version: Software/firmware version string.

    All fields are optional. Do not fabricate values—use None when unknown.
    Values should be populated from reliable vendor metadata, not assumptions.
    """

    manufacturer: str | None = None
    model: str | None = None
    software_version: str | None = None


@dataclass
class DerivedProduct:
    """A derived imaging product attached to an OCT study.

    This is an extension point for future-derived products such as:
    - OCTA en-face images
    - Retinal layer segmentation masks
    - Thickness maps
    - Surface renderings
    - Contours/annotations

    This class is intentionally minimal. The `data` field is a temporary
    extension point and may be refined in future phases.

    Attributes:
        product_type: Type identifier for the derived product
                     (e.g., "octa_enface", "layer_segmentation", "thickness_map").
        data: The derived data (array, image, mesh, etc.). Currently untyped;
              this is a placeholder for future refinement.
        metadata: Optional metadata describing the derivation method, parameters, etc.
        creation_method: Description of how this product was derived
                        (e.g., algorithm name, software version).
    """

    product_type: str
    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    creation_method: str | None = None


@dataclass
class StudyCapabilities:
    """Capabilities report for a processed study.

    Indicates which data modalities and features are available from the source file.

    This class provides both current capabilities and future-facing flags.
    Future flags (has_octa, has_layer_segmentation) default to False and
    will be populated when those features are implemented.

    Attributes:
        has_oct_volume: Whether OCT volume data is available (non-empty).
        has_fundus: Whether fundus image is available (non-empty).
        has_metadata: Whether normalized metadata is available.
        has_pixel_spacing: Whether pixel spacing calibration is available.
        has_contours: Whether segmentation contours are available.
        has_octa: Whether OCT angiography data is available (future).
        has_layer_segmentation: Whether layer segmentation is available (future).
        num_bscans: Number of B-scans (0 if no OCT or empty).
        fundus_shape: Shape of fundus image tuple (H, W, C) or None.
    """

    has_oct_volume: bool = False
    has_fundus: bool = False
    has_metadata: bool = False
    has_pixel_spacing: bool = False
    has_contours: bool = False
    has_octa: bool = False
    has_layer_segmentation: bool = False
    num_bscans: int = 0
    fundus_shape: tuple[int, ...] | None = None

    @classmethod
    def from_study(cls, study: "OCTStudy") -> "StudyCapabilities":
        """Derive capabilities from an OCTStudy.

        This method inspects the current state of the study and determines
        which capabilities are present. It performs robust checks to ensure
        that "has" flags reflect actual data presence, not just object existence.

        For OCT volume, we check that slices exist and have non-zero size.
        We do NOT check pixel values since zero-valued pixels are valid data.

        Args:
            study: The study to analyze.

        Returns:
            StudyCapabilities instance.
        """
        # Check OCT volume: must exist AND contain non-empty slices
        has_oct = False
        num_bscans = 0
        has_pixel_spacing = False
        has_contours = False

        if study.oct_volume is not None:
            volume = study.oct_volume.volume
            if volume is not None:
                # Check if volume is a list/array with actual data
                if isinstance(volume, list):
                    # Count slices that exist and have non-zero size
                    # Note: We check size/shape, not pixel values (zeros are valid data)
                    valid_slices = [
                        s for s in volume
                        if s is not None and hasattr(s, 'size') and s.size > 0
                    ]
                    num_bscans = len(valid_slices)
                    has_oct = num_bscans > 0
                elif hasattr(volume, "__len__") and len(volume) > 0:
                    # Handle array-like volumes
                    try:
                        if hasattr(volume, 'size') and volume.size > 0:
                            has_oct = True
                            num_bscans = len(volume) if hasattr(volume, "__len__") else 1
                    except (ValueError, TypeError):
                        pass

            if study.oct_volume.pixel_spacing is not None:
                has_pixel_spacing = True

            if study.oct_volume.contours is not None:
                has_contours = True

        # Check fundus: must exist AND have non-zero size
        has_fundus = False
        fundus_shape = None

        if study.fundus is not None:
            fundus_image = study.fundus.image
            if fundus_image is not None:
                try:
                    if hasattr(fundus_image, "size") and fundus_image.size > 0:
                        has_fundus = True
                        fundus_shape = tuple(fundus_image.shape)
                except (AttributeError, TypeError):
                    # Fallback for PIL images or other types
                    if hasattr(fundus_image, "size"):
                        size = fundus_image.size
                        if isinstance(size, tuple) and len(size) >= 2:
                            has_fundus = size[0] > 0 and size[1] > 0
                            fundus_shape = size

        return cls(
            has_oct_volume=has_oct,
            has_fundus=has_fundus,
            has_metadata=bool(study.metadata),
            has_pixel_spacing=has_pixel_spacing,
            has_contours=has_contours,
            has_octa=False,  # Future feature
            has_layer_segmentation=False,  # Future feature
            num_bscans=num_bscans,
            fundus_shape=fundus_shape,
        )


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
        vendor_device: Vendor/device identification (may be None if unknown).
        derived_products: List of derived imaging products (empty by default).

    Note on capabilities:
        The `capabilities` attribute is now a computed property that derives
        its values from the current state of the study. This prevents stale
        capability reports when the study's OCT volume or fundus is modified
        after construction.
    """

    source_path: Path
    source_format: str
    oct_volume: OCTVolumeWithMetaData | None = None
    fundus: FundusImageWithMetaData | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: Provenance | None = None
    vendor_device: VendorDevice | None = None
    _derived_products: list[DerivedProduct] = field(default_factory=list)

    @property
    def capabilities(self) -> StudyCapabilities:
        """Compute current capabilities from the study state.

        This is a computed property to ensure capabilities always reflect
        the current state of the study, even if oct_volume or fundus are
        modified after construction.

        Returns:
            StudyCapabilities instance reflecting current state.
        """
        return StudyCapabilities.from_study(self)

    @property
    def derived_products(self) -> list[DerivedProduct]:
        """Get the list of derived products attached to this study."""
        return self._derived_products

    def add_derived_product(self, product: DerivedProduct) -> None:
        """Add a derived product to this study.

        Args:
            product: A DerivedProduct instance to add.

        Raises:
            TypeError: If product is not a DerivedProduct instance.
        """
        if not isinstance(product, DerivedProduct):
            raise TypeError(
                f"Expected DerivedProduct, got {type(product).__name__}"
            )
        self._derived_products.append(product)

    def __post_init__(self):
        """Initialize derived products list if needed."""
        # Ensure _derived_products is initialized (for cases where it's not provided)
        if not hasattr(self, "_derived_products") or self._derived_products is None:
            object.__setattr__(self, "_derived_products", [])

    def __repr__(self) -> str:
        """Return a concise, PHI-safe string representation.

        This repr intentionally excludes raw_metadata and metadata content
        to prevent accidental exposure of patient information in logs or
        debugging output.

        Returns:
            Concise representation showing structure without PHI.
        """
        oct_info = "None"
        if self.oct_volume is not None:
            vol = self.oct_volume.volume
            if vol is not None:
                if isinstance(vol, list):
                    oct_info = f"{len(vol)} B-scans"
                elif hasattr(vol, "shape"):
                    oct_info = f"volume{vol.shape}"
                else:
                    oct_info = "<OCT volume>"
            else:
                oct_info = "<empty>"

        fundus_info = "None"
        if self.fundus is not None:
            img = self.fundus.image
            if img is not None and hasattr(img, "shape"):
                fundus_info = f"{img.shape}"
            elif hasattr(img, "size") and img.size > 0:
                fundus_info = f"<fundus image>"
            else:
                fundus_info = "<empty>"

        caps = self.capabilities
        caps_summary = (
            f"caps(oct={caps.has_oct_volume}, fundus={caps.has_fundus})"
        )

        return (
            f"OCTStudy("
            f"source_format={self.source_format!r}, "
            f"oct_volume={oct_info}, "
            f"fundus={fundus_info}, "
            f"{caps_summary}"
            f")"
        )

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
