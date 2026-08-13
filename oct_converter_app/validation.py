"""Validation for OCT studies.

This module provides validation checks for extracted OCT data to ensure
data integrity before export.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from oct_converter.image_types import FundusImageWithMetaData, OCTVolumeWithMetaData


@dataclass
class ValidationResult:
    """Result of a validation check.

    Attributes:
        is_valid: Whether validation passed (no errors).
        warnings: Non-fatal issues detected.
        errors: Fatal issues that prevent processing.
    """

    is_valid: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        self.is_valid = len(self.errors) == 0

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        self.warnings.extend(other.warnings)
        self.errors.extend(other.errors)
        self.is_valid = self.is_valid and other.is_valid


class OctValidator:
    """Validator for OCT studies.

    Provides methods to validate extracted OCT volumes and fundus images.
    Distinguishes between fatal errors (prevent export) and warnings
    (non-fatal issues).
    """

    @staticmethod
    def validate_oct_volume(oct_volume: OCTVolumeWithMetaData | None) -> ValidationResult:
        """Validate an extracted OCT volume.

        Args:
            oct_volume: The OCT volume to validate (may be None).

        Returns:
            ValidationResult with any issues found.
        """
        result = ValidationResult()

        if oct_volume is None:
            result.add_error("OCT volume is None - no OCT data extracted")
            return result

        # Check volume exists and is not empty
        if not hasattr(oct_volume, "volume") or oct_volume.volume is None:
            result.add_error("OCT volume attribute is None")
            return result

        if len(oct_volume.volume) == 0:
            result.add_error("OCT volume is empty (no B-scans)")
            return result

        # Check first B-scan is valid
        try:
            first_scan = oct_volume.volume[0]
            if first_scan.size == 0:
                result.add_error("First B-scan is empty")
            elif not hasattr(first_scan, "shape") or len(first_scan.shape) != 2:
                result.add_error(f"Invalid B-scan shape: {getattr(first_scan, 'shape', 'unknown')}")
        except (IndexError, AttributeError) as e:
            result.add_error(f"Cannot access B-scan data: {e}")
            return result

        # Check dimension consistency across slices
        if len(oct_volume.volume) > 1:
            first_shape = oct_volume.volume[0].shape
            for i, scan in enumerate(oct_volume.volume[1:], start=1):
                if scan.shape != first_shape:
                    result.add_error(
                        f"B-scan {i} has inconsistent shape {scan.shape} "
                        f"(expected {first_shape})"
                    )
                    break  # Report only first inconsistency

        # Warnings for potentially missing metadata
        if oct_volume.pixel_spacing is None:
            result.add_warning("Missing pixel spacing calibration")

        if not oct_volume.patient_id:
            result.add_warning("Missing patient ID")

        if not oct_volume.laterality:
            result.add_warning("Missing laterality information")

        if oct_volume.num_slices != len(oct_volume.volume):
            result.add_warning(
                f"num_slices ({oct_volume.num_slices}) does not match "
                f"actual volume length ({len(oct_volume.volume)})"
            )

        return result

    @staticmethod
    def validate_fundus_image(
        fundus: FundusImageWithMetaData | None,
    ) -> ValidationResult:
        """Validate an extracted fundus image.

        Args:
            fundus: The fundus image to validate (may be None).

        Returns:
            ValidationResult with any issues found.
        """
        result = ValidationResult()

        if fundus is None:
            # Fundus is optional, so this is just a warning
            result.add_warning("No fundus image extracted (optional)")
            return result

        # Check image exists
        if not hasattr(fundus, "image") or fundus.image is None:
            result.add_error("Fundus image attribute is None")
            return result

        if fundus.image.size == 0:
            result.add_error("Fundus image is empty")
            return result

        if not hasattr(fundus.image, "shape") or len(fundus.image.shape) < 2:
            result.add_error(f"Invalid fundus image shape: {getattr(fundus.image, 'shape', 'unknown')}")
            return result

        # Warnings
        if not fundus.patient_id:
            result.add_warning("Missing patient ID in fundus metadata")

        if not fundus.laterality:
            result.add_warning("Missing laterality in fundus metadata")

        return result

    @staticmethod
    def validate_study(oct_volume: OCTVolumeWithMetaData | None, fundus: FundusImageWithMetaData | None) -> ValidationResult:
        """Validate a complete study (OCT + fundus).

        Args:
            oct_volume: OCT volume (may be None).
            fundus: Fundus image (may be None).

        Returns:
            Combined ValidationResult.
        """
        result = ValidationResult()

        oct_result = OctValidator.validate_oct_volume(oct_volume)
        fundus_result = OctValidator.validate_fundus_image(fundus)

        result.merge(oct_result)
        result.merge(fundus_result)

        # Critical: at least one modality must be present
        if oct_volume is None and fundus is None:
            result.add_error("No data extracted: both OCT volume and fundus are None")

        return result
