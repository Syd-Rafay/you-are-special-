"""Tests for validation module."""

import numpy as np
import pytest

from oct_converter_app.validation import OctValidator, ValidationResult
from oct_converter.image_types import OCTVolumeWithMetaData, FundusImageWithMetaData


class TestValidationResult:
    """Test ValidationResult model."""

    def test_default_valid(self):
        """Test default result is valid."""
        result = ValidationResult()
        assert result.is_valid is True
        assert len(result.warnings) == 0
        assert len(result.errors) == 0

    def test_add_warning(self):
        """Test adding warning."""
        result = ValidationResult()
        result.add_warning("Test warning")
        assert len(result.warnings) == 1
        assert result.is_valid is True  # Warnings don't invalidate

    def test_add_error(self):
        """Test adding error."""
        result = ValidationResult()
        result.add_error("Test error")
        assert len(result.errors) == 1
        assert result.is_valid is False

    def test_merge(self):
        """Test merging results."""
        result1 = ValidationResult()
        result1.add_warning("Warning 1")
        
        result2 = ValidationResult()
        result2.add_error("Error 2")
        
        result1.merge(result2)
        assert len(result1.warnings) == 1
        assert len(result1.errors) == 1
        assert result1.is_valid is False


class TestOctValidator:
    """Test OCT volume validation."""

    def test_validate_none_oct(self):
        """Test validating None OCT volume."""
        result = OctValidator.validate_oct_volume(None)
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_empty_volume(self):
        """Test validating empty volume."""
        oct_vol = OCTVolumeWithMetaData(
            volume=[]
        )
        result = OctValidator.validate_oct_volume(oct_vol)
        assert result.is_valid is False
        assert "empty" in str(result.errors).lower()

    def test_validate_valid_oct(self):
        """Test validating valid OCT volume."""
        oct_vol = OCTVolumeWithMetaData(
            volume=[np.zeros((100, 100), dtype=np.uint16)],
            pixel_spacing=[0.1, 0.1, 0.1],
            patient_id="TEST",
            laterality="L"
        )
        result = OctValidator.validate_oct_volume(oct_vol)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_missing_pixel_spacing(self):
        """Test warning for missing pixel spacing."""
        oct_vol = OCTVolumeWithMetaData(
            volume=[np.zeros((100, 100), dtype=np.uint16)],
            pixel_spacing=None,
            patient_id="TEST",
            laterality="L"
        )
        result = OctValidator.validate_oct_volume(oct_vol)
        assert result.is_valid is True  # Still valid
        assert any("pixel spacing" in w.lower() for w in result.warnings)


class TestFundusValidator:
    """Test fundus image validation."""

    def test_validate_none_fundus(self):
        """Test validating None fundus."""
        result = OctValidator.validate_fundus_image(None)
        # Fundus is optional, so this should be a warning not error
        assert len(result.warnings) > 0

    def test_validate_empty_fundus(self):
        """Test validating empty fundus image."""
        fundus = FundusImageWithMetaData(
            image=np.array([], dtype=np.uint8)
        )
        result = OctValidator.validate_fundus_image(fundus)
        assert result.is_valid is False

    def test_validate_valid_fundus(self):
        """Test validating valid fundus image."""
        fundus = FundusImageWithMetaData(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            patient_id="TEST",
            laterality="R"
        )
        result = OctValidator.validate_fundus_image(fundus)
        assert result.is_valid is True
        assert len(result.errors) == 0


class TestStudyValidator:
    """Test complete study validation."""

    def test_validate_complete_study(self):
        """Test validating study with both OCT and fundus."""
        oct_vol = OCTVolumeWithMetaData(
            volume=[np.zeros((100, 100), dtype=np.uint16)],
            pixel_spacing=[0.1, 0.1, 0.1],
            patient_id="TEST",
            laterality="L"
        )
        fundus = FundusImageWithMetaData(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            patient_id="TEST",
            laterality="L"
        )
        result = OctValidator.validate_study(oct_vol, fundus)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_no_data(self):
        """Test validating study with no data."""
        result = OctValidator.validate_study(None, None)
        assert result.is_valid is False
        assert len(result.errors) > 0
