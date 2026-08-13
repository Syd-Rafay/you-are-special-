"""Tests for data models."""

from pathlib import Path
from datetime import datetime

import numpy as np
import pytest

from oct_converter_app.models import OCTStudy, Provenance, Capabilities
from oct_converter.image_types import OCTVolumeWithMetaData, FundusImageWithMetaData


class TestProvenance:
    """Test Provenance model."""

    def test_create_provenance(self, tmp_path):
        """Test creating provenance without hash."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake content")
        
        prov = Provenance.create(
            source_path=src_file,
            source_format="fds",
            compute_hash=False
        )
        
        assert prov.source_path == src_file
        assert prov.source_format == "fds"
        assert prov.file_hash is None
        assert isinstance(prov.processing_timestamp, datetime)

    def test_create_provenance_with_hash(self, tmp_path):
        """Test creating provenance with SHA-256 hash."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"test content for hashing")
        
        prov = Provenance.create(
            source_path=src_file,
            source_format="fds",
            compute_hash=True
        )
        
        assert prov.file_hash is not None
        assert len(prov.file_hash) == 64  # SHA-256 hex length


class TestCapabilities:
    """Test Capabilities model."""

    def test_empty_capabilities(self):
        """Test default capabilities."""
        caps = Capabilities()
        assert caps.has_oct_volume is False
        assert caps.has_fundus is False
        assert caps.num_bscans == 0

    def test_from_study_with_oct(self, tmp_path):
        """Test deriving capabilities from study with OCT."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        # Create mock OCT volume
        oct_vol = OCTVolumeWithMetaData(
            volume=[np.zeros((100, 100), dtype=np.uint16)],
            pixel_spacing=[0.1, 0.1, 0.1],
            num_slices=1
        )
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds",
            oct_volume=oct_vol,
            fundus=None
        )
        
        caps = Capabilities.from_study(study)
        assert caps.has_oct_volume is True
        assert caps.num_bscans == 1


class TestOCTStudy:
    """Test OCTStudy model."""

    def test_create_empty_study(self, tmp_path):
        """Test creating minimal study."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds"
        )
        
        assert study.source_path == src_file
        assert study.source_format == "fds"
        assert study.oct_volume is None
        assert study.fundus is None
        assert isinstance(study.metadata, dict)
        assert isinstance(study.warnings, list)

    def test_patient_id_property(self, tmp_path):
        """Test patient_id extraction."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        oct_vol = OCTVolumeWithMetaData(
            volume=[np.zeros((100, 100), dtype=np.uint16)],
            patient_id="TEST123",
            num_slices=1
        )
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds",
            oct_volume=oct_vol
        )
        
        assert study.patient_id == "TEST123"

    def test_add_warning(self, tmp_path):
        """Test adding warnings."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds"
        )
        
        study.add_warning("Test warning")
        assert len(study.warnings) == 1
        assert "Test warning" in study.warnings

    def test_has_errors(self, tmp_path):
        """Test has_errors method."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        # No data = errors
        study = OCTStudy(
            source_path=src_file,
            source_format="fds",
            oct_volume=None,
            fundus=None
        )
        assert study.has_errors() is True
        
        # With OCT = no errors
        oct_vol = OCTVolumeWithMetaData(
            volume=[np.zeros((100, 100), dtype=np.uint16)],
            num_slices=1
        )
        study.oct_volume = oct_vol
        assert study.has_errors() is False
