"""Tests for exporters."""

import json
import numpy as np
import pytest
from pathlib import Path

from oct_converter_app.exporters import (
    DicomExporter, NpyExporter, ImageExporter, MetadataExporter
)
from oct_converter_app.exporters.base import ExportError
from oct_converter_app.models import OCTStudy
from oct_converter.image_types import OCTVolumeWithMetaData, FundusImageWithMetaData


class TestNpyExporter:
    """Test NumPy array exporter."""

    def test_export_oct_volume(self, tmp_path):
        """Test exporting OCT volume to NPY."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        oct_vol = OCTVolumeWithMetaData(
            volume=[np.zeros((100, 100), dtype=np.uint16)],
            patient_id="TEST"
        )
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds",
            oct_volume=oct_vol
        )
        
        exporter = NpyExporter()
        files = exporter.export(study, tmp_path / "output")
        
        assert len(files) == 1
        assert files[0].exists()
        assert files[0].suffix == ".npy"

    def test_export_fundus(self, tmp_path):
        """Test exporting fundus image to NPY."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        fundus = FundusImageWithMetaData(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            patient_id="TEST"
        )
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds",
            oct_volume=None,
            fundus=fundus
        )
        
        exporter = NpyExporter()
        files = exporter.export(study, tmp_path / "output")
        
        assert len(files) == 1
        assert files[0].exists()

    def test_no_data_raises_error(self, tmp_path):
        """Test exporting with no data raises error."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds",
            oct_volume=None,
            fundus=None
        )
        
        exporter = NpyExporter()
        with pytest.raises(ExportError):
            exporter.export(study, tmp_path / "output")


class TestImageExporter:
    """Test image exporter."""

    def test_export_bscans(self, tmp_path):
        """Test exporting B-scans as images."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        oct_vol = OCTVolumeWithMetaData(
            volume=[np.zeros((100, 100), dtype=np.uint16)],
            patient_id="TEST"
        )
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds",
            oct_volume=oct_vol
        )
        
        exporter = ImageExporter(format="png")
        files = exporter.export(study, tmp_path / "output")
        
        assert len(files) >= 1
        assert files[0].suffix == ".png"

    def test_export_fundus_image(self, tmp_path):
        """Test exporting fundus as image."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        fundus = FundusImageWithMetaData(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            patient_id="TEST"
        )
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds",
            oct_volume=None,
            fundus=fundus
        )
        
        exporter = ImageExporter(format="png")
        files = exporter.export(study, tmp_path / "output")
        
        assert len(files) == 1
        assert files[0].suffix == ".png"

    def test_unsupported_format_raises_error(self):
        """Test unsupported format raises error."""
        with pytest.raises(ValueError):
            ImageExporter(format="xyz")


class TestMetadataExporter:
    """Test metadata JSON exporter."""

    def test_export_metadata(self, tmp_path):
        """Test exporting metadata to JSON."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        oct_vol = OCTVolumeWithMetaData(
            volume=[np.zeros((100, 100), dtype=np.uint16)],
            patient_id="TEST",
            laterality="L"
        )
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds",
            oct_volume=oct_vol
        )
        
        exporter = MetadataExporter()
        files = exporter.export(study, tmp_path / "output")
        
        assert len(files) == 1
        assert files[0].suffix == ".json"
        
        # Verify JSON is valid
        with open(files[0]) as f:
            data = json.load(f)
        assert "source" in data
        assert "patient" in data


class TestDicomExporter:
    """Test DICOM exporter."""

    def test_dicom_exporter_creation(self):
        """Test creating DICOM exporter."""
        exporter = DicomExporter()
        assert exporter.name == "dicom"
        assert exporter.rows == 1024
        assert exporter.cols == 512
