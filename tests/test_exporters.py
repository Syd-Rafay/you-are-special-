"""Tests for exporters."""

import json
from datetime import date, datetime
import numpy as np
import pytest
from pathlib import Path

from oct_converter_app.exporters import (
    DicomExporter, NpyExporter, ImageExporter, MetadataExporter, sanitize_path_component
)
from oct_converter_app.exporters.base import ExportError
from oct_converter_app.exporters.metadata import NumpyEncoder
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


class TestPathTraversalProtection:
    """Regression tests for path traversal vulnerability prevention."""

    @pytest.mark.parametrize(
        "malicious_id",
        [
            "../../evil",
            "../../../tmp/evil",
            "/absolute/path",
            r"..\..\evil",
            "\x00\x01evil\x1f",
            ".././../evil",
            "....//evil",
        ],
    )
    def test_sanitize_path_component(self, malicious_id):
        """Test that sanitize_path_component strips traversal tokens and separators."""
        sanitized = sanitize_path_component(malicious_id, default="unknown")
        assert "/" not in sanitized
        assert "\\" not in sanitized
        assert "\x00" not in sanitized
        assert not sanitized.startswith("..")
        assert sanitized != "."
        assert sanitized != ".."

    @pytest.mark.parametrize(
        "malicious_id",
        [
            "../../evil",
            "../../../tmp/evil",
            "/absolute/path",
            r"..\..\evil",
            "\x00\x01evil\x1f",
        ],
    )
    def test_npy_exporter_path_traversal(self, tmp_path, malicious_id):
        """Test NpyExporter prevents files from being written outside output_dir."""
        output_dir = tmp_path / "output"
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")

        oct_vol = OCTVolumeWithMetaData(
            volume=[np.zeros((10, 10), dtype=np.uint16)],
            patient_id=malicious_id,
        )
        study = OCTStudy(source_path=src_file, source_format="fds", oct_volume=oct_vol)

        exporter = NpyExporter()
        files = exporter.export(study, output_dir)

        assert len(files) == 1
        output_dir_resolved = output_dir.resolve()
        for f in files:
            assert f.resolve().is_relative_to(output_dir_resolved)
            assert f.exists()
        # Original patient_id in study metadata is NOT mutated
        assert study.patient_id == malicious_id

    @pytest.mark.parametrize(
        "malicious_id",
        [
            "../../evil",
            "../../../tmp/evil",
            "/absolute/path",
            r"..\..\evil",
            "\x00\x01evil\x1f",
        ],
    )
    def test_image_exporter_path_traversal(self, tmp_path, malicious_id):
        """Test ImageExporter prevents files from being written outside output_dir."""
        output_dir = tmp_path / "output"
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")

        oct_vol = OCTVolumeWithMetaData(
            volume=[np.zeros((10, 10), dtype=np.uint8)],
            patient_id=malicious_id,
        )
        study = OCTStudy(source_path=src_file, source_format="fds", oct_volume=oct_vol)

        exporter = ImageExporter()
        files = exporter.export(study, output_dir)

        assert len(files) > 0
        output_dir_resolved = output_dir.resolve()
        for f in files:
            assert f.resolve().is_relative_to(output_dir_resolved)
            assert f.exists()
        assert study.patient_id == malicious_id

    @pytest.mark.parametrize(
        "malicious_id",
        [
            "../../evil",
            "../../../tmp/evil",
            "/absolute/path",
            r"..\..\evil",
            "\x00\x01evil\x1f",
        ],
    )
    def test_metadata_exporter_path_traversal(self, tmp_path, malicious_id):
        """Test MetadataExporter prevents files from being written outside output_dir."""
        output_dir = tmp_path / "output"
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")

        oct_vol = OCTVolumeWithMetaData(
            volume=[np.zeros((10, 10), dtype=np.uint16)],
            patient_id=malicious_id,
        )
        study = OCTStudy(source_path=src_file, source_format="fds", oct_volume=oct_vol)

        exporter = MetadataExporter()
        files = exporter.export(study, output_dir)

        assert len(files) == 1
        output_dir_resolved = output_dir.resolve()
        for f in files:
            assert f.resolve().is_relative_to(output_dir_resolved)
            assert f.exists()
        assert study.patient_id == malicious_id


class TestOverwriteBehavior:
    """Test overwrite and collision handling across exporters."""

    def test_repeated_export_overwrites_by_default(self, tmp_path):
        """Repeated export into same output directory succeeds when overwrite=True (default)."""
        output_dir = tmp_path / "output"
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")

        oct_vol = OCTVolumeWithMetaData(
            volume=[np.zeros((10, 10), dtype=np.uint16)],
            patient_id="PAT001",
        )
        study = OCTStudy(source_path=src_file, source_format="fds", oct_volume=oct_vol)

        exporter = NpyExporter()
        files1 = exporter.export(study, output_dir)
        assert len(files1) == 1

        # Second export overwrites without error
        files2 = exporter.export(study, output_dir, options={"overwrite": True})
        assert len(files2) == 1

    def test_overwrite_disabled_raises_error(self, tmp_path):
        """Exporting with overwrite=False raises ExportError when file already exists."""
        output_dir = tmp_path / "output"
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")

        oct_vol = OCTVolumeWithMetaData(
            volume=[np.zeros((10, 10), dtype=np.uint16)],
            patient_id="PAT001",
        )
        study = OCTStudy(source_path=src_file, source_format="fds", oct_volume=oct_vol)

        exporter = NpyExporter()
        exporter.export(study, output_dir)

        with pytest.raises(ExportError, match="overwrite is disabled"):
            exporter.export(study, output_dir, options={"overwrite": False})


class TestMetadataEncoderRobustness:
    """Test JSON serialization of bytes and non-standard types."""

    def test_bytes_and_types_serialization(self, tmp_path):
        """Test encoding bytes, bytearray, NumPy scalars/arrays, datetimes, and Paths."""
        data = {
            "binary_header": b"\x00\x01\x02\xff",
            "binary_mutable": bytearray(b"hello"),
            "int_val": np.int64(42),
            "float_val": np.float32(3.14),
            "arr": np.array([1, 2, 3]),
            "bool_val": np.bool_(True),
            "now": datetime(2026, 8, 13, 12, 0, 0),
            "today": date(2026, 8, 13),
            "path": Path("/tmp/test.fds"),
            "nested": {
                "tags": {1, 2, 3},
                "raw_bytes": b"raw_data",
            },
        }

        encoded = json.dumps(data, cls=NumpyEncoder)
        decoded = json.loads(encoded)

        assert decoded["binary_header"] == "000102ff"
        assert decoded["binary_mutable"] == "68656c6c6f"
        assert decoded["int_val"] == 42
        assert decoded["float_val"] == pytest.approx(3.14, abs=1e-2)
        assert decoded["arr"] == [1, 2, 3]
        assert decoded["bool_val"] is True
        assert decoded["now"] == "2026-08-13T12:00:00"
        assert decoded["today"] == "2026-08-13"
        assert decoded["path"] == "/tmp/test.fds"
        assert sorted(decoded["nested"]["tags"]) == [1, 2, 3]
        assert decoded["nested"]["raw_bytes"] == "7261775f64617461"

