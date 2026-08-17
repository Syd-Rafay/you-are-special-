"""Tests for exporters."""

import json
from datetime import date, datetime
import numpy as np
import pytest
from pathlib import Path

from oct_converter_app.exporters import (
    DicomExporter, NpyExporter, ImageExporter, MetadataExporter, ZarrExporter, sanitize_path_component
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

    def test_dicom_exporter_supports_oct(self, tmp_path):
        """Test supports_oct checks in-memory OCT volume data."""
        exporter = DicomExporter()
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"dummy")

        study_no_vol = OCTStudy(source_path=src_file, source_format="fds", oct_volume=None)
        assert exporter.supports_oct(study_no_vol) is False

        oct_vol = OCTVolumeWithMetaData(volume=[np.zeros((10, 10), dtype=np.uint16)], patient_id="PAT1")
        study_with_vol = OCTStudy(source_path=src_file, source_format="fds", oct_volume=oct_vol)
        assert exporter.supports_oct(study_with_vol) is True

    def test_export_canonical_study(self, tmp_path):
        """Test exporting canonical study directly."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"dummy")

        oct_vol = OCTVolumeWithMetaData(volume=[np.zeros((10, 10), dtype=np.uint16)], patient_id="PAT1")
        study = OCTStudy(source_path=src_file, source_format="fds", oct_volume=oct_vol)

        exporter = DicomExporter()
        files = exporter.export(study, tmp_path / "output")

        assert len(files) == 1
        assert files[0].exists()
        assert files[0].suffix == ".dcm"


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


class TestZarrExporter:
    """Test Zarr v3 exporter implementation."""

    def test_numerical_roundtrip(self, tmp_path):
        """A. Basic numerical round-trip (shape, dtype, exact pixel equality)."""
        import zarr
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")

        source_data = np.arange(10 * 20 * 30, dtype=np.uint16).reshape((10, 20, 30))
        oct_vol = OCTVolumeWithMetaData(
            volume=[slice_arr for slice_arr in source_data],
            patient_id="PAT001",
        )
        study = OCTStudy(source_path=src_file, source_format="fds", oct_volume=oct_vol)

        exporter = ZarrExporter()
        out_files = exporter.export(study, tmp_path / "out")
        assert len(out_files) == 1
        zarr_path = out_files[0]
        assert zarr_path.exists()
        assert zarr_path.suffix == ".zarr"

        root = zarr.open_group(store=str(zarr_path), mode="r")
        assert "volume" in root
        volume_arr = root["volume"]

        assert volume_arr.shape == source_data.shape
        assert volume_arr.dtype == source_data.dtype
        restored = np.asarray(volume_arr)
        assert np.array_equal(source_data, restored)

    def test_non_default_dimensions_and_chunking(self, tmp_path):
        """B & C. Non-default dimensions (7 x 201 x 199) and chunking (1, height, width)."""
        import zarr
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")

        shape = (7, 201, 199)
        source_data = np.random.randint(0, 65535, size=shape, dtype=np.uint16)
        oct_vol = OCTVolumeWithMetaData(
            volume=[slice_arr for slice_arr in source_data],
            patient_id="PAT002",
        )
        study = OCTStudy(source_path=src_file, source_format="fds", oct_volume=oct_vol)

        exporter = ZarrExporter()
        out_files = exporter.export(study, tmp_path / "out")
        zarr_path = out_files[0]

        root = zarr.open_group(store=str(zarr_path), mode="r")
        volume_arr = root["volume"]
        assert volume_arr.shape == (7, 201, 199)
        assert volume_arr.chunks == (1, 201, 199)

    def test_fda_fds_spacing_mapping(self, tmp_path):
        """D. FDA/FDS spacing mapping: [width, slice_thickness, height] -> z, y, x scale."""
        import zarr
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")

        # Distinct spacing values: width=0.01 (x), slice_thickness=0.05 (z), height=0.002 (y)
        pixel_spacing = [0.01, 0.05, 0.002]
        oct_vol = OCTVolumeWithMetaData(
            volume=[np.ones((20, 30), dtype=np.uint16) for _ in range(5)],
            patient_id="PAT_FDS",
            pixel_spacing=pixel_spacing,
        )
        study = OCTStudy(source_path=src_file, source_format="fds", oct_volume=oct_vol)

        exporter = ZarrExporter()
        out_files = exporter.export(study, tmp_path / "out")
        root = zarr.open_group(store=str(out_files[0]), mode="r")

        attrs = dict(root.attrs)
        assert "scale" in attrs
        scale = attrs["scale"]
        assert scale["z"] == 0.05
        assert scale["y"] == 0.002
        assert scale["x"] == 0.01

        axes = attrs["axes"]
        assert axes == [
            {"name": "z", "type": "space", "unit": "millimeter"},
            {"name": "y", "type": "space", "unit": "millimeter"},
            {"name": "x", "type": "space", "unit": "millimeter"},
        ]

    def test_e2e_spacing_mapping(self, tmp_path):
        """E. E2E spacing mapping: [scalex, scaley, slice_thickness] -> z, y, x scale."""
        import zarr
        src_file = tmp_path / "test.e2e"
        src_file.write_bytes(b"fake e2e")

        # Distinct spacing values: scalex=0.01 (x), scaley=0.002 (y), slice_thickness=0.05 (z)
        pixel_spacing = [0.01, 0.002, 0.05]
        oct_vol = OCTVolumeWithMetaData(
            volume=[np.ones((20, 30), dtype=np.uint16) for _ in range(5)],
            patient_id="PAT_E2E",
            pixel_spacing=pixel_spacing,
        )
        study = OCTStudy(source_path=src_file, source_format="e2e", oct_volume=oct_vol)

        exporter = ZarrExporter()
        out_files = exporter.export(study, tmp_path / "out")
        root = zarr.open_group(store=str(out_files[0]), mode="r")

        attrs = dict(root.attrs)
        scale = attrs["scale"]
        assert scale["z"] == 0.05
        assert scale["y"] == 0.002
        assert scale["x"] == 0.01

    def test_poct_spacing_mapping(self, tmp_path):
        """F. POCT spacing mapping: [scale_x, scale_y] -> y, x scale without z."""
        import zarr
        src_file = tmp_path / "test.OCT"
        src_file.write_bytes(b"fake poct")

        pixel_spacing = [0.015, 0.025]
        oct_vol = OCTVolumeWithMetaData(
            volume=[np.ones((20, 30), dtype=np.uint16) for _ in range(5)],
            patient_id="PAT_POCT",
            pixel_spacing=pixel_spacing,
        )
        study = OCTStudy(source_path=src_file, source_format="poct", oct_volume=oct_vol)

        exporter = ZarrExporter()
        out_files = exporter.export(study, tmp_path / "out")
        root = zarr.open_group(store=str(out_files[0]), mode="r")

        attrs = dict(root.attrs)
        scale = attrs["scale"]
        assert "z" not in scale
        assert scale["y"] == 0.025
        assert scale["x"] == 0.015

        axes = attrs["axes"]
        assert axes == [
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space", "unit": "millimeter"},
            {"name": "x", "type": "space", "unit": "millimeter"},
        ]

    def test_missing_spacing(self, tmp_path):
        """G. Missing spacing: no scale attribute, no physical units."""
        import zarr
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")

        oct_vol = OCTVolumeWithMetaData(
            volume=[np.ones((20, 30), dtype=np.uint16) for _ in range(5)],
            patient_id="PAT_NOSPACING",
            pixel_spacing=None,
        )
        study = OCTStudy(source_path=src_file, source_format="fds", oct_volume=oct_vol)

        exporter = ZarrExporter()
        out_files = exporter.export(study, tmp_path / "out")
        root = zarr.open_group(store=str(out_files[0]), mode="r")

        attrs = dict(root.attrs)
        assert "scale" not in attrs
        axes = attrs["axes"]
        for axis in axes:
            assert "unit" not in axis

    def test_overwrite_behavior(self, tmp_path):
        """H. Overwrite behavior: overwrite=False raises ExportError, overwrite=True succeeds."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")

        oct_vol = OCTVolumeWithMetaData(
            volume=[np.zeros((10, 10), dtype=np.uint16)],
            patient_id="PAT_OVERWRITE",
        )
        study = OCTStudy(source_path=src_file, source_format="fds", oct_volume=oct_vol)
        exporter = ZarrExporter()

        out_dir = tmp_path / "out_overwrite"
        files1 = exporter.export(study, out_dir, options={"overwrite": True})
        assert len(files1) == 1

        with pytest.raises(ExportError, match="overwrite is disabled"):
            exporter.export(study, out_dir, options={"overwrite": False})

        files2 = exporter.export(study, out_dir, options={"overwrite": True})
        assert len(files2) == 1

    @pytest.mark.parametrize(
        "malicious_id",
        [
            "../../evil",
            "/absolute/path",
            r"..\..\evil",
        ],
    )
    def test_path_safety(self, tmp_path, malicious_id):
        """I. Path safety with adversarial patient IDs."""
        output_dir = tmp_path / "zarr_out"
        output_dir_resolved = output_dir.resolve()
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")

        oct_vol = OCTVolumeWithMetaData(
            volume=[np.zeros((10, 10), dtype=np.uint16)],
            patient_id=malicious_id,
        )
        study = OCTStudy(source_path=src_file, source_format="fds", oct_volume=oct_vol)

        exporter = ZarrExporter()
        files = exporter.export(study, output_dir)
        for f in files:
            assert f.resolve().is_relative_to(output_dir_resolved)
            assert f.exists()


