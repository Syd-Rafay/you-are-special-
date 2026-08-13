"""Tests for the OCT conversion service layer."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from oct_converter.image_types import FundusImageWithMetaData, OCTVolumeWithMetaData
from oct_converter_app.cli import main as cli_main
from oct_converter_service.batch import BatchConversionService
from oct_converter_service.config import ConversionConfig
from oct_converter_service.errors import (
    ConversionFailedError,
    ConversionServiceError,
    InputNotFoundError,
    InvalidRequestError,
    OutputError,
    OverwriteNotAllowedError,
    UnsupportedFormatError,
)
from oct_converter_service.requests import ConversionRequest
from oct_converter_service.results import ConversionResult
from oct_converter_service.service import ConversionService


@pytest.fixture
def fake_fds_file(tmp_path):
    """Create a dummy file path for testing."""
    p = tmp_path / "test_scan.fds"
    p.write_bytes(b"dummy fds data")
    return p


@pytest.fixture
def mock_pipeline_reader(monkeypatch):
    """Mock detector and reader factory to allow real exporter execution."""
    oct_vol = OCTVolumeWithMetaData(
        volume=[np.zeros((10, 10), dtype=np.uint16)],
        patient_id="PAT001",
        laterality="OD",
    )
    fundus_img = FundusImageWithMetaData(
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        patient_id="PAT001",
    )
    mock_reader = MagicMock()
    mock_reader.read_oct_volume.return_value = oct_vol
    mock_reader.read_fundus_image.return_value = fundus_img
    mock_reader.read_all_metadata.return_value = {"patient_id": "PAT001"}

    monkeypatch.setattr("oct_converter_app.pipeline.detect_format", lambda p: "fds")
    monkeypatch.setattr(
        "oct_converter_app.pipeline.ReaderFactory.create", lambda fmt, path: mock_reader
    )
    return mock_reader


class TestConversionConfig:
    """Tests for ConversionConfig."""

    def test_default_config(self):
        config = ConversionConfig()
        assert config.overwrite is False
        assert config.validate is True
        assert config.continue_on_warning is True
        assert config.compute_hash is False

    def test_custom_config(self):
        config = ConversionConfig(
            overwrite=True,
            validate=False,
            continue_on_warning=False,
            compute_hash=True,
        )
        assert config.overwrite is True
        assert config.validate is False
        assert config.continue_on_warning is False
        assert config.compute_hash is True

    def test_invalid_overwrite_type(self):
        with pytest.raises(ValueError, match="overwrite must be a boolean"):
            ConversionConfig(overwrite="true")

    def test_invalid_validate_type(self):
        with pytest.raises(ValueError, match="validate must be a boolean"):
            ConversionConfig(validate="false")


class TestConversionRequest:
    """Tests for ConversionRequest."""

    def test_minimal_request(self):
        request = ConversionRequest(
            input_path="/path/to/scan.fds",
            output_dir="/output/dir",
        )
        assert request.input_path == "/path/to/scan.fds"
        assert request.outputs == ["metadata"]
        assert request.overwrite is None

    def test_full_request(self):
        request = ConversionRequest(
            input_path="/path/to/scan.fds",
            output_dir="/output/dir",
            outputs=["dicom", "npy", "images", "metadata"],
            overwrite=True,
            validate=False,
            continue_on_warning=False,
            compute_hash=True,
            exporter_options={"dicom": {"key": "value"}},
        )
        assert request.outputs == ["dicom", "npy", "images", "metadata"]
        assert request.overwrite is True
        assert request.validate is False

    def test_request_schema_validation(self):
        with pytest.raises(ValueError, match="Unsupported output format"):
            ConversionRequest(
                input_path="/path/to/scan.fds",
                output_dir="/output/dir",
                outputs=["invalid_format"],
            )

    def test_request_empty_outputs(self):
        with pytest.raises(ValueError, match="At least one output format"):
            ConversionRequest(
                input_path="/path/to/scan.fds",
                output_dir="/output/dir",
                outputs=[],
            )


class TestConversionResult:
    """Tests for ConversionResult."""

    def test_result_creation(self):
        result = ConversionResult(
            success=True,
            input_path=Path("/input/test.fds"),
            output_dir=Path("/output"),
        )
        assert result.success is True
        assert result.input_path == Path("/input/test.fds")
        assert result.output_dir == Path("/output")
        assert result.generated_files == []

    def test_result_elapsed_time(self):
        started = datetime(2026, 1, 1, 12, 0, 0)
        completed = datetime(2026, 1, 1, 12, 0, 30)
        result = ConversionResult(
            success=True,
            input_path=Path("/input/test.fds"),
            output_dir=Path("/output"),
            started_at=started,
            completed_at=completed,
        )
        assert result.elapsed_time == 30.0

    def test_result_from_study(self):
        class MockStudy:
            source_path = Path("/input/test.fds")
            source_format = "fds"
            warnings = ["w1"]
            metadata = {"patient_id": "TEST123"}

        study = MockStudy()
        generated = [Path("/output/test.json")]
        result = ConversionResult.from_study(
            study=study,
            success=True,
            generated_files=generated,
            requested_outputs=["metadata"],
            output_dir=Path("/output"),
        )

        assert result.success is True
        assert result.output_dir == Path("/output")
        assert result.requested_outputs == ["metadata"]
        assert result.generated_files == generated


class TestConversionServiceE2E:
    """End-to-end service integration tests."""

    def test_a_successful_service_conversion(self, fake_fds_file, tmp_path, mock_pipeline_reader):
        """TEST A: Service conversion creates real files and returns valid ConversionResult."""
        output_dir = tmp_path / "out"
        service = ConversionService()
        request = ConversionRequest(
            input_path=fake_fds_file,
            output_dir=output_dir,
            outputs=["npy", "images", "metadata"],
        )

        result = service.convert(request)

        assert result.success is True
        assert len(result.generated_files) > 0
        assert result.output_dir == output_dir
        assert result.requested_outputs == ["npy", "images", "metadata"]
        assert result.skipped_outputs == []
        for f in result.generated_files:
            assert f.exists()

    def test_b_overwrite_through_service(self, fake_fds_file, tmp_path, mock_pipeline_reader):
        """TEST B: Service respects overwrite flag across requests."""
        output_dir = tmp_path / "out_overwrite"
        service = ConversionService()

        req1 = ConversionRequest(
            input_path=fake_fds_file,
            output_dir=output_dir,
            outputs=["metadata"],
            overwrite=True,
        )
        res1 = service.convert(req1)
        assert res1.success is True
        assert len(res1.generated_files) == 1
        assert res1.generated_files[0].exists()

        req2 = ConversionRequest(
            input_path=fake_fds_file,
            output_dir=output_dir,
            outputs=["metadata"],
            overwrite=False,
        )
        with pytest.raises(OverwriteNotAllowedError):
            service.convert(req2)

        req3 = ConversionRequest(
            input_path=fake_fds_file,
            output_dir=output_dir,
            outputs=["metadata"],
            overwrite=True,
        )
        res3 = service.convert(req3)
        assert res3.success is True

    def test_c_cli_uses_service(self, fake_fds_file, tmp_path, monkeypatch, mock_pipeline_reader):
        """TEST C: CLI uses ConversionService and handles execution."""
        output_dir = tmp_path / "cli_out"
        called = False

        original_convert = ConversionService.convert

        def spy_convert(self, request):
            nonlocal called
            called = True
            return original_convert(self, request)

        monkeypatch.setattr(ConversionService, "convert", spy_convert)

        exit_code = cli_main([str(fake_fds_file), str(output_dir), "--metadata", "--verbose"])
        assert exit_code == 0
        assert called is True

    def test_d_mixed_success_batch(self, fake_fds_file, tmp_path, mock_pipeline_reader):
        """TEST D: Batch processing handles mixed success and failure."""
        valid1 = fake_fds_file
        invalid2 = tmp_path / "nonexistent.fds"
        valid3 = tmp_path / "scan2.fds"
        valid3.write_bytes(b"data")

        batch = BatchConversionService()
        requests = [
            ConversionRequest(input_path=valid1, output_dir=tmp_path / "b1", outputs=["metadata"]),
            ConversionRequest(input_path=invalid2, output_dir=tmp_path / "b2", outputs=["metadata"]),
            ConversionRequest(input_path=valid3, output_dir=tmp_path / "b3", outputs=["metadata"]),
        ]

        # continue_on_error = True
        results = batch.convert_batch(requests, continue_on_error=True)
        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True

        # continue_on_error = False
        with pytest.raises(ConversionServiceError):
            batch.convert_batch(requests, continue_on_error=False)

    def test_e_skipped_output_failure_reporting(self, fake_fds_file, tmp_path, monkeypatch, capsys):
        """TEST E: Unproduced/skipped requested output results in failure with actionable error messages."""
        mock_reader = MagicMock()
        mock_reader.read_oct_volume.side_effect = Exception("No OCT data")
        mock_reader.read_fundus_image.side_effect = Exception("No Fundus data")
        mock_reader.read_all_metadata.return_value = {}

        monkeypatch.setattr("oct_converter_app.pipeline.detect_format", lambda p: "fds")
        monkeypatch.setattr(
            "oct_converter_app.pipeline.ReaderFactory.create", lambda fmt, path: mock_reader
        )

        output_dir = tmp_path / "out_skipped"
        service = ConversionService()
        req = ConversionRequest(
            input_path=fake_fds_file,
            output_dir=output_dir,
            outputs=["npy"],
            validate=False,
        )

        res = service.convert(req)

        assert res.success is False
        assert len(res.failures) > 0
        assert any("npy" in f or "not produced" in f or "No output files" in f for f in res.failures)

        # Verify CLI behavior
        capsys.readouterr()
        exit_code = cli_main([str(fake_fds_file), str(output_dir), "--npy", "--no-validate"])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "ERROR:" in captured.err
        assert "npy" in captured.err or "not produced" in captured.err


class TestErrorHierarchy:
    """Tests for exception hierarchy."""

    def test_all_errors_inherit_from_base(self):
        errors = [
            InvalidRequestError("test"),
            InputNotFoundError("test"),
            UnsupportedFormatError("test"),
            ConversionFailedError("test"),
            OutputError("test"),
            OverwriteNotAllowedError("test"),
        ]

        for error in errors:
            assert isinstance(error, ConversionServiceError)

    def test_catch_base_error(self):
        try:
            raise InvalidRequestError("test")
        except ConversionServiceError as e:
            assert isinstance(e, InvalidRequestError)

        try:
            raise InputNotFoundError("test")
        except ConversionServiceError as e:
            assert isinstance(e, InputNotFoundError)
