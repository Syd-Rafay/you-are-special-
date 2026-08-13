"""Tests for the OCT conversion service layer."""

import tempfile
from pathlib import Path
from datetime import datetime

import pytest

from oct_converter_service.service import ConversionService
from oct_converter_service.requests import ConversionRequest
from oct_converter_service.results import ConversionResult
from oct_converter_service.config import ConversionConfig
from oct_converter_service.batch import BatchConversionService
from oct_converter_service.errors import (
    ConversionServiceError,
    InvalidRequestError,
    InputNotFoundError,
    UnsupportedFormatError,
    ConversionFailedError,
    OutputError,
    OverwriteNotAllowedError,
)


class TestConversionConfig:
    """Tests for ConversionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ConversionConfig()
        assert config.overwrite is False
        assert config.validate is True
        assert config.continue_on_warning is True
        assert config.compute_hash is False

    def test_custom_config(self):
        """Test custom configuration values."""
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
        """Test that invalid overwrite type raises error."""
        with pytest.raises(ValueError, match="overwrite must be a boolean"):
            ConversionConfig(overwrite="true")

    def test_invalid_validate_type(self):
        """Test that invalid validate type raises error."""
        with pytest.raises(ValueError, match="validate must be a boolean"):
            ConversionConfig(validate="false")


class TestConversionRequest:
    """Tests for ConversionRequest."""

    def test_minimal_request(self):
        """Test minimal valid request."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".fds") as tmp:
            input_path = Path(tmp.name)

        try:
            request = ConversionRequest(
                input_path=input_path,
                output_dir=tempfile.gettempdir(),
            )
            assert request.input_path == input_path
            assert request.outputs == ["metadata"]  # default
            assert request.overwrite is False
        finally:
            input_path.unlink()

    def test_full_request(self):
        """Test request with all parameters."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".fds") as tmp:
            input_path = Path(tmp.name)

        output_dir = Path(tempfile.mkdtemp())
        try:
            request = ConversionRequest(
                input_path=input_path,
                output_dir=output_dir,
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
            assert request.exporter_options == {"dicom": {"key": "value"}}
        finally:
            input_path.unlink()

    def test_request_input_not_exists(self):
        """Test that non-existent input raises error."""
        with pytest.raises(ValueError, match="Input path does not exist"):
            ConversionRequest(
                input_path="/nonexistent/file.fds",
                output_dir=tempfile.gettempdir(),
            )

    def test_request_input_is_directory(self):
        """Test that directory as input raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Input path must be a file"):
                ConversionRequest(
                    input_path=tmpdir,
                    output_dir=tempfile.gettempdir(),
                )

    def test_request_no_outputs(self):
        """Test that empty outputs list raises error."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".fds") as tmp:
            input_path = Path(tmp.name)

        try:
            with pytest.raises(ValueError, match="At least one output format"):
                ConversionRequest(
                    input_path=input_path,
                    output_dir=tempfile.gettempdir(),
                    outputs=[],
                )
        finally:
            input_path.unlink()

    def test_request_invalid_output_format(self):
        """Test that invalid output format raises error."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".fds") as tmp:
            input_path = Path(tmp.name)

        try:
            with pytest.raises(ValueError, match="Unsupported output format"):
                ConversionRequest(
                    input_path=input_path,
                    output_dir=tempfile.gettempdir(),
                    outputs=["invalid_format"],
                )
        finally:
            input_path.unlink()

    def test_request_valid_outputs(self):
        """Test all valid output formats."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".fds") as tmp:
            input_path = Path(tmp.name)

        try:
            # Each valid output should work
            for output in ["dicom", "npy", "images", "metadata"]:
                request = ConversionRequest(
                    input_path=input_path,
                    output_dir=tempfile.gettempdir(),
                    outputs=[output],
                )
                assert request.outputs == [output]
        finally:
            input_path.unlink()


class TestConversionResult:
    """Tests for ConversionResult."""

    def test_result_creation(self):
        """Test basic result creation."""
        result = ConversionResult(
            success=True,
            input_path=Path("/input/test.fds"),
            output_dir=Path("/output"),
        )
        assert result.success is True
        assert result.input_path == Path("/input/test.fds")
        assert result.output_dir == Path("/output")
        assert result.generated_files == []
        assert result.failures == []

    def test_result_elapsed_time(self):
        """Test elapsed time calculation."""
        started = datetime(2023, 1, 1, 12, 0, 0)
        completed = datetime(2023, 1, 1, 12, 0, 30)

        result = ConversionResult(
            success=True,
            input_path=Path("/input/test.fds"),
            output_dir=Path("/output"),
            started_at=started,
            completed_at=completed,
        )
        assert result.elapsed_time == 30.0

    def test_result_elapsed_time_none(self):
        """Test elapsed time when not set."""
        result = ConversionResult(
            success=True,
            input_path=Path("/input/test.fds"),
            output_dir=Path("/output"),
        )
        assert result.elapsed_time is None

    def test_result_from_study(self):
        """Test creating result from study object."""
        # Create a mock study
        class MockStudy:
            source_path = Path("/input/test.fds")
            source_format = "fds"
            warnings = ["warning1", "warning2"]
            metadata = {"patient_id": "TEST123"}

        study = MockStudy()
        generated = [Path("/output/test.dcm")]

        result = ConversionResult.from_study(
            study=study,
            success=True,
            generated_files=generated,
        )

        assert result.success is True
        assert result.input_path == Path("/input/test.fds")
        assert result.detected_format == "fds"
        assert result.generated_files == [Path("/output/test.dcm")]
        assert result.warnings == ["warning1", "warning2"]
        assert result.metadata == {"patient_id": "TEST123"}


class TestConversionService:
    """Tests for ConversionService."""

    def test_service_initialization(self):
        """Test service initialization."""
        service = ConversionService()
        assert service.config is not None
        assert service._pipeline is not None

    def test_service_with_custom_config(self):
        """Test service with custom config."""
        config = ConversionConfig(overwrite=True, validate=False)
        service = ConversionService(config)
        assert service.config.overwrite is True
        assert service.config.validate is False

    def test_service_input_not_found(self):
        """Test service raises InputNotFoundError for missing file."""
        service = ConversionService()
        request = ConversionRequest(
            input_path="/nonexistent/file.fds",
            output_dir=tempfile.gettempdir(),
            outputs=["metadata"],
        )

        with pytest.raises(InputNotFoundError):
            service.convert(request)

    def test_service_invalid_request(self):
        """Test service raises InvalidRequestError for invalid request."""
        service = ConversionService()

        # Create a request with invalid output format
        with tempfile.NamedTemporaryFile(delete=False, suffix=".fds") as tmp:
            input_path = Path(tmp.name)

        try:
            # Bypass validation by creating request differently
            request = object.__new__(ConversionRequest)
            request.input_path = input_path
            request.output_dir = tempfile.gettempdir()
            request.outputs = ["invalid"]
            request.overwrite = False
            request.validate = True
            request.continue_on_warning = True
            request.compute_hash = False
            request.exporter_options = {}

            with pytest.raises(InvalidRequestError):
                service.convert(request)
        finally:
            input_path.unlink()


class TestBatchConversionService:
    """Tests for BatchConversionService."""

    def test_batch_initialization(self):
        """Test batch service initialization."""
        batch = BatchConversionService()
        assert batch.service is not None

    def test_batch_with_custom_service(self):
        """Test batch service with custom service."""
        service = ConversionService()
        batch = BatchConversionService(service)
        assert batch.service == service

    def test_batch_empty_list(self):
        """Test batch processing with empty list."""
        batch = BatchConversionService()
        results = batch.convert_batch([])
        assert results == []

    def test_batch_continue_on_error(self):
        """Test batch processing continues on error."""
        batch = BatchConversionService()

        # Create requests - some will fail (non-existent files)
        requests = [
            ConversionRequest(
                input_path="/nonexistent1.fds",
                output_dir=tempfile.gettempdir(),
                outputs=["metadata"],
            ),
            ConversionRequest(
                input_path="/nonexistent2.fds",
                output_dir=tempfile.gettempdir(),
                outputs=["metadata"],
            ),
        ]

        # With continue_on_error=True, should get results for all
        results = batch.convert_batch(requests, continue_on_error=True)
        assert len(results) == 2
        # Both should be failures
        assert all(not r.success for r in results)

    def test_batch_stop_on_error(self):
        """Test batch processing stops on error."""
        batch = BatchConversionService()

        requests = [
            ConversionRequest(
                input_path="/nonexistent.fds",
                output_dir=tempfile.gettempdir(),
                outputs=["metadata"],
            ),
        ]

        # With continue_on_error=False, should raise on first error
        with pytest.raises(ConversionServiceError):
            batch.convert_batch(requests, continue_on_error=False)


class TestErrorHierarchy:
    """Tests for exception hierarchy."""

    def test_all_errors_inherit_from_base(self):
        """Test all errors inherit from ConversionServiceError."""
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
        """Test catching base error catches all."""
        try:
            raise InvalidRequestError("test")
        except ConversionServiceError as e:
            assert isinstance(e, InvalidRequestError)

        try:
            raise InputNotFoundError("test")
        except ConversionServiceError as e:
            assert isinstance(e, InputNotFoundError)
