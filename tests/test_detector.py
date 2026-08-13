"""Tests for format detection."""

import tempfile
from pathlib import Path

import pytest

from oct_converter_app.detector import FormatDetector, UnsupportedFormatError, detect_format


class TestDetectFormat:
    """Test format detection functionality."""

    def test_detect_fds(self, tmp_path):
        """Test FDS format detection."""
        fds_file = tmp_path / "test.fds"
        fds_file.write_bytes(b"fake fds content")
        assert detect_format(fds_file) == "fds"

    def test_detect_fda(self, tmp_path):
        """Test FDA format detection."""
        fda_file = tmp_path / "test.fda"
        fda_file.write_bytes(b"fake fda content")
        assert detect_format(fda_file) == "fda"

    def test_detect_e2e(self, tmp_path):
        """Test E2E format detection."""
        e2e_file = tmp_path / "test.e2e"
        e2e_file.write_bytes(b"fake e2e content")
        assert detect_format(e2e_file) == "e2e"

    def test_detect_img(self, tmp_path):
        """Test IMG format detection."""
        img_file = tmp_path / "test.img"
        img_file.write_bytes(b"fake img content")
        assert detect_format(img_file) == "img"

    def test_detect_dcm(self, tmp_path):
        """Test DICOM format detection."""
        dcm_file = tmp_path / "test.dcm"
        dcm_file.write_bytes(b"fake dcm content")
        assert detect_format(dcm_file) == "dcm"

    def test_unsupported_extension(self, tmp_path):
        """Test unsupported extension raises error."""
        xyz_file = tmp_path / "test.xyz"
        xyz_file.write_bytes(b"fake content")
        with pytest.raises(UnsupportedFormatError):
            detect_format(xyz_file)

    def test_file_not_found(self, tmp_path):
        """Test nonexistent file raises error."""
        nonexistent = tmp_path / "nonexistent.fds"
        with pytest.raises(FileNotFoundError):
            detect_format(nonexistent)

    def test_path_is_directory(self, tmp_path):
        """Test directory path raises error."""
        with pytest.raises(ValueError):
            detect_format(tmp_path)


class TestFormatDetectorClass:
    """Test FormatDetector class wrapper."""

    def test_detect_method(self, tmp_path):
        """Test detect method."""
        fds_file = tmp_path / "test.fds"
        fds_file.write_bytes(b"fake fds content")
        assert FormatDetector.detect(fds_file) == "fds"

    def test_supported_formats(self):
        """Test supported formats list."""
        formats = FormatDetector.supported_formats()
        assert isinstance(formats, set)
        assert "fds" in formats
        assert "fda" in formats
        assert "e2e" in formats
        assert "img" in formats
