"""Tests for reader factory."""

import pytest
from pathlib import Path

from oct_converter_app.factory import ReaderFactory, ReaderCreationError


class TestReaderFactory:
    """Test ReaderFactory functionality."""

    def test_create_fds_reader(self, tmp_path):
        """Test creating FDS reader."""
        fds_file = tmp_path / "test.fds"
        fds_file.write_bytes(b"fake fds content")
        # Should create without error (may fail on actual read)
        reader = ReaderFactory.create("fds", fds_file)
        assert reader is not None

    def test_create_fda_reader(self, tmp_path):
        """Test creating FDA reader."""
        fda_file = tmp_path / "test.fda"
        fda_file.write_bytes(b"fake fda content")
        reader = ReaderFactory.create("fda", fda_file)
        assert reader is not None

    def test_create_e2e_reader(self, tmp_path):
        """Test creating E2E reader."""
        e2e_file = tmp_path / "test.e2e"
        e2e_file.write_bytes(b"fake e2e content")
        reader = ReaderFactory.create("e2e", e2e_file)
        assert reader is not None

    def test_create_img_reader(self, tmp_path):
        """Test creating IMG reader."""
        img_file = tmp_path / "test.img"
        img_file.write_bytes(b"fake img content")
        reader = ReaderFactory.create("img", img_file)
        assert reader is not None

    def test_create_unknown_format(self, tmp_path):
        """Test creating reader for unknown format raises error."""
        fake_file = tmp_path / "test.xyz"
        fake_file.write_bytes(b"fake content")
        with pytest.raises(ReaderCreationError):
            ReaderFactory.create("xyz", fake_file)

    def test_supported_formats(self):
        """Test supported formats list."""
        formats = ReaderFactory.supported_formats()
        assert isinstance(formats, list)
        assert "fds" in formats
        assert "fda" in formats
        assert "e2e" in formats
        assert "img" in formats
        assert "boct" in formats
        assert "poct" in formats
        assert "dcm" in formats

    def test_get_reader_class(self):
        """Test getting reader class without instantiation."""
        from oct_converter.readers import FDS
        cls = ReaderFactory.get_reader_class("fds")
        assert cls == FDS

    def test_register_custom_reader(self):
        """Test registering custom reader."""
        class FakeReader:
            def __init__(self, path):
                self.path = path
        
        ReaderFactory.register("fake", FakeReader)
        assert "fake" in ReaderFactory.supported_formats()
        
        # Clean up
        ReaderFactory.unregister("fake")

    def test_unregister_reader(self):
        """Test unregistering a reader."""
        # Register first
        class FakeReader:
            pass
        
        ReaderFactory.register("temp_fake", FakeReader)
        assert ReaderFactory.unregister("temp_fake") is True
        assert "temp_fake" not in ReaderFactory.supported_formats()
