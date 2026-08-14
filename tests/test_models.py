"""Tests for data models."""

from pathlib import Path
from datetime import datetime

import numpy as np
import pytest

from oct_converter_app.models import (
    OCTStudy,
    Provenance,
    Capabilities,
    StudyCapabilities,
    VendorDevice,
    DerivedProduct,
)
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


class TestVendorDevice:
    """Test VendorDevice model."""

    def test_all_fields(self):
        """Test VendorDevice with all fields populated."""
        vd = VendorDevice(
            manufacturer="Topcon",
            model="TRITON",
            software_version="1.0.0"
        )
        assert vd.manufacturer == "Topcon"
        assert vd.model == "TRITON"
        assert vd.software_version == "1.0.0"

    def test_missing_fields(self):
        """Test VendorDevice with all fields missing."""
        vd = VendorDevice()
        assert vd.manufacturer is None
        assert vd.model is None
        assert vd.software_version is None

    def test_partial_fields(self):
        """Test VendorDevice with partial fields."""
        vd = VendorDevice(manufacturer="Heidelberg")
        assert vd.manufacturer == "Heidelberg"
        assert vd.model is None
        assert vd.software_version is None

    def test_frozen_dataclass(self):
        """Test that VendorDevice is immutable."""
        vd = VendorDevice(manufacturer="Test")
        with pytest.raises((AttributeError, TypeError)):
            vd.manufacturer = "Changed"

    def test_repr_safety(self):
        """Test that repr does not expose sensitive info."""
        vd = VendorDevice(manufacturer="TestMfg", model="TestModel")
        repr_str = repr(vd)
        # Should be a simple representation
        assert "VendorDevice" in repr_str


class TestDerivedProduct:
    """Test DerivedProduct extension point."""

    def test_minimal_product(self):
        """Test minimal derived product."""
        dp = DerivedProduct(product_type="test_type")
        assert dp.product_type == "test_type"
        assert dp.data is None
        assert dp.metadata == {}
        assert dp.creation_method is None

    def test_full_product(self):
        """Test derived product with all fields."""
        dp = DerivedProduct(
            product_type="octa_enface",
            data=np.zeros((100, 100)),
            metadata={"algorithm": "test"},
            creation_method="test_algo_v1"
        )
        assert dp.product_type == "octa_enface"
        assert dp.data is not None
        assert dp.metadata == {"algorithm": "test"}
        assert dp.creation_method == "test_algo_v1"

    def test_empty_default_metadata(self):
        """Test that metadata defaults to empty dict, not shared."""
        dp1 = DerivedProduct(product_type="type1")
        dp2 = DerivedProduct(product_type="type2")
        dp1.metadata["key"] = "value"
        assert dp2.metadata == {}


class TestStudyCapabilities:
    """Test StudyCapabilities model."""

    def test_empty_capabilities(self):
        """Test default capabilities."""
        caps = StudyCapabilities()
        assert caps.has_oct_volume is False
        assert caps.has_fundus is False
        assert caps.num_bscans == 0
        assert caps.has_octa is False
        assert caps.has_layer_segmentation is False

    def test_from_study_with_oct(self, tmp_path):
        """Test deriving capabilities from study with OCT."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        oct_vol = OCTVolumeWithMetaData(
            volume=[np.ones((100, 100), dtype=np.uint16)],
            pixel_spacing=[0.1, 0.1, 0.1]
        )
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds",
            oct_volume=oct_vol,
            fundus=None
        )
        
        caps = StudyCapabilities.from_study(study)
        assert caps.has_oct_volume is True
        assert caps.num_bscans == 1
        assert caps.has_pixel_spacing is True

    def test_from_study_with_fundus(self, tmp_path):
        """Test deriving capabilities from study with fundus."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        fundus_img = FundusImageWithMetaData(
            image=np.ones((100, 100, 3), dtype=np.uint8)
        )
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds",
            oct_volume=None,
            fundus=fundus_img
        )
        
        caps = StudyCapabilities.from_study(study)
        assert caps.has_fundus is True
        assert caps.fundus_shape == (100, 100, 3)
        assert caps.has_oct_volume is False

    def test_from_study_with_both(self, tmp_path):
        """Test deriving capabilities from study with both modalities."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        oct_vol = OCTVolumeWithMetaData(
            volume=[np.ones((50, 50), dtype=np.uint16)] * 10
        )
        fundus_img = FundusImageWithMetaData(
            image=np.ones((200, 300, 3), dtype=np.uint8)
        )
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds",
            oct_volume=oct_vol,
            fundus=fundus_img
        )
        
        caps = StudyCapabilities.from_study(study)
        assert caps.has_oct_volume is True
        assert caps.num_bscans == 10
        assert caps.has_fundus is True
        assert caps.fundus_shape == (200, 300, 3)

    def test_future_flags_default_false(self):
        """Test that future-facing flags default to False."""
        caps = StudyCapabilities()
        assert caps.has_octa is False
        assert caps.has_layer_segmentation is False


class TestCapabilitiesBackwardCompatibility:
    """Test backward compatibility of original Capabilities class."""

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
        
        oct_vol = OCTVolumeWithMetaData(
            volume=[np.ones((100, 100), dtype=np.uint16)],
            pixel_spacing=[0.1, 0.1, 0.1]
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
        assert study.vendor_device is None
        assert study.derived_products == []

    def test_patient_id_property(self, tmp_path):
        """Test patient_id extraction."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        oct_vol = OCTVolumeWithMetaData(
            volume=[np.ones((100, 100), dtype=np.uint16)],
            patient_id="TEST123"
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
            volume=[np.ones((100, 100), dtype=np.uint16)]
        )
        study.oct_volume = oct_vol
        assert study.has_errors() is False

    def test_capabilities_property_is_computed(self, tmp_path):
        """Test that capabilities is a computed property, not cached."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        oct_vol = OCTVolumeWithMetaData(
            volume=[np.ones((100, 100), dtype=np.uint16)]
        )
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds",
            oct_volume=oct_vol
        )
        
        # Initial state: has OCT
        assert study.capabilities.has_oct_volume is True
        
        # Remove OCT volume
        study.oct_volume = None
        
        # Capabilities should reflect new state (not stale)
        assert study.capabilities.has_oct_volume is False

    def test_derived_products_empty_by_default(self, tmp_path):
        """Test that derived_products is empty by default."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds"
        )
        
        assert study.derived_products == []

    def test_add_derived_product(self, tmp_path):
        """Test adding derived products."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds"
        )
        
        dp = DerivedProduct(product_type="test_product")
        study.add_derived_product(dp)
        
        assert len(study.derived_products) == 1
        assert study.derived_products[0].product_type == "test_product"

    def test_add_derived_product_type_check(self, tmp_path):
        """Test that add_derived_product validates type."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds"
        )
        
        with pytest.raises(TypeError):
            study.add_derived_product("not a DerivedProduct")

    def test_derived_products_independent_instances(self, tmp_path):
        """Test that different studies have independent derived product lists."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        study1 = OCTStudy(source_path=src_file, source_format="fds")
        study2 = OCTStudy(source_path=src_file, source_format="fds")
        
        dp = DerivedProduct(product_type="test")
        study1.add_derived_product(dp)
        
        assert len(study1.derived_products) == 1
        assert len(study2.derived_products) == 0

    def test_repr_phi_safe(self, tmp_path):
        """Test that repr does not expose PHI or raw metadata."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        secret_value = "PATIENT_SECRET_12345"
        
        oct_vol = OCTVolumeWithMetaData(
            volume=[np.ones((100, 100), dtype=np.uint16)],
            patient_id=secret_value
        )
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds",
            oct_volume=oct_vol,
            metadata={"patient_name": secret_value},
            raw_metadata={"secret_field": secret_value}
        )
        
        repr_str = repr(study)
        
        # Secret value should NOT appear in repr
        assert secret_value not in repr_str
        
        # But useful debug info should be present
        assert "OCTStudy" in repr_str
        assert "fds" in repr_str
        assert "caps" in repr_str.lower()

    def test_repr_with_oct_and_fundus(self, tmp_path):
        """Test repr shows dimensions for OCT and fundus."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        oct_vol = OCTVolumeWithMetaData(
            volume=[np.ones((50, 60), dtype=np.uint16)] * 10
        )
        fundus_img = FundusImageWithMetaData(
            image=np.ones((200, 300, 3), dtype=np.uint8)
        )
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds",
            oct_volume=oct_vol,
            fundus=fundus_img
        )
        
        repr_str = repr(study)
        
        assert "10 B-scans" in repr_str or "B-scans" in repr_str
        assert "(200, 300, 3)" in repr_str or "fundus" in repr_str.lower()


class TestCapabilityStaleness:
    """Test that capabilities cannot become stale."""

    def test_oct_removal_updates_capabilities(self, tmp_path):
        """Test that removing OCT updates capabilities."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        oct_vol = OCTVolumeWithMetaData(
            volume=[np.ones((100, 100), dtype=np.uint16)]
        )
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds",
            oct_volume=oct_vol
        )
        
        # Verify initial state
        assert study.capabilities.has_oct_volume is True
        
        # Mutate study
        study.oct_volume = None
        
        # Verify capabilities updated
        assert study.capabilities.has_oct_volume is False
        assert study.capabilities.num_bscans == 0

    def test_oct_replacement_updates_capabilities(self, tmp_path):
        """Test that replacing OCT with different size updates capabilities."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        oct_vol_small = OCTVolumeWithMetaData(
            volume=[np.ones((50, 50), dtype=np.uint16)] * 5
        )
        oct_vol_large = OCTVolumeWithMetaData(
            volume=[np.ones((100, 100), dtype=np.uint16)] * 20
        )
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds",
            oct_volume=oct_vol_small
        )
        
        assert study.capabilities.num_bscans == 5
        
        study.oct_volume = oct_vol_large
        
        assert study.capabilities.num_bscans == 20

    def test_fundus_removal_updates_capabilities(self, tmp_path):
        """Test that removing fundus updates capabilities."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds")
        
        fundus_img = FundusImageWithMetaData(
            image=np.ones((100, 100, 3), dtype=np.uint8)
        )
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds",
            fundus=fundus_img
        )
        
        assert study.capabilities.has_fundus is True
        
        study.fundus = None
        
        assert study.capabilities.has_fundus is False


class TestIntegration:
    """Integration tests using pipeline-like construction."""

    def test_study_construction_pattern(self, tmp_path):
        """Test study construction similar to pipeline usage."""
        src_file = tmp_path / "test.fds"
        src_file.write_bytes(b"fake fds content")
        
        # Simulate what pipeline does
        oct_volume = OCTVolumeWithMetaData(
            volume=[np.ones((64, 64), dtype=np.uint16)] * 8,
            patient_id="INT_TEST_001",
            laterality="R"
        )
        
        fundus = FundusImageWithMetaData(
            image=np.ones((512, 512, 3), dtype=np.uint8),
            laterality="R"
        )
        
        provenance = Provenance.create(
            source_path=src_file,
            source_format="fds",
            compute_hash=False
        )
        
        vendor_device = VendorDevice(
            manufacturer="TestVendor",
            model="TestModel"
        )
        
        study = OCTStudy(
            source_path=src_file,
            source_format="fds",
            oct_volume=oct_volume,
            fundus=fundus,
            provenance=provenance,
            vendor_device=vendor_device
        )
        
        # Verify all components are accessible
        assert study.patient_id == "INT_TEST_001"
        assert study.laterality == "R"
        assert study.capabilities.has_oct_volume is True
        assert study.capabilities.has_fundus is True
        assert study.capabilities.num_bscans == 8
        assert study.vendor_device.manufacturer == "TestVendor"
        assert study.provenance.source_format == "fds"
        
        # Add a derived product
        dp = DerivedProduct(
            product_type="simulated_thickness_map",
            data=np.zeros((50, 50)),
            creation_method="simulation_v1"
        )
        study.add_derived_product(dp)
        
        assert len(study.derived_products) == 1
        assert study.derived_products[0].product_type == "simulated_thickness_map"
