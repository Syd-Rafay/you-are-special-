"""Tests for Ophthalmic Tomography DICOM export (SOP Class 1.2.840.10008.5.1.4.1.1.77.1.5.4)."""

import numpy as np
import pydicom
import pytest
from pathlib import Path

from oct_converter.image_types import OCTVolumeWithMetaData
from oct_converter_app.exporters.base import ExportError
from oct_converter_app.exporters.dicom import DicomExporter
from oct_converter_app.models import OCTStudy, VendorDevice
from oct_converter_service.config import ConversionConfig
from oct_converter_service.requests import ConversionRequest
from oct_converter_service.service import ConversionService


OPHTHALMIC_TOMOGRAPHY_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.77.1.5.4"


@pytest.fixture
def sample_study_factory(tmp_path):
    """Factory fixture for generating test OCTStudy objects."""

    def _create_study(
        volume=None,
        patient_id="PAT_TEST_123",
        source_format="fds",
        pixel_spacing=None,
        laterality="OD",
    ):
        if volume is None:
            volume = np.ones((5, 20, 30), dtype=np.uint16) * 1000

        src_file = tmp_path / f"test_input.{source_format}"
        src_file.write_bytes(b"dummy source data")

        oct_vol = OCTVolumeWithMetaData(
            volume=volume,
            patient_id=patient_id,
            laterality=laterality,
            pixel_spacing=pixel_spacing,
        )

        vendor_dev = VendorDevice(
            manufacturer="Topcon",
            model="3D OCT-2000",
            software_version="8.11",
        )

        return OCTStudy(
            source_path=src_file,
            source_format=source_format,
            oct_volume=oct_vol,
            vendor_device=vendor_dev,
        )

    return _create_study


def test_1_sop_class(tmp_path, sample_study_factory):
    """Test 1: Verify SOPClassUID and MediaStorageSOPClassUID are 1.2.840.10008.5.1.4.1.1.77.1.5.4."""
    study = sample_study_factory()
    exporter = DicomExporter()
    files = exporter.export(study, tmp_path)

    assert len(files) == 1
    ds = pydicom.dcmread(files[0])

    assert ds.SOPClassUID == OPHTHALMIC_TOMOGRAPHY_SOP_CLASS_UID
    assert ds.file_meta.MediaStorageSOPClassUID == OPHTHALMIC_TOMOGRAPHY_SOP_CLASS_UID
    assert ds.Modality == "OPT"


def test_2_multiframe_dimensions(tmp_path, sample_study_factory):
    """Test 2: Verify multi-frame dimensions for 5 x 20 x 30 uint16 volume."""
    vol = np.zeros((5, 20, 30), dtype=np.uint16)
    study = sample_study_factory(volume=vol)

    exporter = DicomExporter()
    files = exporter.export(study, tmp_path)

    ds = pydicom.dcmread(files[0])
    assert ds.NumberOfFrames == 5
    assert ds.Rows == 20
    assert ds.Columns == 30


def test_3_pixel_fidelity(tmp_path, sample_study_factory):
    """Test 3: Verify write and read-back pixel fidelity."""
    # Create volume with non-trivial uint16 values
    source_vol = np.arange(5 * 20 * 30, dtype=np.uint16).reshape((5, 20, 30))
    study = sample_study_factory(volume=source_vol)

    exporter = DicomExporter()
    files = exporter.export(study, tmp_path)

    ds = pydicom.dcmread(files[0])
    read_vol = ds.pixel_array

    assert np.array_equal(source_vol, read_vol)


def test_4_image_pixel_attributes(tmp_path, sample_study_factory):
    """Test 4: Verify DICOM Image Pixel Module attributes."""
    study = sample_study_factory()
    exporter = DicomExporter()
    files = exporter.export(study, tmp_path)

    ds = pydicom.dcmread(files[0])
    assert ds.SamplesPerPixel == 1
    assert ds.PhotometricInterpretation == "MONOCHROME2"
    assert ds.BitsAllocated == 16
    assert ds.BitsStored == 16
    assert ds.HighBit == 15
    assert ds.PixelRepresentation == 0


def test_5_multiframe_dimension_module(tmp_path, sample_study_factory):
    """Test 5: Verify Multi-frame Dimension Module sequences and pointers."""
    study = sample_study_factory()
    exporter = DicomExporter()
    files = exporter.export(study, tmp_path)

    ds = pydicom.dcmread(files[0])

    assert hasattr(ds, "DimensionOrganizationSequence")
    assert len(ds.DimensionOrganizationSequence) == 1
    dim_org_uid = ds.DimensionOrganizationSequence[0].DimensionOrganizationUID

    assert hasattr(ds, "DimensionIndexSequence")
    assert len(ds.DimensionIndexSequence) == 1
    dim_index = ds.DimensionIndexSequence[0]

    assert dim_index.DimensionOrganizationUID == dim_org_uid
    # In-Stack Position Number: (0020, 9057)
    assert dim_index.DimensionIndexPointer == 0x00209057
    # Frame Content Sequence: (0020, 9111)
    assert dim_index.FunctionalGroupPointer == 0x00209111

    # Verify per-frame sequence contains matching DimensionIndexValues
    assert hasattr(ds, "PerFrameFunctionalGroupsSequence")
    assert len(ds.PerFrameFunctionalGroupsSequence) == 5
    for idx, frame_fg in enumerate(ds.PerFrameFunctionalGroupsSequence):
        fc = frame_fg.FrameContentSequence[0]
        assert fc.InStackPositionNumber == idx + 1
        dim_val = fc.DimensionIndexValues[0] if hasattr(fc.DimensionIndexValues, "__getitem__") and not isinstance(fc.DimensionIndexValues, (int, str)) else fc.DimensionIndexValues
        assert int(dim_val) == idx + 1


def test_6_fda_fds_spacing(tmp_path, sample_study_factory):
    """Test 6: Verify FDA/FDS vendor spacing mapping [x, z, y]."""
    # x (col) = 0.01, z (slice) = 0.05, y (row) = 0.002
    pixel_spacing = [0.01, 0.05, 0.002]
    study = sample_study_factory(source_format="fds", pixel_spacing=pixel_spacing)

    exporter = DicomExporter()
    files = exporter.export(study, tmp_path)

    ds = pydicom.dcmread(files[0])
    shared_fg = ds.SharedFunctionalGroupsSequence[0]
    pm = shared_fg.PixelMeasuresSequence[0]

    # PixelSpacing = [row_spacing (y), col_spacing (x)]
    assert float(pm.PixelSpacing[0]) == pytest.approx(0.002)
    assert float(pm.PixelSpacing[1]) == pytest.approx(0.01)
    assert float(pm.SliceThickness) == pytest.approx(0.05)


def test_7_e2e_spacing(tmp_path, sample_study_factory):
    """Test 7: Verify E2E vendor spacing mapping [x, y, z]."""
    # x (col) = 0.01, y (row) = 0.002, z (slice) = 0.05
    pixel_spacing = [0.01, 0.002, 0.05]
    study = sample_study_factory(source_format="e2e", pixel_spacing=pixel_spacing)

    exporter = DicomExporter()
    files = exporter.export(study, tmp_path)

    ds = pydicom.dcmread(files[0])
    shared_fg = ds.SharedFunctionalGroupsSequence[0]
    pm = shared_fg.PixelMeasuresSequence[0]

    # PixelSpacing = [row_spacing (y), col_spacing (x)]
    assert float(pm.PixelSpacing[0]) == pytest.approx(0.002)
    assert float(pm.PixelSpacing[1]) == pytest.approx(0.01)
    assert float(pm.SliceThickness) == pytest.approx(0.05)


def test_8_missing_spacing(tmp_path, sample_study_factory):
    """Test 8: Verify missing spacing does not fabricate calibration."""
    study = sample_study_factory(pixel_spacing=None)

    exporter = DicomExporter()
    files = exporter.export(study, tmp_path)

    ds = pydicom.dcmread(files[0])
    shared_fg = ds.SharedFunctionalGroupsSequence[0]

    # PixelMeasuresSequence should be absent when spacing is None
    assert not hasattr(shared_fg, "PixelMeasuresSequence") or len(shared_fg.PixelMeasuresSequence) == 0


def test_9_laterality(tmp_path, sample_study_factory):
    """Test 9: Verify reliable canonical laterality mapping to DICOM fields."""
    study = sample_study_factory(laterality="OD")

    exporter = DicomExporter()
    files = exporter.export(study, tmp_path)

    ds = pydicom.dcmread(files[0])

    assert ds.Laterality == "R"
    assert ds.ImageLaterality == "R"

    shared_fg = ds.SharedFunctionalGroupsSequence[0]
    assert shared_fg.FrameAnatomySequence[0].FrameLaterality == "R"


def test_10_float_rejection(tmp_path, sample_study_factory):
    """Test 10: Verify floating-point volume input fails clearly with ExportError."""
    float_vol = np.zeros((5, 20, 30), dtype=np.float32)
    study = sample_study_factory(volume=float_vol)

    exporter = DicomExporter()

    with pytest.raises(ExportError, match="Float pixel data is not supported"):
        exporter.export(study, tmp_path)


def test_11_service_integration(tmp_path, sample_study_factory, monkeypatch):
    """Test 11: Verify ConversionRequest(outputs=["dicom"]) -> ConversionService integration."""
    study = sample_study_factory()

    mock_reader = pytest.importorskip("unittest.mock").MagicMock()
    mock_reader.read_oct_volume.return_value = study.oct_volume
    mock_reader.read_fundus_image.return_value = None
    mock_reader.read_all_metadata.return_value = {}

    monkeypatch.setattr("oct_converter_app.pipeline.detect_format", lambda p: "fds")
    monkeypatch.setattr(
        "oct_converter_app.pipeline.ReaderFactory.create", lambda fmt, path: mock_reader
    )

    request = ConversionRequest(
        input_path=str(study.source_path),
        output_dir=str(tmp_path),
        outputs=["dicom"],
        overwrite=True,
    )

    service = ConversionService(config=ConversionConfig(validate=False))
    result = service.convert(request)

    assert result.success is True
    assert len(result.generated_files) == 1

    dcm_path = result.generated_files[0]
    assert dcm_path.suffix == ".dcm"

    ds = pydicom.dcmread(dcm_path)
    assert ds.SOPClassUID == OPHTHALMIC_TOMOGRAPHY_SOP_CLASS_UID
