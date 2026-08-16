"""DICOM exporter for structural OCT volumes using Ophthalmic Tomography Image IOD.

This exporter creates DICOM Ophthalmic Tomography Image Storage objects
(SOP Class UID: 1.2.840.10008.5.1.4.1.1.77.1.5.4) directly from the
in-memory OCTStudy object, without re-reading the source file.

The implementation follows the DICOM standard requirements for:
- Ophthalmic Tomography Image IOD (PS3.3 A.52)
- Ophthalmic Tomography Series (PS3.3 C.8.17.6)
- Ophthalmic Tomography Image (PS3.3 C.8.17.7)
- Ophthalmic Tomography Acquisition Parameters (PS3.3 C.8.17.8)
- Ophthalmic Tomography Parameters (PS3.3 C.8.17.9)
- Multi-frame Functional Groups (PS3.3 C.7.6.16)
- Image Pixel Module

Pixel data is preserved exactly without normalization, rescaling, or conversion.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    OphthalmicTomographyImageStorage,
    generate_uid,
)
from pydicom.valuerep import DSfloat

from oct_converter_app.exporters.base import BaseExporter, ExportError
from oct_converter_app.models import OCTStudy


# Deterministic implementation UID based on package name and version
def _get_implementation_uid() -> str:
    """Get implementation UID for this software."""
    try:
        from importlib import metadata
        version = metadata.version("oct_converter") or "0.1.0"
    except Exception:
        version = "0.1.0"
    return generate_uid(entropy_srcs=["oct_converter", version])


_IMPLEMENTATION_UID = _get_implementation_uid()


class DicomExporter(BaseExporter):
    """Exporter for DICOM Ophthalmic Tomography Image Storage format.
    
    This exporter creates multi-frame DICOM objects representing structural
    OCT volumes using the Ophthalmic Tomography Image IOD.
    
    The volume is encoded as:
    - NumberOfFrames = number of B-scans (z-axis)
    - Rows = B-scan height (y-axis)
    - Columns = B-scan width (x-axis)
    
    Pixel data is preserved exactly without modification.
    
    Attributes:
        name: Exporter name ('dicom').
    """

    name = "dicom"

    def export(
        self, study: OCTStudy, output_dir: Path | str, options: dict | None = None
    ) -> list[Path]:
        """Export study to DICOM Ophthalmic Tomography Image Storage format.
        
        This method creates DICOM files directly from the in-memory OCTStudy
        object without re-reading the source file.
        
        Args:
            study: The OCTStudy containing OCT volume data.
            output_dir: Directory to write DICOM files.
            options: Optional overrides. Currently supports 'overwrite'.
        
        Returns:
            List of paths to created DICOM files.
        
        Raises:
            ExportError: If DICOM export fails due to missing data or errors.
        """
        output_path = self._ensure_output_dir(output_dir)
        
        # Check overwrite option
        options_copy = dict(options) if options else {}
        overwrite = options_copy.pop("overwrite", True)
        
        if not overwrite:
            stem = study.source_path.stem
            existing_dicoms = list(output_path.glob(f"{stem}*.dcm"))
            if existing_dicoms:
                raise ExportError(
                    f"File already exists and overwrite is disabled: {existing_dicoms[0]}"
                )
        
        # Check that OCT volume data exists
        if study.oct_volume is None or study.oct_volume.volume is None:
            raise ExportError("No OCT volume data available for DICOM export")
        
        volume = study.oct_volume.volume
        
        # Handle list of slices vs array
        if isinstance(volume, list):
            if len(volume) == 0:
                raise ExportError("OCT volume contains no B-scans")
            # Stack into 3D array preserving dtype
            pixel_data = np.stack([np.asarray(slice) for slice in volume])
        else:
            pixel_data = np.asarray(volume)
        
        # Ensure 3D array [z, y, x]
        if pixel_data.ndim != 3:
            raise ExportError(
                f"Expected 3D volume [z, y, x], got {pixel_data.ndim}D"
            )
        
        num_frames, rows, cols = pixel_data.shape
        
        try:
            # Create DICOM file
            filepath = output_path / f"{study.source_path.stem}.dcm"
            ds = _create_ophthalmic_tomography_dicom(
                study=study,
                pixel_data=pixel_data,
                filepath=filepath,
            )
            return [filepath]
        except Exception as e:
            raise ExportError(
                f"DICOM export failed for {study.source_path}: {e}"
            ) from e

    def supports_oct(self, study: OCTStudy) -> bool:
        """Check if DICOM export is possible for this study.
        
        DICOM export requires OCT volume data to be present.
        
        Args:
            study: The study to check.
        
        Returns:
            True if OCT volume data exists.
        """
        if study.oct_volume is None:
            return False
        if study.oct_volume.volume is None:
            return False
        vol = study.oct_volume.volume
        if isinstance(vol, list):
            return len(vol) > 0
        return hasattr(vol, "__len__") and len(vol) > 0


def _create_ophthalmic_tomography_dicom(
    study: OCTStudy,
    pixel_data: np.ndarray,
    filepath: Path,
) -> FileDataset:
    """Create a DICOM Ophthalmic Tomography Image Storage object.
    
    This function creates a complete DICOM file following the Ophthalmic
    Tomography Image IOD requirements.
    
    Args:
        study: OCTStudy with metadata.
        pixel_data: 3D numpy array [z, y, x] with volume data.
        filepath: Output file path.
    
    Returns:
        FileDataset that was written to disk.
    """
    num_frames, rows, cols = pixel_data.shape
    
    # Create file meta
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = OphthalmicTomographyImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = _IMPLEMENTATION_UID
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    
    # Create dataset
    ds = FileDataset(str(filepath), {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Patient Module (PS3.3 C.7.1.1)
    patient_id = study.patient_id or ""
    ds.PatientID = patient_id
    ds.PatientName = ""  # Do not fabricate
    ds.PatientSex = ""  # Not available from canonical model
    ds.PatientBirthDate = ""  # Not available from canonical model
    
    # Study Module (PS3.3 C.7.2.1)
    ds.StudyInstanceUID = generate_uid()
    ds.StudyID = ""
    ds.StudyDate = ""
    ds.StudyTime = ""
    
    # Series Module (PS3.3 C.7.3.1) + Ophthalmic Tomography Series (PS3.3 C.8.17.6)
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "OPT"
    ds.SeriesNumber = 1
    ds.Laterality = study.laterality or ""
    ds.ProtocolName = ""
    ds.SeriesDescription = "Structural OCT Volume"
    
    # Equipment Module (PS3.3 C.7.5.1)
    if study.vendor_device:
        ds.Manufacturer = study.vendor_device.manufacturer or ""
        ds.ManufacturerModelName = study.vendor_device.model or ""
        ds.DeviceSerialNumber = study.vendor_device.device_serial or ""
        ds.SoftwareVersions = study.vendor_device.software_version or ""
    else:
        ds.Manufacturer = ""
        ds.ManufacturerModelName = ""
        ds.DeviceSerialNumber = ""
        ds.SoftwareVersions = ""
    
    # SOP Common Module (PS3.3 C.12.1)
    ds.SOPClassUID = OphthalmicTomographyImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    
    # Ocular Region Imaged Module (PS3.3 C.8.17.5)
    ds.ImageLaterality = study.laterality or ""
    # Use generic retina code if laterality is known
    if study.laterality:
        ds.AnatomicRegionSequence = [Dataset()]
        ds.AnatomicRegionSequence[0].CodeValue = "T-AA610"
        ds.AnatomicRegionSequence[0].CodingSchemeDesignator = "SRT"
        ds.AnatomicRegionSequence[0].CodeMeaning = "Retina"
    
    # Image Pixel Module (PS3.3 C.7.6.3)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0  # Unsigned integer
    ds.NumberOfFrames = num_frames
    ds.Rows = rows
    ds.Columns = cols
    
    # Derive bits from dtype
    if pixel_data.dtype == np.uint16:
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
    elif pixel_data.dtype == np.uint8:
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
    else:
        # Default to 16-bit for other types
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
    
    # Ophthalmic Tomography Image Module (PS3.3 C.8.17.7)
    ds.ImageType = ["DERIVED", "SECONDARY", "VOLUME"]
    ds.OphthalmicVolumetricPropertiesFlag = 'Y'
    
    # Acquisition DateTime from study if available
    acq_date = study.acquisition_date
    if acq_date:
        if isinstance(acq_date, str):
            try:
                acq_date = datetime.strptime(acq_date, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                acq_date = None
        if acq_date:
            ds.AcquisitionDateTime = acq_date.strftime("%Y%m%d%H%M%S.%f")
        else:
            ds.AcquisitionDateTime = ""
    else:
        ds.AcquisitionDateTime = ""
    
    ds.AcquisitionNumber = 1
    
    # Content Date/Time (when DICOM was created)
    now = datetime.now()
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S.%f")
    ds.InstanceNumber = 1
    
    # Shared Functional Groups (PS3.3 C.7.6.16)
    shared_ds = [Dataset()]
    
    # Frame Anatomy Sequence
    shared_ds[0].FrameAnatomySequence = [Dataset()]
    shared_ds[0].FrameAnatomySequence[0].FrameLaterality = study.laterality or ""
    if study.laterality:
        shared_ds[0].FrameAnatomySequence[0].AnatomicRegionSequence = [
            ds.AnatomicRegionSequence[0].copy()
        ]
    
    # Pixel Measures Sequence (PS3.3 C.7.6.16.2.1)
    # Use vendor-aware spacing semantics from Phase 5
    shared_ds[0].PixelMeasuresSequence = [Dataset()]
    
    # Get pixel spacing from OCT volume
    pixel_spacing = None
    slice_thickness = 0.05  # Default fallback
    
    if study.oct_volume and study.oct_volume.pixel_spacing:
        ps = study.oct_volume.pixel_spacing
        if isinstance(ps, (list, tuple)) and len(ps) >= 2:
            # Phase 5 corrected semantics:
            # For FDA/FDS: pixel_spacing = [width (x), slice_thickness (z), height (y)]
            # We need: PixelSpacing = [row spacing (y), column spacing (x)]
            # And SliceThickness = z spacing
            if len(ps) >= 3:
                # Assume [x, z, y] format from FDA/FDS readers
                pixel_spacing = [ps[2], ps[0]]  # [y, x]
                slice_thickness = ps[1]  # z
            else:
                # Assume [y, x] format
                pixel_spacing = list(ps[:2])
                slice_thickness = 0.05
    
    if pixel_spacing:
        shared_ds[0].PixelMeasuresSequence[0].PixelSpacing = [
            DSfloat(pixel_spacing[0], auto_format=True),
            DSfloat(pixel_spacing[1], auto_format=True),
        ]
    else:
        # No spacing available - use empty/default
        shared_ds[0].PixelMeasuresSequence[0].PixelSpacing = [
            DSfloat(1.0, auto_format=True),
            DSfloat(1.0, auto_format=True),
        ]
    
    shared_ds[0].PixelMeasuresSequence[0].SliceThickness = DSfloat(
        slice_thickness, auto_format=True
    )
    
    # Plane Orientation Sequence (PS3.3 C.7.6.16.2.4)
    shared_ds[0].PlaneOrientationSequence = [Dataset()]
    shared_ds[0].PlaneOrientationSequence[0].ImageOrientationPatient = [
        DSfloat(1, auto_format=True),
        DSfloat(0, auto_format=True),
        DSfloat(0, auto_format=True),
        DSfloat(0, auto_format=True),
        DSfloat(1, auto_format=True),
        DSfloat(0, auto_format=True),
    ]
    
    ds.SharedFunctionalGroupsSequence = shared_ds
    
    # Per-Frame Functional Groups
    per_frame = []
    for i in range(num_frames):
        frame_fgs = Dataset()
        
        # Plane Position Sequence (PS3.3 C.7.6.16.2.2)
        frame_fgs.PlanePositionSequence = [Dataset()]
        ipp = [
            DSfloat(0, auto_format=True),
            DSfloat(0, auto_format=True),
            DSfloat(i * slice_thickness, auto_format=True),
        ]
        frame_fgs.PlanePositionSequence[0].ImagePositionPatient = ipp
        
        # Frame Content Sequence
        frame_fgs.FrameContentSequence = [Dataset()]
        frame_fgs.FrameContentSequence[0].InStackPositionNumber = i + 1
        frame_fgs.FrameContentSequence[0].StackID = "1"
        
        per_frame.append(frame_fgs)
    
    ds.PerFrameFunctionalGroupsSequence = per_frame
    
    # Set pixel data - preserve exact values
    # Ensure proper byte order for Explicit VR Little Endian
    if pixel_data.dtype != np.uint16:
        pixel_data = pixel_data.astype(np.uint16)
    
    ds.PixelData = pixel_data.tobytes()
    
    # Save the file
    ds.save_as(filepath, implicit_vr=False, little_endian=True, enforce_file_format=True)
    
    return ds
