# Canonical OCTStudy Model

## Purpose

`OCTStudy` is the central canonical data model for the OCT ingestion pipeline. It represents the normalized, validated output of parsing a vendor-specific OCT file. The model serves as the boundary between vendor-specific format readers and downstream consumers (exporters, analysis tools, etc.).

The design principle is: **parse once → canonicalize once → validate once → export many ways**.

## Reader Contract

A reader that produces an `OCTStudy` must:

1. **Extract OCT volume** (if present) as an `OCTVolumeWithMetaData` object containing B-scans with pixel data and associated metadata.

2. **Extract fundus image** (if present) as a `FundusImageWithMetaData` object.

3. **Preserve raw metadata** without interpretation in `raw_metadata`. This ensures no vendor-specific information is lost.

4. **Populate normalized metadata** in `metadata` only for fields that can be reliably standardized across vendors (e.g., `patient_id`, `laterality`, `acquisition_date`).

5. **Not fabricate values**. Unknown or unavailable fields should remain `None` or empty.

6. **Record provenance** including source path, format, processing timestamp, and optionally a SHA-256 hash of the source file.

7. **Optionally populate `vendor_device`** if reliable manufacturer/model/software_version information is available from the source file. Do not assume vendor identity from the reader class name.

## Capabilities

The `StudyCapabilities` class indicates which data modalities and features are available from a processed study. It is computed dynamically from the current state of the `OCTStudy` to prevent stale reports when the study is modified.

### Current Capability Flags

| Flag | Meaning |
|------|---------|
| `has_oct_volume` | OCT volume data is present and contains non-empty slices |
| `has_fundus` | Fundus image is present with non-zero size |
| `has_metadata` | Normalized metadata dictionary is non-empty |
| `has_pixel_spacing` | Pixel spacing calibration is available |
| `has_contours` | Segmentation contours are attached to the OCT volume |
| `num_bscans` | Number of valid B-scans (0 if no OCT or all slices empty) |
| `fundus_shape` | Tuple `(H, W, C)` describing fundus image dimensions, or `None` |

### Future-Facing Flags

These flags default to `False` and will be populated when those features are implemented:

| Flag | Meaning (Future) |
|------|------------------|
| `has_octa` | OCT angiography data is available |
| `has_layer_segmentation` | Retinal layer segmentation masks are available |

**Important**: Capability values are computed on access via the `capabilities` property. They cannot become stale if `study.oct_volume` or `study.fundus` is modified after construction.

## Vendor/Device Information

The `VendorDevice` structure provides a typed location for device identification:

```python
@dataclass(frozen=True)
class VendorDevice:
    manufacturer: str | None = None
    model: str | None = None
    software_version: str | None = None
```

**Population rules**:
- Values must come from reliable vendor metadata in the source file.
- Do not fabricate values based on assumptions (e.g., do not set `manufacturer="Topcon"` merely because the reader class is named `FDSReader`).
- All fields are optional; use `None` when unknown.
- The structure is frozen (immutable) to prevent accidental modification.

## Metadata Policy

### Normalized Metadata (`metadata`)

Common fields that can be reliably standardized across vendors:
- `patient_id`
- `laterality`
- `acquisition_date`
- `patient_name`
- `sex`
- `date_of_birth`

Rules:
- Use `None` for unknown values.
- Do not copy the entire raw vendor tree into normalized fields.
- Preserve the distinction between normalized and raw metadata.

### Raw Metadata (`raw_metadata`)

Vendor-specific fields retained without interpretation:
- Preserved exactly as extracted from the source file.
- May contain nested structures, vendor-specific naming conventions, and proprietary fields.
- Not included in the PHI-safe `repr()` output.

## Provenance

The `Provenance` class tracks processing history:

| Field | Description |
|-------|-------------|
| `source_path` | Original file path |
| `source_format` | Detected format identifier (e.g., "fds", "fda") |
| `processing_timestamp` | When the file was processed |
| `file_hash` | SHA-256 hash of source file (if computed) |
| `reader_version` | Version string of the reader/package used |

Provenance is created during pipeline execution and should not be modified afterward.

## Derived Products

The `DerivedProduct` class is an extension point for future-derived imaging products:

```python
@dataclass
class DerivedProduct:
    product_type: str  # e.g., "octa_enface", "layer_segmentation", "thickness_map"
    data: Any = None   # Temporary extension point; may be refined later
    metadata: dict[str, Any] = field(default_factory=dict)
    creation_method: str | None = None
```

**What this means**:
- Studies can attach derived products (segmentation masks, thickness maps, OCTA en-face images, etc.) via `study.add_derived_product()`.
- The `data` field is currently untyped (`Any`) as a placeholder for future refinement.
- Each study has its own independent list of derived products (no shared mutable defaults).

**What this does NOT mean**:
- This is not an implementation of segmentation or OCTA algorithms.
- This is not a permanent schema; the structure may evolve in future phases.
- Exporters are not required to handle derived products (existing exporters ignore them).

## PHI-Safe Representation

The `OCTStudy.__repr__()` method intentionally excludes:
- `metadata` content
- `raw_metadata` content
- Patient identifiers, names, dates of birth, or other PHI

Example safe output:
```
OCTStudy(source_format='fds', oct_volume=128 B-scans, fundus=(1536, 2048, 3), caps(oct=True, fundus=True))
```

This prevents accidental PHI exposure in logs, debugging output, or error messages.

## Compatibility

Existing APIs remain unchanged:

| Property/Attribute | Status |
|--------------------|--------|
| `study.oct_volume` | Unchanged |
| `study.fundus` | Unchanged |
| `study.patient_id` | Unchanged (property) |
| `study.laterality` | Unchanged (property) |
| `study.acquisition_date` | Unchanged (property) |
| `study.metadata` | Unchanged |
| `study.raw_metadata` | Unchanged |
| `study.warnings` | Unchanged |
| `study.source_path` | Unchanged |
| `study.source_format` | Unchanged |
| `study.provenance` | Unchanged |
| `study.capabilities` | Now a computed property (was optional cached attribute) |

New additions:
- `study.vendor_device`: Optional `VendorDevice` instance.
- `study.derived_products`: List of `DerivedProduct` instances (empty by default).
- `study.add_derived_product(product)`: Method to add derived products.

Exporters continue to consume `OCTStudy` objects without modification. The `capabilities` property now returns `StudyCapabilities` instead of the legacy `Capabilities` class, but both classes share the same core fields for backward compatibility.

## External Reuse

**eyepy reference**: Used as a design reference only. No code was copied or adapted from eyepy. No dependency added.

Key concepts referenced:
- Unified OCT/fundus volume representation
- Acquisition metadata separation
- Derived information attached to (not replacing) the core volume
