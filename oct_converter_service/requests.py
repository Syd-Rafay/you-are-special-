"""Request model for OCT conversion operations."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ConversionRequest:
    """A request to convert an OCT file.

    This class represents a single conversion operation with all
    necessary parameters and configuration.

    Attributes:
        input_path: Path to the input OCT file.
        output_dir: Directory where output files will be written.
        outputs: List of output formats to generate.
                 Valid options: 'dicom', 'npy', 'images', 'metadata'
        overwrite: Optional override for overwriting existing output files.
        validate: Optional override for validating extracted data before export.
        continue_on_warning: Optional override for continuing if validation has warnings.
        compute_hash: Optional override for computing SHA-256 hash of source file.
        exporter_options: Per-exporter configuration options.
                          Dict mapping exporter name to options dict.
    """

    input_path: Path | str
    output_dir: Path | str
    outputs: list[str] = field(default_factory=lambda: ["metadata"])
    overwrite: bool | None = None
    validate: bool | None = None
    continue_on_warning: bool | None = None
    compute_hash: bool | None = None
    exporter_options: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the request schema after initialization."""
        self._validate()

    def validate_request(self):
        """Validate request parameters explicitly.

        Raises:
            ValueError: If any parameter is invalid.
        """
        self._validate()

    def _validate(self):
        """Validate request parameters schema.

        Raises:
            ValueError: If any parameter is invalid.
        """
        errors = []

        # Validate input path presence and type
        if not self.input_path:
            errors.append("Input path must be provided")

        # Validate output directory type and non-file check
        if not self.output_dir:
            errors.append("Output directory must be provided")
        else:
            output_dir = Path(self.output_dir)
            if output_dir.exists() and not output_dir.is_dir():
                errors.append(f"Output path exists but is not a directory: {output_dir}")

        # Validate outputs list
        valid_outputs = {"dicom", "npy", "images", "metadata", "zarr"}
        if not self.outputs:
            errors.append("At least one output format must be specified")
        else:
            for output in self.outputs:
                if output not in valid_outputs:
                    errors.append(
                        f"Unsupported output format: '{output}'. "
                        f"Valid options: {sorted(valid_outputs)}"
                    )

        # Validate optional boolean fields if set
        if self.overwrite is not None and not isinstance(self.overwrite, bool):
            errors.append("overwrite must be a boolean or None")
        if self.validate is not None and not isinstance(self.validate, bool):
            errors.append("validate must be a boolean or None")
        if self.continue_on_warning is not None and not isinstance(self.continue_on_warning, bool):
            errors.append("continue_on_warning must be a boolean or None")
        if self.compute_hash is not None and not isinstance(self.compute_hash, bool):
            errors.append("compute_hash must be a boolean or None")

        # Validate exporter_options is a dict
        if not isinstance(self.exporter_options, dict):
            errors.append("exporter_options must be a dictionary")

        if errors:
            raise ValueError(f"Invalid ConversionRequest: {'; '.join(errors)}")
