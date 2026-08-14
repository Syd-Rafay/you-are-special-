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
        overwrite: Whether to overwrite existing output files.
        validate: Whether to validate extracted data before export.
        continue_on_warning: Whether to continue if validation has warnings.
        compute_hash: Whether to compute SHA-256 hash of source file.
        exporter_options: Per-exporter configuration options.
                          Dict mapping exporter name to options dict.
    """

    input_path: Path | str
    output_dir: Path | str
    outputs: list[str] = field(default_factory=lambda: ["metadata"])
    overwrite: bool = False
    validate: bool = True
    continue_on_warning: bool = True
    compute_hash: bool = False
    exporter_options: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the request after initialization."""
        self._validate()

    def _validate(self):
        """Validate request parameters.

        Raises:
            ValueError: If any parameter is invalid.
        """
        errors = []

        # Validate input path
        input_path = Path(self.input_path)
        if not input_path.exists():
            errors.append(f"Input path does not exist: {input_path}")
        elif not input_path.is_file():
            errors.append(f"Input path must be a file, not a directory: {input_path}")

        # Validate output directory
        output_dir = Path(self.output_dir)
        if output_dir.exists() and not output_dir.is_dir():
            errors.append(f"Output path exists but is not a directory: {output_dir}")

        # Validate outputs list
        valid_outputs = {"dicom", "npy", "images", "metadata"}
        if not self.outputs:
            errors.append("At least one output format must be specified")
        else:
            for output in self.outputs:
                if output not in valid_outputs:
                    errors.append(
                        f"Unsupported output format: '{output}'. "
                        f"Valid options: {sorted(valid_outputs)}"
                    )

        # Validate boolean fields
        if not isinstance(self.overwrite, bool):
            errors.append("overwrite must be a boolean")
        if not isinstance(self.validate, bool):
            errors.append("validate must be a boolean")
        if not isinstance(self.continue_on_warning, bool):
            errors.append("continue_on_warning must be a boolean")
        if not isinstance(self.compute_hash, bool):
            errors.append("compute_hash must be a boolean")

        # Validate exporter_options is a dict
        if not isinstance(self.exporter_options, dict):
            errors.append("exporter_options must be a dictionary")

        if errors:
            raise ValueError(f"Invalid ConversionRequest: {'; '.join(errors)}")
