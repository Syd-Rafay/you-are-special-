"""Configuration for the OCT conversion service."""

from dataclasses import dataclass


@dataclass
class ConversionConfig:
    """Configuration options for conversion operations.

    Attributes:
        overwrite: Whether to overwrite existing output files.
        validate: Whether to validate extracted data before export.
        continue_on_warning: Whether to continue if validation has warnings.
        compute_hash: Whether to compute SHA-256 hash of source file.
    """

    overwrite: bool = False
    validate: bool = True
    continue_on_warning: bool = True
    compute_hash: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.overwrite, bool):
            raise ValueError("overwrite must be a boolean")
        if not isinstance(self.validate, bool):
            raise ValueError("validate must be a boolean")
        if not isinstance(self.continue_on_warning, bool):
            raise ValueError("continue_on_warning must be a boolean")
        if not isinstance(self.compute_hash, bool):
            raise ValueError("compute_hash must be a boolean")
