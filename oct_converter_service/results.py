"""Result model for OCT conversion operations."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ConversionResult:
    """Result of a conversion operation.

    This class provides structured information about the outcome
    of a conversion request, including success status, generated
    files, warnings, and metadata.

    Attributes:
        success: Whether the conversion was successful overall.
        input_path: Path to the input OCT file.
        output_dir: Directory where output files were written.
        detected_format: Detected input format (e.g., 'fds', 'fda').
        requested_outputs: List of requested output formats.
        generated_files: List of paths to created files.
        skipped_outputs: List of outputs that were skipped.
        failures: List of failure messages if any occurred.
        warnings: List of warning messages from processing.
        metadata: Extracted metadata dictionary.
        started_at: When processing started.
        completed_at: When processing completed.
    """

    success: bool
    input_path: Path
    output_dir: Path
    detected_format: str | None = None
    requested_outputs: list[str] = field(default_factory=list)
    generated_files: list[Path] = field(default_factory=list)
    skipped_outputs: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def elapsed_time(self) -> float | None:
        """Get elapsed time in seconds, or None if times not set."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @classmethod
    def from_study(
        cls,
        study: Any,
        success: bool,
        generated_files: list[Path],
        skipped_outputs: list[str] | None = None,
        failures: list[str] | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> "ConversionResult":
        """Create a ConversionResult from an OCTStudy.

        Args:
            study: The OCTStudy from the application layer.
            success: Overall success status.
            generated_files: List of created file paths.
            skipped_outputs: List of skipped output formats.
            failures: List of failure messages.
            started_at: When processing started.
            completed_at: When processing completed.

        Returns:
            ConversionResult instance.
        """
        return cls(
            success=success,
            input_path=study.source_path,
            output_dir=generated_files[0].parent if generated_files else Path.cwd(),
            detected_format=study.source_format,
            requested_outputs=[],  # Will be set by caller
            generated_files=generated_files,
            skipped_outputs=skipped_outputs or [],
            failures=failures or [],
            warnings=study.warnings.copy() if hasattr(study, "warnings") else [],
            metadata=study.metadata.copy() if hasattr(study, "metadata") else {},
            started_at=started_at,
            completed_at=completed_at,
        )
