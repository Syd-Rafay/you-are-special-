"""Exception hierarchy for the OCT conversion service."""


class ConversionServiceError(Exception):
    """Base exception for all conversion service errors.

    All service-level exceptions inherit from this base class,
    allowing callers to catch all service errors with a single handler.
    """

    pass


class InvalidRequestError(ConversionServiceError):
    """Raised when a conversion request is invalid.

    This includes:
    - Missing or invalid input path
    - Invalid output directory
    - Unsupported or missing output formats
    - Invalid configuration options
    """

    pass


class InputNotFoundError(ConversionServiceError):
    """Raised when the input file does not exist."""

    pass


class UnsupportedFormatError(ConversionServiceError):
    """Raised when an unsupported input or output format is requested."""

    pass


class ConversionFailedError(ConversionServiceError):
    """Raised when the conversion process fails.

    This is a general error for failures during the actual
    conversion/extraction process.
    """

    pass


class OutputError(ConversionServiceError):
    """Raised when there is an error writing output files.

    This includes:
    - Cannot create output directory
    - Permission errors
    - Disk space issues
    """

    pass


class OverwriteNotAllowedError(ConversionServiceError):
    """Raised when overwrite is disabled but output files already exist."""

    pass
