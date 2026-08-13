#!/usr/bin/env python3
"""Command-line interface for OCT file conversion.

Provides a user-friendly CLI for converting OCT files to various output formats.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oct_converter_app.detector import FormatDetector
from oct_converter_service import (
    ConversionRequest,
    ConversionService,
    ConversionServiceError,
)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="oct-convert",
        description="Convert proprietary OCT files to standard formats (DICOM, NPY, images, metadata).",
        epilog=(
            "Examples:\n"
            "  oct-convert scan.fds output/ --dicom\n"
            "  oct-convert scan.fda output/ --npy --images --metadata\n"
            "  oct-convert scan.e2e output/ --all\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional arguments
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input OCT file (.fds, .fda, .e2e, .img, .OCT, .dcm)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for converted files",
    )

    # Output format selection
    output_group = parser.add_argument_group("Output formats")
    output_group.add_argument(
        "--dicom",
        action="store_true",
        help="Export as DICOM files",
    )
    output_group.add_argument(
        "--npy",
        action="store_true",
        help="Export OCT volume and fundus as NumPy .npy files",
    )
    output_group.add_argument(
        "--images",
        action="store_true",
        help="Export B-scans and fundus as PNG images",
    )
    output_group.add_argument(
        "--metadata",
        action="store_true",
        help="Export metadata as JSON file",
    )
    output_group.add_argument(
        "--all",
        action="store_true",
        help="Export all formats (DICOM, NPY, images, metadata)",
    )

    # Options
    options_group = parser.add_argument_group("Options")
    options_group.add_argument(
        "--image-format",
        type=str,
        default="png",
        choices=["png", "jpg", "tiff"],
        help="Image format for --images export (default: png)",
    )
    options_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if present",
    )
    options_group.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation checks",
    )
    options_group.add_argument(
        "--stop-on-warning",
        action="store_true",
        help="Stop processing if validation warnings occur",
    )
    options_group.add_argument(
        "--compute-hash",
        action="store_true",
        help="Compute SHA-256 hash of source file (adds I/O overhead)",
    )
    options_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output with detailed information",
    )

    # Info
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    parser.add_argument(
        "--list-formats",
        action="store_true",
        help="List supported input formats and exit",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle --list-formats
    if args.list_formats:
        print("Supported input formats:")
        for fmt in sorted(FormatDetector.supported_formats()):
            print(f"  - {fmt}")
        return 0

    # Validate output selection
    outputs = []
    if args.all:
        outputs = ["dicom", "npy", "images", "metadata"]
    else:
        if args.dicom:
            outputs.append("dicom")
        if args.npy:
            outputs.append("npy")
        if args.images:
            outputs.append("images")
        if args.metadata:
            outputs.append("metadata")

    # Default to metadata if no output specified
    if not outputs:
        outputs = ["metadata"]
        if args.verbose:
            print("Note: No output format specified, defaulting to metadata only.")

    # Build exporter options
    exporter_options = {}
    if "images" in outputs:
        exporter_options["images"] = {"format": args.image_format}

    # Build service conversion request
    try:
        request = ConversionRequest(
            input_path=args.input_file,
            output_dir=args.output_dir,
            outputs=outputs,
            overwrite=args.overwrite,
            validate=not args.no_validate,
            continue_on_warning=not args.stop_on_warning,
            compute_hash=args.compute_hash,
            exporter_options=exporter_options,
        )

        service = ConversionService()
        result = service.convert(request)

        if result.warnings:
            for warning in result.warnings:
                print(f"WARNING: {warning}", file=sys.stderr)

        if not result.success:
            for failure in result.failures:
                print(f"ERROR: {failure}", file=sys.stderr)
            return 1

        if args.verbose:
            print(f"\nProcessing complete for: {args.input_file}")
            if result.detected_format:
                print(f"  Format: {result.detected_format}")
            print(f"  Generated {len(result.generated_files)} file(s)")

        print(f"\nSuccessfully processed: {args.input_file.name}")
        return 0

    except ConversionServiceError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
