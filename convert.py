from pathlib import Path
import sys
from oct_converter.dicom import create_dicom_from_oct

if len(sys.argv) != 3:
    print("Usage: python convert.py <input_file> <output_dir>")
    sys.exit(1)

input_file = Path(sys.argv[1])
output_dir = Path(sys.argv[2])

if not input_file.exists():
    print(f"ERROR: Input file does not exist: {input_file}")
    sys.exit(1)

try:
    files = create_dicom_from_oct(str(input_file), output_dir=str(output_dir))
    print("Conversion successful.")
    for file in files:
        print(f"Created: {file}")
except Exception as e:
    print(f"ERROR: Conversion failed: {e}")
    sys.exit(1)
