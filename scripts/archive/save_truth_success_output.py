#!/usr/bin/env python3
"""Save the Truth and Success OCR output with proper naming"""
from pathlib import Path

# The extraction completed - we need to save it with the right name
# Output files are named ocr_output_<stem>.txt by the processor

possible_names = [
    "ocr_output_truth_and_success_temp.txt",
    "truth_and_success_temp.txt",
]

output_dir = Path("data/extracted")
target_name = "truth_and_success_ocr.txt"

# Check if any output file exists
found = None
for name in possible_names:
    if Path(name).exists():
        found = Path(name)
        break

if found:
    # Move to proper location
    target = output_dir / target_name
    found.rename(target)
    print(f"✓ Moved {found} to {target}")
    word_count = len(target.read_text().split())
    print(f"✓ File contains {word_count:,} words")
else:
    print("File not found - OCR may not have saved output properly")
    print("Need to re-run extraction")
