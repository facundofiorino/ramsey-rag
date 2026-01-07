#!/usr/bin/env python3
"""Helper script to rename file with Unicode characters"""
import shutil
from pathlib import Path

# Find the file using glob
source_dir = Path("ramsey_data")
matches = list(source_dir.glob("*Truth and Success*.pdf"))

if matches:
    source = matches[0]
    dest = source_dir / "truth_and_success_temp.pdf"
    print(f"Found: {source}")
    print(f"Copying to: {dest}")
    shutil.copy2(source, dest)
    print("✓ Copy successful")
else:
    print("ERROR: File not found")
