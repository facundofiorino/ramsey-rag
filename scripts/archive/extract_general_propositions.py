#!/usr/bin/env python3
"""Extract General Propositions and save properly"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from processors.ocr_processor import OCRProcessor

# Extract General Propositions
processor = OCRProcessor(default_dpi=800, save_debug_images=False, enable_semantic_validation=True)
pdf_path = Path("ramsey_data/Ramsey, Frank - General Propositions and Causality - libgen.li.pdf")

print(f"Extracting General Propositions...")
result = processor.extract_pdf(pdf_path, "poor")

if result and 'full_text' in result:
    # Save to data/extracted/
    output_file = Path("data/extracted/general_propositions_ocr.txt")
    output_file.write_text(result['full_text'], encoding='utf-8')
    print(f"\n✓ Saved {len(result['full_text'].split())} words to: {output_file}")
    print(f"Average OCR confidence: {result.get('avg_confidence', 0):.1f}%")
    print(f"Average semantic quality: {result.get('avg_dictionary_ratio', 0):.1f}% real words")
else:
    print(f"ERROR: No text in result")
