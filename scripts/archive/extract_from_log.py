#!/usr/bin/env python3
"""Extract text from successful OCR run using the result object"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from processors.ocr_processor import OCRProcessor

# Re-run the extraction with proper saving
processor = OCRProcessor(default_dpi=800, save_debug_images=False, enable_semantic_validation=True)
pdf_path = Path("ramsey_data/truth_and_success_temp.pdf")

print(f"Re-extracting to properly save output...")
result = processor.extract_pdf(pdf_path, "poor")

if result and 'full_text' in result:
    # Save to data/extracted/ with proper naming
    output_file = Path("data/extracted/truth_and_success_temp.txt")
    output_file.write_text(result['full_text'], encoding='utf-8')
    print(f"\n✓ Saved {len(result['full_text'].split())} words to: {output_file}")
    print(f"Average OCR confidence: {result.get('avg_confidence', 0):.1f}%")
    print(f"Average semantic quality: {result.get('avg_dictionary_ratio', 0):.1f}% real words")
else:
    print(f"ERROR: No text in result")
