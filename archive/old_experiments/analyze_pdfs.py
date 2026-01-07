#!/usr/bin/env python3
"""Quick script to analyze PDF text extractability"""

import os
from pathlib import Path
import pdfplumber
from PyPDF2 import PdfReader

def analyze_pdf(pdf_path):
    """Analyze if PDF has extractable text or needs OCR"""
    try:
        # Get file size
        size_mb = os.path.getsize(pdf_path) / (1024 * 1024)

        # Try PyPDF2 first (faster)
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)

        # Sample first 3 pages for text content
        text_chars = 0
        sample_pages = min(3, num_pages)

        with pdfplumber.open(pdf_path) as pdf:
            for i in range(sample_pages):
                page = pdf.pages[i]
                text = page.extract_text()
                if text:
                    text_chars += len(text.strip())

        # Heuristic: if we get less than 100 chars per sampled page, likely scanned
        avg_chars_per_page = text_chars / sample_pages if sample_pages > 0 else 0
        needs_ocr = avg_chars_per_page < 100

        return {
            'filename': Path(pdf_path).name,
            'size_mb': round(size_mb, 2),
            'pages': num_pages,
            'sample_chars': text_chars,
            'avg_chars_per_page': round(avg_chars_per_page, 1),
            'needs_ocr': needs_ocr,
            'status': 'NEEDS OCR' if needs_ocr else 'Text OK'
        }
    except Exception as e:
        return {
            'filename': Path(pdf_path).name,
            'error': str(e),
            'status': 'ERROR'
        }

# Analyze all PDFs
data_dir = Path('ramsey_data')
pdf_files = sorted(data_dir.glob('*.pdf'))

print(f"Found {len(pdf_files)} PDF files\n")
print("="*100)

ocr_needed = []
text_ok = []
errors = []

for pdf_path in pdf_files:
    print(f"\nAnalyzing: {pdf_path.name[:80]}...")
    result = analyze_pdf(pdf_path)

    if result['status'] == 'ERROR':
        print(f"  ❌ ERROR: {result.get('error', 'Unknown error')}")
        errors.append(result)
    elif result['needs_ocr']:
        print(f"  ⚠️  NEEDS OCR")
        print(f"     Size: {result['size_mb']} MB | Pages: {result['pages']} | Chars/page: {result['avg_chars_per_page']}")
        ocr_needed.append(result)
    else:
        print(f"  ✓ Text extractable")
        print(f"     Size: {result['size_mb']} MB | Pages: {result['pages']} | Chars/page: {result['avg_chars_per_page']}")
        text_ok.append(result)

print("\n" + "="*100)
print("\n📊 SUMMARY:")
print(f"  ✓ Text-based PDFs: {len(text_ok)}")
print(f"  ⚠️  Need OCR: {len(ocr_needed)}")
print(f"  ❌ Errors: {len(errors)}")

if ocr_needed:
    print("\n⚠️  PDFs requiring OCR:")
    for pdf in ocr_needed:
        print(f"  - {pdf['filename'][:80]}")
        print(f"    ({pdf['size_mb']} MB, {pdf['pages']} pages, {pdf['avg_chars_per_page']} chars/page)")

if text_ok:
    print("\n✓ Text-based PDFs (ready for extraction):")
    for pdf in text_ok:
        print(f"  - {pdf['filename'][:80]}")
        print(f"    ({pdf['size_mb']} MB, {pdf['pages']} pages)")

if errors:
    print("\n❌ PDFs with errors:")
    for pdf in errors:
        print(f"  - {pdf['filename'][:80]}")
        print(f"    Error: {pdf.get('error', 'Unknown')}")

print("\n" + "="*100)
