#!/usr/bin/env python3
"""
Quick PDF Assessment Tool

Checks each PDF to determine:
1. Does it have native text (searchable)?
2. What's the text quality?
3. Does it need OCR?
"""

import sys
from pathlib import Path
from typing import Dict, List

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("⚠️  Install PyPDF2: pip install PyPDF2")
    sys.exit(1)


def assess_pdf(pdf_path: Path, sample_pages: int = 3) -> Dict:
    """
    Assess a PDF to determine extraction strategy

    Args:
        pdf_path: Path to PDF file
        sample_pages: Number of pages to sample

    Returns:
        Assessment results
    """
    result = {
        'file': pdf_path.name,
        'file_size_mb': pdf_path.stat().st_size / (1024 * 1024),
        'total_pages': 0,
        'has_text': False,
        'avg_chars_per_page': 0,
        'sample_text': '',
        'recommendation': '',
        'extraction_method': '',
        'estimated_time': ''
    }

    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            result['total_pages'] = len(reader.pages)

            # Sample first few pages
            total_chars = 0
            sample_texts = []
            pages_to_check = min(sample_pages, result['total_pages'])

            for i in range(pages_to_check):
                try:
                    page = reader.pages[i]
                    text = page.extract_text()
                    total_chars += len(text)
                    if i == 0:  # Save first page sample
                        sample_texts.append(text[:500])
                except Exception as e:
                    print(f"  Error reading page {i}: {e}")

            result['avg_chars_per_page'] = total_chars / pages_to_check if pages_to_check > 0 else 0
            result['sample_text'] = '\n'.join(sample_texts)
            result['has_text'] = result['avg_chars_per_page'] > 50

            # Determine recommendation
            if result['avg_chars_per_page'] > 1000:
                result['recommendation'] = '✅ EXCELLENT - Native text extraction'
                result['extraction_method'] = 'Native (PyPDF2)'
                result['estimated_time'] = f'~{result["total_pages"] * 0.01:.1f}s'
            elif result['avg_chars_per_page'] > 500:
                result['recommendation'] = '✅ GOOD - Native text extraction'
                result['extraction_method'] = 'Native (PyPDF2)'
                result['estimated_time'] = f'~{result["total_pages"] * 0.01:.1f}s'
            elif result['avg_chars_per_page'] > 100:
                result['recommendation'] = '⚠️ POOR - Verify quality, may need OCR'
                result['extraction_method'] = 'Native first, OCR if needed'
                result['estimated_time'] = f'~{result["total_pages"] * 0.02:.1f}s + OCR time'
            elif result['avg_chars_per_page'] > 10:
                result['recommendation'] = '❌ VERY POOR - OCR required'
                result['extraction_method'] = 'OCR (800 DPI + Ultra)'
                result['estimated_time'] = f'~{result["total_pages"] * 12:.0f}s ({result["total_pages"] * 12 / 60:.1f}min)'
            else:
                result['recommendation'] = '❌ NO TEXT - OCR required'
                result['extraction_method'] = 'OCR (800 DPI + Ultra)'
                result['estimated_time'] = f'~{result["total_pages"] * 12:.0f}s ({result["total_pages"] * 12 / 60:.1f}min)'

    except Exception as e:
        result['error'] = str(e)
        result['recommendation'] = f'❌ ERROR: {e}'

    return result


def main():
    """Assess all PDFs in ramsey_data directory"""

    # Get project root (two levels up from src/utils/)
    project_root = Path(__file__).parent.parent.parent
    ramsey_dir = project_root / 'ramsey_data'
    if not ramsey_dir.exists():
        print(f"Error: {ramsey_dir} not found")
        sys.exit(1)

    pdf_files = list(ramsey_dir.glob('*.pdf'))
    print(f"\n{'='*100}")
    print(f"PDF ASSESSMENT REPORT")
    print(f"{'='*100}\n")
    print(f"Found {len(pdf_files)} PDF files\n")

    results = []
    for pdf_file in sorted(pdf_files):
        print(f"Assessing: {pdf_file.name[:80]}...")
        result = assess_pdf(pdf_file)
        results.append(result)
        print(f"  Pages: {result['total_pages']}")
        print(f"  Size: {result['file_size_mb']:.1f} MB")
        print(f"  Avg chars/page: {result['avg_chars_per_page']:.1f}")
        print(f"  {result['recommendation']}")
        print(f"  Method: {result['extraction_method']}")
        print(f"  Est. time: {result['estimated_time']}")
        print()

    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}\n")

    native_extraction = [r for r in results if 'Native' in r['extraction_method']]
    ocr_required = [r for r in results if 'OCR' in r['extraction_method']]

    print(f"✅ Native Extraction (fast): {len(native_extraction)} documents")
    print(f"❌ OCR Required (slow): {len(ocr_required)} documents\n")

    if native_extraction:
        print("Native Extraction Documents:")
        for r in native_extraction:
            print(f"  • {r['file'][:70]}")
            print(f"    {r['total_pages']} pages, {r['avg_chars_per_page']:.0f} chars/page, ~{r['estimated_time']}")
        print()

    if ocr_required:
        print("OCR Required Documents:")
        total_ocr_time = 0
        for r in ocr_required:
            time_seconds = r['total_pages'] * 12
            total_ocr_time += time_seconds
            print(f"  • {r['file'][:70]}")
            print(f"    {r['total_pages']} pages, {r['avg_chars_per_page']:.0f} chars/page, ~{time_seconds/60:.1f} min")
        print(f"\n  Total OCR time estimate: ~{total_ocr_time/60:.1f} minutes ({total_ocr_time/3600:.1f} hours)")
        print()

    # Calculate totals
    total_pages = sum(r['total_pages'] for r in results)
    total_size_mb = sum(r['file_size_mb'] for r in results)

    print(f"Total: {len(results)} documents, {total_pages} pages, {total_size_mb:.1f} MB")
    print()

    # Recommendations
    print(f"{'='*100}")
    print("RECOMMENDATIONS")
    print(f"{'='*100}\n")

    if len(native_extraction) >= len(ocr_required):
        print("✅ GOOD NEWS: Most documents have native text!")
        print("   - Quick extraction with PyPDF2 for majority")
        print("   - Only a few documents need OCR")
    else:
        print("⚠️  Many documents require OCR")
        print(f"   - {len(ocr_required)} documents need OCR processing")
        print(f"   - Estimated time: {total_ocr_time/3600:.1f} hours")

    print("\nNext Steps:")
    print("1. Run native extraction on all documents (fast)")
    print("2. Validate semantic quality of native extractions")
    print("3. Apply OCR only to documents that need it")
    print("4. Apply post-processing to improve quality")
    print()

    # Save results
    import json
    output_file = project_root / 'data' / 'test_outputs' / 'pdf_assessment.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, indent=2, fp=f)
    print(f"✓ Detailed results saved to: {output_file}\n")


if __name__ == '__main__':
    main()
