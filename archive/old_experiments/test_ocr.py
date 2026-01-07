#!/usr/bin/env python3
"""Test OCR on a scanned PDF from ramsey_data"""

import sys
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import time

def test_ocr_on_pdf(pdf_path, max_pages=2):
    """Test OCR extraction on a PDF file"""
    print(f"\n{'='*80}")
    print(f"Testing OCR on: {Path(pdf_path).name}")
    print(f"{'='*80}\n")

    try:
        start_time = time.time()

        # Convert PDF pages to images
        print(f"Converting first {max_pages} pages to images...")
        images = convert_from_path(pdf_path, first_page=1, last_page=max_pages, dpi=300)

        conversion_time = time.time() - start_time
        print(f"✓ Converted {len(images)} pages in {conversion_time:.2f} seconds")

        # Process each page with OCR
        total_text = ""
        total_confidence = 0
        page_results = []

        for i, image in enumerate(images, 1):
            print(f"\nProcessing page {i}...")
            ocr_start = time.time()

            # Extract text with confidence data
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            # Calculate page statistics
            confidences = [int(conf) for conf in ocr_data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Extract text
            page_text = pytesseract.image_to_string(image)

            ocr_time = time.time() - ocr_start

            page_results.append({
                'page': i,
                'text_length': len(page_text.strip()),
                'word_count': len(page_text.split()),
                'confidence': avg_confidence,
                'time': ocr_time,
                'text_preview': page_text[:200].strip()
            })

            total_text += page_text
            total_confidence += avg_confidence

            print(f"  ✓ Page {i} completed in {ocr_time:.2f}s")
            print(f"  ✓ Confidence: {avg_confidence:.1f}%")
            print(f"  ✓ Extracted {len(page_text.split())} words, {len(page_text.strip())} characters")

        # Overall statistics
        total_time = time.time() - start_time
        avg_confidence = total_confidence / len(images) if images else 0

        print(f"\n{'='*80}")
        print("OVERALL RESULTS:")
        print(f"{'='*80}")
        print(f"Total pages processed: {len(images)}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per page: {total_time/len(images):.2f} seconds")
        print(f"Average OCR confidence: {avg_confidence:.1f}%")
        print(f"Total characters extracted: {len(total_text.strip())}")
        print(f"Total words extracted: {len(total_text.split())}")

        # Quality assessment
        print(f"\nQUALITY ASSESSMENT:")
        if avg_confidence >= 70:
            print("  ✓ EXCELLENT - High confidence, ready for training")
        elif avg_confidence >= 50:
            print("  ⚠️  ACCEPTABLE - Medium confidence, may need review")
        else:
            print("  ❌ POOR - Low confidence, needs preprocessing or manual review")

        # Show sample text from first page
        print(f"\n{'='*80}")
        print("SAMPLE TEXT FROM PAGE 1:")
        print(f"{'='*80}")
        if page_results:
            print(page_results[0]['text_preview'])
            if len(page_results[0]['text_preview']) >= 200:
                print("...")

        print(f"\n{'='*80}\n")

        return {
            'success': True,
            'pages': len(images),
            'confidence': avg_confidence,
            'total_time': total_time,
            'page_results': page_results
        }

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    # Test on the two scanned PDFs identified during analysis
    scanned_pdfs = [
        "ramsey_data/Ramsey, Frank - General Propositions and Causality - libgen.li.pdf",
        "ramsey_data/Frank Ramsey _ Truth and Success -- Jérôme Dokic; Pascal Engel -- Taylor & Francis (Unlimited), London, 2003 -- London ; New York_ Routledge -- 9780203217832 -- fd21e06107c0fdb53d4016fd3c2f9116 -- Anna's Archive.pdf"
    ]

    print("\n" + "="*80)
    print("OCR TEST SUITE - RAMSEY DATA SCANNED PDFs")
    print("="*80)

    results = []
    for pdf_path in scanned_pdfs:
        if Path(pdf_path).exists():
            result = test_ocr_on_pdf(pdf_path, max_pages=2)
            results.append({
                'file': Path(pdf_path).name,
                'result': result
            })
        else:
            print(f"\n⚠️  File not found: {pdf_path}")

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for r in results:
        if r['result']['success']:
            print(f"\n✓ {r['file'][:70]}...")
            print(f"  Confidence: {r['result']['confidence']:.1f}%")
            print(f"  Processing time: {r['result']['total_time']:.2f}s")
            print(f"  Status: {'READY' if r['result']['confidence'] >= 70 else 'NEEDS REVIEW'}")
        else:
            print(f"\n❌ {r['file'][:70]}...")
            print(f"  Error: {r['result'].get('error', 'Unknown error')}")

    print("\n" + "="*80 + "\n")
