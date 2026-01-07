#!/usr/bin/env python3
"""
Ultra-high resolution OCR test for extremely degraded documents.
Tests with 800-1200 DPI for maximum detail extraction.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import time

def preprocess_ultra(image):
    """Ultra preprocessing optimized for very poor scans"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Heavy denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=20, templateWindowSize=7, searchWindowSize=25)

    # Aggressive CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # Adaptive thresholding with large block
    binary = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 71, 20
    )

    # Morphological closing to connect broken characters
    kernel2 = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel2)

    return closed

def test_ultra_dpi(pdf_path, page_num=1):
    """Test multiple DPI settings"""
    print(f"\n{'='*80}")
    print("ULTRA-HIGH DPI OCR TEST")
    print(f"{'='*80}\n")

    dpis = [600, 800, 1200]
    results = {}

    for dpi in dpis:
        print(f"\nTesting DPI: {dpi}")
        print(f"{'-'*80}")

        try:
            start = time.time()

            # Convert
            print(f"  Converting page {page_num} at {dpi} DPI...")
            images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=dpi)

            if not images:
                print(f"  ❌ No images converted")
                continue

            image = images[0]
            conv_time = time.time() - start
            print(f"  ✓ Converted in {conv_time:.2f}s")

            # Preprocess
            print(f"  Preprocessing...")
            preprocessed = preprocess_ultra(image)
            pil_image = Image.fromarray(preprocessed)

            # OCR
            print(f"  Running OCR...")
            ocr_start = time.time()

            custom_config = r'--oem 1 --psm 6'  # Assume uniform text block

            ocr_data = pytesseract.image_to_data(
                pil_image,
                output_type=pytesseract.Output.DICT,
                config=custom_config
            )

            text = pytesseract.image_to_string(pil_image, config=custom_config)

            ocr_time = time.time() - ocr_start

            # Analyze
            confidences = [int(conf) for conf in ocr_data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            total_time = time.time() - start

            results[dpi] = {
                'confidence': avg_confidence,
                'words': len(text.split()),
                'chars': len(text.strip()),
                'time': total_time,
                'text': text[:400]
            }

            print(f"  ✓ OCR complete in {ocr_time:.2f}s")
            print(f"  ✓ Confidence: {avg_confidence:.1f}%")
            print(f"  ✓ Extracted: {len(text.split())} words")
            print(f"  ✓ Total time: {total_time:.2f}s")

        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results[dpi] = {'error': str(e)}

    # Compare
    print(f"\n{'='*80}")
    print("DPI COMPARISON")
    print(f"{'='*80}\n")

    print(f"{'DPI':<8} {'Confidence':<12} {'Words':<10} {'Time':<10} {'Status'}")
    print(f"{'-'*8} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    best_dpi = None
    best_conf = 0

    for dpi, res in results.items():
        if 'error' in res:
            print(f"{dpi:<8} ERROR: {res['error']}")
        else:
            status = "✓ Best" if res['confidence'] > best_conf else ""
            print(f"{dpi:<8} {res['confidence']:>6.1f}%      {res['words']:>6}     {res['time']:>6.2f}s   {status}")
            if res['confidence'] > best_conf:
                best_conf = res['confidence']
                best_dpi = dpi

    if best_dpi:
        print(f"\n{'='*80}")
        print(f"BEST RESULT: {best_dpi} DPI ({best_conf:.1f}% confidence)")
        print(f"{'='*80}")
        print(f"\nSample text (first 400 chars):")
        print(f"{'-'*80}")
        print(results[best_dpi]['text'])
        print(f"{'-'*80}\n")

        # Final assessment
        if best_conf >= 50:
            print("✓ Quality is acceptable for extraction")
        elif best_conf >= 35:
            print("⚠️  Quality is marginal - manual review recommended")
        else:
            print("❌ Quality remains poor even at highest DPI")
            print("   This document may require:")
            print("   - Professional scanning/restoration")
            print("   - Manual transcription")
            print("   - Alternative OCR engine (EasyOCR, PaddleOCR)")

    print(f"\n{'='*80}\n")

    return results

if __name__ == "__main__":
    pdf_path = "ramsey_data/Ramsey, Frank - General Propositions and Causality - libgen.li.pdf"

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print("Testing page 1 at multiple DPI settings...")
    print("This may take several minutes for high DPI...\n")

    results = test_ultra_dpi(pdf_path, page_num=1)
