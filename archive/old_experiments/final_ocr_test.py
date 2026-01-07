#!/usr/bin/env python3
"""
Final OCR test with optimized settings: 800 DPI + ultra preprocessing
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import time

def preprocess_optimized(image):
    """Optimized preprocessing based on testing"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Heavy denoising for old documents
    denoised = cv2.fastNlMeansDenoising(gray, h=20, templateWindowSize=7, searchWindowSize=25)

    # Aggressive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Sharpen to clarify text edges
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 71, 20
    )

    # Close gaps in characters
    kernel2 = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel2)

    return closed

def extract_with_best_settings(pdf_path, max_pages=3, save_images=False):
    """Extract text using optimized settings from testing"""
    print(f"\n{'='*80}")
    print("OPTIMIZED OCR EXTRACTION")
    print(f"{'='*80}")
    print(f"File: {Path(pdf_path).name}")
    print(f"Settings: 800 DPI + Ultra Preprocessing")
    print(f"Pages: {max_pages}")
    print(f"{'='*80}\n")

    try:
        # Convert at optimal DPI
        print("Converting PDF to images at 800 DPI...")
        start_time = time.time()
        images = convert_from_path(pdf_path, first_page=1, last_page=max_pages, dpi=800)
        conv_time = time.time() - start_time
        print(f"✓ Converted {len(images)} pages in {conv_time:.2f}s\n")

        total_text = ""
        total_confidence = 0
        page_results = []

        for i, image in enumerate(images, 1):
            print(f"Processing page {i}...")
            page_start = time.time()

            # Preprocess
            preprocessed = preprocess_optimized(image)
            pil_image = Image.fromarray(preprocessed)

            # Save preprocessed image for inspection
            if save_images:
                output_dir = Path("debug_images")
                output_dir.mkdir(exist_ok=True)
                pil_image.save(output_dir / f"page_{i}_preprocessed.png")

            # OCR with optimized config
            custom_config = r'--oem 1 --psm 6'  # LSTM, uniform text block

            ocr_data = pytesseract.image_to_data(
                pil_image,
                output_type=pytesseract.Output.DICT,
                config=custom_config
            )

            page_text = pytesseract.image_to_string(pil_image, config=custom_config)

            # Calculate confidence
            confidences = [int(conf) for conf in ocr_data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            page_time = time.time() - page_start

            page_results.append({
                'page': i,
                'confidence': avg_confidence,
                'words': len(page_text.split()),
                'chars': len(page_text.strip()),
                'time': page_time,
                'text': page_text
            })

            total_text += page_text + "\n\n"
            total_confidence += avg_confidence

            print(f"  ✓ Confidence: {avg_confidence:.1f}%")
            print(f"  ✓ Extracted: {len(page_text.split())} words, {len(page_text.strip())} chars")
            print(f"  ✓ Time: {page_time:.2f}s\n")

        # Summary
        total_time = time.time() - start_time
        avg_confidence = total_confidence / len(images) if images else 0

        print(f"{'='*80}")
        print("EXTRACTION SUMMARY")
        print(f"{'='*80}")
        print(f"Pages processed: {len(images)}")
        print(f"Average confidence: {avg_confidence:.1f}%")
        print(f"Total words: {len(total_text.split())}")
        print(f"Total characters: {len(total_text.strip())}")
        print(f"Total time: {total_time:.2f}s ({total_time/len(images):.2f}s per page)")

        # Quality assessment
        print(f"\nQUALITY ASSESSMENT:")
        if avg_confidence >= 50:
            print("  ✓ ACCEPTABLE - Suitable for training with minor review")
        elif avg_confidence >= 35:
            print("  ⚠️  MARGINAL - Usable but may benefit from manual correction")
        else:
            print("  ❌ POOR - Consider alternative approaches")

        # Show samples from each page
        print(f"\n{'='*80}")
        print("TEXT SAMPLES (first 200 chars per page)")
        print(f"{'='*80}\n")

        for result in page_results:
            print(f"Page {result['page']} ({result['confidence']:.1f}% confidence):")
            print(f"{'-'*80}")
            sample = result['text'][:200].strip()
            print(sample)
            if len(result['text']) > 200:
                print("...")
            print()

        # Save full text
        output_file = Path("extracted_text.txt")
        output_file.write_text(total_text)
        print(f"✓ Full extracted text saved to: {output_file}")

        print(f"\n{'='*80}\n")

        return {
            'success': True,
            'pages': len(images),
            'avg_confidence': avg_confidence,
            'total_words': len(total_text.split()),
            'total_chars': len(total_text.strip()),
            'page_results': page_results,
            'full_text': total_text
        }

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    pdf_path = "ramsey_data/Ramsey, Frank - General Propositions and Causality - libgen.li.pdf"

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print("\n" + "="*80)
    print("FINAL OCR TEST - OPTIMIZED SETTINGS")
    print("Based on extensive testing, using:")
    print("  • 800 DPI (best quality/speed tradeoff)")
    print("  • Ultra preprocessing (heavy denoising + CLAHE + sharpening)")
    print("  • Adaptive thresholding with large blocks")
    print("  • Tesseract LSTM engine with PSM 6")
    print("="*80)

    result = extract_with_best_settings(pdf_path, max_pages=3, save_images=True)

    if result['success']:
        print("\n✓ Extraction complete!")
        print(f"Achieved {result['avg_confidence']:.1f}% confidence")
        print(f"Extracted {result['total_words']} words from {result['pages']} pages")
    else:
        print("\n❌ Extraction failed")
