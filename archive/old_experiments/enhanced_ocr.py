#!/usr/bin/env python3
"""
Enhanced OCR with advanced image preprocessing for old, degraded documents.
Optimized for 1920s-era manuscripts and poor quality scans.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
import time

def preprocess_image_basic(image):
    """Basic preprocessing - convert to grayscale"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    return gray

def preprocess_image_enhanced(image, method="adaptive"):
    """
    Enhanced preprocessing pipeline optimized for old, degraded documents

    Args:
        image: PIL Image
        method: "adaptive", "otsu", "aggressive", or "gentle"

    Returns:
        Preprocessed image as numpy array
    """
    # Convert to numpy array
    img_array = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    if method == "gentle":
        # Gentle preprocessing for relatively good scans
        # Slight denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )
        return binary

    elif method == "otsu":
        # Otsu's method - automatic threshold detection
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        # Otsu's thresholding
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    elif method == "aggressive":
        # Aggressive preprocessing for very poor quality documents

        # 1. Denoise heavily
        denoised = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)

        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 3. Morphological operations to remove noise
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

        # 4. Adaptive thresholding with larger block size
        binary = cv2.adaptiveThreshold(
            morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, 15
        )

        # 5. Dilation to strengthen text
        kernel2 = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(binary, kernel2, iterations=1)

        return dilated

    else:  # adaptive (default)
        # Standard adaptive preprocessing

        # 1. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # 2. Contrast enhancement with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 3. Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )

        return binary

def deskew_image(image):
    """Detect and correct skew in image"""
    coords = np.column_stack(np.where(image > 0))
    if len(coords) == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Only deskew if angle is significant (> 0.5 degrees)
    if abs(angle) > 0.5:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    return image

def test_enhanced_ocr(pdf_path, dpi=600, max_pages=3, preprocessing="all"):
    """
    Test enhanced OCR with multiple preprocessing methods

    Args:
        pdf_path: Path to PDF file
        dpi: DPI for image conversion (300, 400, 600)
        max_pages: Number of pages to test
        preprocessing: "all", "adaptive", "otsu", "aggressive", or "gentle"
    """
    print(f"\n{'='*80}")
    print(f"ENHANCED OCR TEST")
    print(f"{'='*80}")
    print(f"File: {Path(pdf_path).name}")
    print(f"DPI: {dpi}")
    print(f"Max pages: {max_pages}")
    print(f"Preprocessing: {preprocessing}")
    print(f"{'='*80}\n")

    try:
        # Convert PDF to images at high DPI
        print(f"Converting PDF to images at {dpi} DPI...")
        start_time = time.time()
        images = convert_from_path(pdf_path, first_page=1, last_page=max_pages, dpi=dpi)
        conversion_time = time.time() - start_time
        print(f"✓ Converted {len(images)} pages in {conversion_time:.2f} seconds\n")

        # Define preprocessing methods to test
        if preprocessing == "all":
            methods = ["basic", "gentle", "adaptive", "otsu", "aggressive"]
        else:
            methods = [preprocessing]

        results = {}

        for method in methods:
            print(f"\n{'='*80}")
            print(f"TESTING METHOD: {method.upper()}")
            print(f"{'='*80}\n")

            method_start = time.time()
            total_text = ""
            total_confidence = 0
            page_results = []

            for i, image in enumerate(images, 1):
                print(f"Processing page {i} with {method} preprocessing...")
                page_start = time.time()

                # Apply preprocessing
                if method == "basic":
                    preprocessed = preprocess_image_basic(image)
                else:
                    preprocessed = preprocess_image_enhanced(image, method=method)
                    # Apply deskewing for enhanced methods
                    preprocessed = deskew_image(preprocessed)

                # Convert back to PIL Image for Tesseract
                pil_image = Image.fromarray(preprocessed)

                # Configure Tesseract for best results with old documents
                custom_config = r'--oem 1 --psm 3'  # LSTM engine, auto page segmentation

                # Extract text with confidence
                ocr_data = pytesseract.image_to_data(
                    pil_image,
                    output_type=pytesseract.Output.DICT,
                    config=custom_config
                )

                # Calculate confidence
                confidences = [int(conf) for conf in ocr_data['conf'] if conf != '-1']
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0

                # Extract text
                page_text = pytesseract.image_to_string(pil_image, config=custom_config)

                page_time = time.time() - page_start

                page_results.append({
                    'page': i,
                    'text_length': len(page_text.strip()),
                    'word_count': len(page_text.split()),
                    'confidence': avg_confidence,
                    'time': page_time,
                    'text_preview': page_text[:300].strip()
                })

                total_text += page_text
                total_confidence += avg_confidence

                print(f"  ✓ Page {i}: {avg_confidence:.1f}% confidence, {len(page_text.split())} words, {page_time:.2f}s")

            method_time = time.time() - method_start
            avg_confidence = total_confidence / len(images) if images else 0

            results[method] = {
                'pages': len(images),
                'avg_confidence': avg_confidence,
                'total_time': method_time,
                'total_chars': len(total_text.strip()),
                'total_words': len(total_text.split()),
                'page_results': page_results,
                'sample_text': page_results[0]['text_preview'] if page_results else ""
            }

            print(f"\n  SUMMARY for {method}:")
            print(f"  Average confidence: {avg_confidence:.1f}%")
            print(f"  Total time: {method_time:.2f}s ({method_time/len(images):.2f}s per page)")
            print(f"  Total extracted: {len(total_text.split())} words, {len(total_text.strip())} characters")

        # Compare results
        print(f"\n{'='*80}")
        print("COMPARISON OF PREPROCESSING METHODS")
        print(f"{'='*80}\n")

        print(f"{'Method':<15} {'Confidence':<12} {'Words':<10} {'Time/Page':<12} {'Quality'}")
        print(f"{'-'*15} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")

        best_method = None
        best_confidence = 0

        for method, result in results.items():
            quality = "❌ POOR" if result['avg_confidence'] < 50 else \
                     "⚠️  MEDIUM" if result['avg_confidence'] < 70 else \
                     "✓ GOOD"

            time_per_page = result['total_time'] / result['pages']

            print(f"{method:<15} {result['avg_confidence']:>6.1f}%      {result['total_words']:>6}     {time_per_page:>6.2f}s      {quality}")

            if result['avg_confidence'] > best_confidence:
                best_confidence = result['avg_confidence']
                best_method = method

        print(f"\n{'='*80}")
        print(f"BEST METHOD: {best_method.upper()} ({best_confidence:.1f}% confidence)")
        print(f"{'='*80}")

        # Show sample text from best method
        print(f"\nSAMPLE TEXT (First 300 chars from page 1 using {best_method}):")
        print(f"{'-'*80}")
        print(results[best_method]['sample_text'])
        if len(results[best_method]['sample_text']) >= 300:
            print("...")
        print(f"{'-'*80}\n")

        # Recommendation
        print("\nRECOMMENDATION:")
        if best_confidence >= 70:
            print(f"  ✓ Use '{best_method}' preprocessing - High quality output")
            print(f"  ✓ Ready for training data extraction")
        elif best_confidence >= 50:
            print(f"  ⚠️  Use '{best_method}' preprocessing - Acceptable quality")
            print(f"  ⚠️  May benefit from manual review of sample pages")
        else:
            print(f"  ❌ Even with '{best_method}', quality is low ({best_confidence:.1f}%)")
            print(f"  ❌ Consider:")
            print(f"     - Trying even higher DPI (800-1200)")
            print(f"     - Manual transcription")
            print(f"     - Alternative OCR engines (EasyOCR, PaddleOCR)")
            print(f"     - Skipping this document if not critical")

        print(f"\n{'='*80}\n")

        return results

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test on the poor quality scanned PDF
    pdf_path = "ramsey_data/Ramsey, Frank - General Propositions and Causality - libgen.li.pdf"

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print("\n" + "="*80)
    print("ENHANCED OCR TESTING FOR OLD MANUSCRIPTS")
    print("Optimized for 1920s-era typewritten documents")
    print("="*80)

    # Test with multiple preprocessing methods at high DPI
    results = test_enhanced_ocr(
        pdf_path,
        dpi=600,  # High DPI for old documents
        max_pages=3,  # Test first 3 pages
        preprocessing="all"  # Test all methods
    )

    if results:
        print("\n✓ Testing complete! Check results above for best preprocessing method.")
    else:
        print("\n❌ Testing failed. Check error messages above.")
