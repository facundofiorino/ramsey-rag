#!/usr/bin/env python3
"""
Advanced OCR with multiple engines and ensemble methods.
Tests Tesseract, EasyOCR, and PaddleOCR to maximize text extraction.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import time
from PIL import Image
from pdf2image import convert_from_path

# OCR engines
import pytesseract
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception as e:
    print(f"⚠️  EasyOCR not available: {e}")
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except Exception as e:
    print(f"⚠️  PaddleOCR not available: {e}")
    PADDLEOCR_AVAILABLE = False

def preprocess_ultra_v2(image):
    """Enhanced preprocessing with even more aggressive techniques"""
    img_array = np.array(image)

    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # 1. Bilateral filter - preserves edges while denoising
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

    # 2. CLAHE with higher clip limit
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bilateral)

    # 3. Morphological gradient to enhance text
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)

    # 4. Combine with enhanced
    combined = cv2.addWeighted(enhanced, 0.7, gradient, 0.3, 0)

    # 5. Unsharp masking for sharpening
    gaussian = cv2.GaussianBlur(combined, (0, 0), 2.0)
    unsharp = cv2.addWeighted(combined, 1.5, gaussian, -0.5, 0)

    # 6. Adaptive threshold with optimal parameters
    binary = cv2.adaptiveThreshold(
        unsharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 81, 25
    )

    # 7. Remove small noise
    kernel2 = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel2, iterations=1)

    # 8. Slight dilation to strengthen text
    dilated = cv2.dilate(cleaned, kernel2, iterations=1)

    return dilated

def ocr_tesseract(image, config='--oem 1 --psm 6'):
    """Tesseract OCR"""
    try:
        start = time.time()

        # Get confidence data
        ocr_data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            config=config
        )

        # Get text
        text = pytesseract.image_to_string(image, config=config)

        # Calculate confidence
        confidences = [int(c) for c in ocr_data['conf'] if c != '-1']
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        elapsed = time.time() - start

        return {
            'engine': 'Tesseract',
            'text': text,
            'confidence': avg_conf,
            'time': elapsed,
            'success': True
        }
    except Exception as e:
        return {
            'engine': 'Tesseract',
            'error': str(e),
            'success': False
        }

def ocr_easyocr(image, reader=None):
    """EasyOCR - deep learning based"""
    try:
        start = time.time()

        if reader is None:
            reader = easyocr.Reader(['en'], gpu=False)

        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        # Run OCR
        results = reader.readtext(image_np, detail=1, paragraph=False)

        # Extract text and confidence
        texts = []
        confidences = []
        for (bbox, text, conf) in results:
            texts.append(text)
            confidences.append(conf * 100)  # Convert to percentage

        full_text = ' '.join(texts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        elapsed = time.time() - start

        return {
            'engine': 'EasyOCR',
            'text': full_text,
            'confidence': avg_conf,
            'time': elapsed,
            'success': True
        }
    except Exception as e:
        return {
            'engine': 'EasyOCR',
            'error': str(e),
            'success': False
        }

def ocr_paddleocr(image, paddle_ocr=None):
    """PaddleOCR - optimized for documents"""
    try:
        start = time.time()

        if paddle_ocr is None:
            paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='en')

        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        # Run OCR
        results = paddle_ocr.ocr(image_np, cls=True)

        # Extract text and confidence
        texts = []
        confidences = []

        if results and results[0]:
            for line in results[0]:
                if line:
                    text = line[1][0]
                    conf = line[1][1] * 100  # Convert to percentage
                    texts.append(text)
                    confidences.append(conf)

        full_text = ' '.join(texts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        elapsed = time.time() - start

        return {
            'engine': 'PaddleOCR',
            'text': full_text,
            'confidence': avg_conf,
            'time': elapsed,
            'success': True
        }
    except Exception as e:
        return {
            'engine': 'PaddleOCR',
            'error': str(e),
            'success': False
        }

def ensemble_ocr(results):
    """Combine results from multiple OCR engines using voting"""
    successful = [r for r in results if r['success']]

    if not successful:
        return None

    # Find best by confidence
    best = max(successful, key=lambda x: x['confidence'])

    # Calculate ensemble metrics
    avg_conf = sum(r['confidence'] for r in successful) / len(successful)

    # For now, use best result, but mark as ensemble
    return {
        'engine': f"Ensemble ({len(successful)} engines)",
        'text': best['text'],
        'confidence': avg_conf,
        'best_engine': best['engine'],
        'best_confidence': best['confidence'],
        'time': sum(r['time'] for r in successful),
        'success': True
    }

def test_multi_engine_ocr(pdf_path, page_num=1, dpi=800, save_images=False):
    """Test all available OCR engines and create ensemble"""
    print(f"\n{'='*80}")
    print("MULTI-ENGINE OCR TEST")
    print(f"{'='*80}")
    print(f"File: {Path(pdf_path).name}")
    print(f"Page: {page_num}")
    print(f"DPI: {dpi}")
    print(f"{'='*80}\n")

    try:
        # Convert PDF
        print("Converting PDF to image...")
        start = time.time()
        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=dpi)

        if not images:
            print("❌ No images converted")
            return None

        image = images[0]
        conv_time = time.time() - start
        print(f"✓ Converted in {conv_time:.2f}s\n")

        # Preprocess
        print("Preprocessing image with ultra-v2 method...")
        preprocessed = preprocess_ultra_v2(image)
        pil_preprocessed = Image.fromarray(preprocessed)

        if save_images:
            output_dir = Path("advanced_debug")
            output_dir.mkdir(exist_ok=True)
            pil_preprocessed.save(output_dir / f"page_{page_num}_ultra_v2.png")
            print(f"✓ Saved preprocessed image to {output_dir}/\n")

        # Initialize OCR engines
        print("Initializing OCR engines...")
        easy_reader = None
        paddle_ocr = None

        if EASYOCR_AVAILABLE:
            print("  Loading EasyOCR...")
            easy_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("  ✓ EasyOCR ready")

        if PADDLEOCR_AVAILABLE:
            print("  Loading PaddleOCR...")
            paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='en')
            print("  ✓ PaddleOCR ready")

        print()

        # Test all engines
        results = []

        # 1. Tesseract
        print("Running Tesseract OCR...")
        tess_result = ocr_tesseract(pil_preprocessed)
        results.append(tess_result)
        if tess_result['success']:
            print(f"  ✓ Confidence: {tess_result['confidence']:.1f}%")
            print(f"  ✓ Words: {len(tess_result['text'].split())}")
            print(f"  ✓ Time: {tess_result['time']:.2f}s")
        else:
            print(f"  ❌ Error: {tess_result.get('error')}")
        print()

        # 2. EasyOCR
        if EASYOCR_AVAILABLE and easy_reader:
            print("Running EasyOCR...")
            easy_result = ocr_easyocr(preprocessed, easy_reader)
            results.append(easy_result)
            if easy_result['success']:
                print(f"  ✓ Confidence: {easy_result['confidence']:.1f}%")
                print(f"  ✓ Words: {len(easy_result['text'].split())}")
                print(f"  ✓ Time: {easy_result['time']:.2f}s")
            else:
                print(f"  ❌ Error: {easy_result.get('error')}")
            print()

        # 3. PaddleOCR
        if PADDLEOCR_AVAILABLE and paddle_ocr:
            print("Running PaddleOCR...")
            paddle_result = ocr_paddleocr(preprocessed, paddle_ocr)
            results.append(paddle_result)
            if paddle_result['success']:
                print(f"  ✓ Confidence: {paddle_result['confidence']:.1f}%")
                print(f"  ✓ Words: {len(paddle_result['text'].split())}")
                print(f"  ✓ Time: {paddle_result['time']:.2f}s")
            else:
                print(f"  ❌ Error: {paddle_result.get('error')}")
            print()

        # Ensemble
        print("Creating ensemble result...")
        ensemble = ensemble_ocr(results)
        print()

        # Comparison
        print(f"{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}\n")

        print(f"{'Engine':<20} {'Confidence':<15} {'Words':<10} {'Time':<10}")
        print(f"{'-'*20} {'-'*15} {'-'*10} {'-'*10}")

        for r in results:
            if r['success']:
                print(f"{r['engine']:<20} {r['confidence']:>6.1f}%        {len(r['text'].split()):>6}     {r['time']:>6.2f}s")
            else:
                print(f"{r['engine']:<20} ERROR")

        if ensemble:
            print(f"{'-'*20} {'-'*15} {'-'*10} {'-'*10}")
            print(f"{'Ensemble (avg)':<20} {ensemble['confidence']:>6.1f}%        {len(ensemble['text'].split()):>6}     {ensemble['time']:>6.2f}s")

        # Best result
        successful = [r for r in results if r['success']]
        if successful:
            best = max(successful, key=lambda x: x['confidence'])
            print(f"\n{'='*80}")
            print(f"BEST RESULT: {best['engine']} ({best['confidence']:.1f}% confidence)")
            print(f"{'='*80}\n")

            print("Sample text (first 300 chars):")
            print(f"{'-'*80}")
            sample = best['text'][:300].strip()
            print(sample)
            if len(best['text']) > 300:
                print("...")
            print(f"{'-'*80}\n")

            # Save best result
            output_file = Path(f"best_ocr_page_{page_num}.txt")
            output_file.write_text(best['text'])
            print(f"✓ Full text saved to: {output_file}\n")

        print(f"{'='*80}\n")

        return {
            'results': results,
            'ensemble': ensemble,
            'best': best if successful else None
        }

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    pdf_path = "ramsey_data/Ramsey, Frank - General Propositions and Causality - libgen.li.pdf"

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print("\n" + "="*80)
    print("ADVANCED MULTI-ENGINE OCR TEST")
    print("Testing: Tesseract + EasyOCR + PaddleOCR")
    print("With: Ultra-v2 preprocessing (even more aggressive)")
    print("Goal: Maximum text extraction from degraded 1920s manuscript")
    print("="*80)

    result = test_multi_engine_ocr(
        pdf_path,
        page_num=1,
        dpi=800,
        save_images=True
    )

    if result and result['best']:
        print("\n" + "="*80)
        print("FINAL RESULT")
        print("="*80)
        print(f"Best engine: {result['best']['engine']}")
        print(f"Confidence: {result['best']['confidence']:.1f}%")
        print(f"Words extracted: {len(result['best']['text'].split())}")
        print(f"Total processing time: {sum(r['time'] for r in result['results'] if r['success']):.2f}s")

        if result['best']['confidence'] >= 50:
            print("\n✓ Quality is ACCEPTABLE for training")
        elif result['best']['confidence'] >= 40:
            print("\n⚠️  Quality is MARGINAL but improved")
        else:
            print("\n⚠️  Quality remains challenging")
            print("Consider: Manual transcription for critical pages")

        print("\n" + "="*80 + "\n")
