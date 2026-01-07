#!/usr/bin/env python3
"""
OCR Need Detection Module

Automatically detects whether a PDF requires OCR based on text density analysis.
Routes documents to appropriate extraction method:
- High text density (>100 chars/page): Native text extraction
- Medium text density (50-100 chars/page): OCR at 600 DPI
- Low text density (<50 chars/page): OCR at 800 DPI with ultra preprocessing
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import PyPDF2
import pdfplumber

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDensityAnalyzer:
    """Analyzes PDF text density to determine if OCR is needed"""

    # Text density thresholds (chars per page)
    HIGH_QUALITY_THRESHOLD = 100  # Text-based PDF
    MEDIUM_QUALITY_THRESHOLD = 50  # Medium quality scan
    # Below MEDIUM_QUALITY_THRESHOLD = Poor quality scan

    @staticmethod
    def analyze_pdf(pdf_path: Path, sample_pages: int = 3) -> Dict:
        """
        Analyze PDF text extractability

        Args:
            pdf_path: Path to PDF file
            sample_pages: Number of pages to sample (default 3)

        Returns:
            Dictionary with analysis results:
            {
                'file': str,
                'total_pages': int,
                'sample_pages': int,
                'sample_chars': int,
                'avg_chars_per_page': float,
                'quality_level': str,  # 'high', 'medium', or 'poor'
                'needs_ocr': bool,
                'recommended_dpi': int,
                'recommended_preprocessing': str,
                'extraction_method': str
            }
        """
        logger.info(f"Analyzing PDF: {pdf_path.name}")

        try:
            # Get total page count
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                total_pages = len(pdf_reader.pages)

            logger.info(f"Total pages: {total_pages}")

            # Sample first N pages to check text extractability
            sample_pages = min(sample_pages, total_pages)
            sample_text = ""

            with pdfplumber.open(pdf_path) as pdf:
                for page_num in range(sample_pages):
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    if text:
                        sample_text += text

            # Calculate text density
            text_chars = len(sample_text.strip())
            avg_chars_per_page = text_chars / sample_pages if sample_pages > 0 else 0

            logger.info(f"Sample: {sample_pages} pages, {text_chars} chars")
            logger.info(f"Average: {avg_chars_per_page:.1f} chars/page")

            # Determine quality level and extraction method
            if avg_chars_per_page >= TextDensityAnalyzer.HIGH_QUALITY_THRESHOLD:
                quality_level = "high"
                needs_ocr = False
                recommended_dpi = None
                recommended_preprocessing = None
                extraction_method = "native_text"
                logger.info("✓ High quality: Use native text extraction")

            elif avg_chars_per_page >= TextDensityAnalyzer.MEDIUM_QUALITY_THRESHOLD:
                quality_level = "medium"
                needs_ocr = True
                recommended_dpi = 600
                recommended_preprocessing = "adaptive"
                extraction_method = "ocr"
                logger.info("⚠️  Medium quality: Use OCR at 600 DPI")

            else:
                quality_level = "poor"
                needs_ocr = True
                recommended_dpi = 800
                recommended_preprocessing = "ultra"
                extraction_method = "ocr"
                logger.info("⚠️  Poor quality: Use OCR at 800 DPI with ultra preprocessing")

            return {
                'file': pdf_path.name,
                'path': str(pdf_path),
                'total_pages': total_pages,
                'sample_pages': sample_pages,
                'sample_chars': text_chars,
                'avg_chars_per_page': avg_chars_per_page,
                'quality_level': quality_level,
                'needs_ocr': needs_ocr,
                'recommended_dpi': recommended_dpi,
                'recommended_preprocessing': recommended_preprocessing,
                'extraction_method': extraction_method,
                'success': True
            }

        except Exception as e:
            logger.error(f"Analysis failed for {pdf_path.name}: {str(e)}")
            return {
                'file': pdf_path.name,
                'path': str(pdf_path),
                'success': False,
                'error': str(e)
            }


class OCRRouter:
    """Routes documents to appropriate extraction method based on quality analysis"""

    def __init__(self):
        self.analyzer = TextDensityAnalyzer()

    def analyze_collection(self, directory: Path) -> Dict:
        """
        Analyze entire document collection

        Args:
            directory: Path to directory containing PDFs

        Returns:
            Dictionary with collection analysis:
            {
                'total_files': int,
                'high_quality': List[Dict],  # Files for native extraction
                'medium_quality': List[Dict],  # Files for 600 DPI OCR
                'poor_quality': List[Dict],  # Files for 800 DPI OCR
                'failed': List[Dict],  # Files that failed analysis
                'summary': Dict  # Summary statistics
            }
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"ANALYZING DOCUMENT COLLECTION: {directory}")
        logger.info(f"{'='*80}\n")

        pdf_files = list(directory.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files\n")

        high_quality = []
        medium_quality = []
        poor_quality = []
        failed = []

        for pdf_file in pdf_files:
            result = self.analyzer.analyze_pdf(pdf_file)

            if not result['success']:
                failed.append(result)
                continue

            if result['quality_level'] == 'high':
                high_quality.append(result)
            elif result['quality_level'] == 'medium':
                medium_quality.append(result)
            else:  # poor
                poor_quality.append(result)

            print()  # Blank line between files

        # Calculate summary statistics
        total_pages = sum(r['total_pages'] for r in high_quality + medium_quality + poor_quality)
        high_pages = sum(r['total_pages'] for r in high_quality)
        medium_pages = sum(r['total_pages'] for r in medium_quality)
        poor_pages = sum(r['total_pages'] for r in poor_quality)

        summary = {
            'total_files': len(pdf_files),
            'total_pages': total_pages,
            'high_quality_files': len(high_quality),
            'high_quality_pages': high_pages,
            'medium_quality_files': len(medium_quality),
            'medium_quality_pages': medium_pages,
            'poor_quality_files': len(poor_quality),
            'poor_quality_pages': poor_pages,
            'failed_files': len(failed)
        }

        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info("COLLECTION ANALYSIS SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"\nTotal files: {summary['total_files']}")
        logger.info(f"Total pages: {summary['total_pages']}\n")

        logger.info(f"High quality (native text extraction):")
        logger.info(f"  Files: {summary['high_quality_files']} ({summary['high_quality_files']/summary['total_files']*100:.1f}%)")
        logger.info(f"  Pages: {summary['high_quality_pages']} ({summary['high_quality_pages']/summary['total_pages']*100:.1f}%)")
        for doc in high_quality:
            logger.info(f"    • {doc['file']} ({doc['total_pages']} pages, {doc['avg_chars_per_page']:.1f} chars/page)")

        logger.info(f"\nMedium quality (OCR at 600 DPI):")
        logger.info(f"  Files: {summary['medium_quality_files']} ({summary['medium_quality_files']/summary['total_files']*100:.1f}%)")
        logger.info(f"  Pages: {summary['medium_quality_pages']} ({summary['medium_quality_pages']/summary['total_pages']*100:.1f}%)")
        for doc in medium_quality:
            logger.info(f"    • {doc['file']} ({doc['total_pages']} pages, {doc['avg_chars_per_page']:.1f} chars/page)")

        logger.info(f"\nPoor quality (OCR at 800 DPI + ultra preprocessing):")
        logger.info(f"  Files: {summary['poor_quality_files']} ({summary['poor_quality_files']/summary['total_files']*100:.1f}%)")
        logger.info(f"  Pages: {summary['poor_quality_pages']} ({summary['poor_quality_pages']/summary['total_pages']*100:.1f}%)")
        for doc in poor_quality:
            logger.info(f"    • {doc['file']} ({doc['total_pages']} pages, {doc['avg_chars_per_page']:.1f} chars/page)")

        if failed:
            logger.info(f"\nFailed analysis:")
            logger.info(f"  Files: {summary['failed_files']}")
            for doc in failed:
                logger.info(f"    • {doc['file']}: {doc.get('error', 'Unknown error')}")

        # Processing time estimates
        logger.info(f"\n{'='*80}")
        logger.info("ESTIMATED PROCESSING TIME")
        logger.info(f"{'='*80}")

        native_time = high_pages * 0.1  # ~0.1s per page for native extraction
        medium_time = medium_pages * 3.0  # ~3s per page at 600 DPI
        poor_time = poor_pages * 5.0  # ~5s per page at 800 DPI

        total_time = native_time + medium_time + poor_time

        logger.info(f"Native text extraction: ~{native_time:.1f}s ({high_pages} pages)")
        logger.info(f"Medium quality OCR: ~{medium_time:.1f}s ({medium_pages} pages)")
        logger.info(f"Poor quality OCR: ~{poor_time:.1f}s ({poor_pages} pages)")
        logger.info(f"\nTotal estimated time: ~{total_time:.1f}s ({total_time/60:.1f} minutes)")
        logger.info(f"{'='*80}\n")

        return {
            'high_quality': high_quality,
            'medium_quality': medium_quality,
            'poor_quality': poor_quality,
            'failed': failed,
            'summary': summary
        }

    def get_extraction_config(self, pdf_path: Path) -> Dict:
        """
        Get extraction configuration for a single PDF

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with extraction configuration
        """
        result = self.analyzer.analyze_pdf(pdf_path)

        if not result['success']:
            return result

        config = {
            'file': result['file'],
            'path': result['path'],
            'method': result['extraction_method'],
            'quality_level': result['quality_level']
        }

        if result['needs_ocr']:
            config['ocr_config'] = {
                'dpi': result['recommended_dpi'],
                'preprocessing': result['recommended_preprocessing']
            }

        return config


def detect_ocr_need(pdf_path: Path) -> bool:
    """
    Simple function to check if PDF needs OCR

    Args:
        pdf_path: Path to PDF file

    Returns:
        True if OCR is needed, False otherwise
    """
    analyzer = TextDensityAnalyzer()
    result = analyzer.analyze_pdf(pdf_path)
    return result.get('needs_ocr', True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Analyze single file: python ocr_detector.py <pdf_path>")
        print("  Analyze directory: python ocr_detector.py <directory_path>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if not path.exists():
        print(f"Error: Path not found: {path}")
        sys.exit(1)

    router = OCRRouter()

    if path.is_file():
        # Analyze single file
        print(f"\nAnalyzing single file: {path}\n")
        config = router.get_extraction_config(path)

        if config.get('success', True):
            print(f"\n{'='*80}")
            print("EXTRACTION CONFIGURATION")
            print(f"{'='*80}")
            print(f"File: {config['file']}")
            print(f"Method: {config['method']}")
            print(f"Quality: {config['quality_level']}")

            if 'ocr_config' in config:
                print(f"\nOCR Settings:")
                print(f"  DPI: {config['ocr_config']['dpi']}")
                print(f"  Preprocessing: {config['ocr_config']['preprocessing']}")

            print(f"{'='*80}\n")
        else:
            print(f"\n❌ Analysis failed: {config.get('error', 'Unknown error')}")

    elif path.is_dir():
        # Analyze directory
        results = router.analyze_collection(path)

    else:
        print(f"Error: Invalid path type: {path}")
        sys.exit(1)
