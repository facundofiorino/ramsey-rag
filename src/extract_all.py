#!/usr/bin/env python3
"""
Unified Document Extraction Pipeline

Automatically detects document quality and routes to appropriate extraction method:
- High quality PDFs: Native text extraction (fast, accurate)
- Medium quality PDFs: OCR at 600 DPI with adaptive preprocessing
- Poor quality PDFs: OCR at 800 DPI with ultra preprocessing
- EPUB files: Native extraction with ebooklib

Supports parallel processing for faster extraction of large collections.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import PyPDF2
import pdfplumber
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup

# Import local modules
from ocr_detector import OCRRouter, TextDensityAnalyzer
from processors.ocr_processor import OCRProcessor, extract_with_ocr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextExtractor:
    """Native text extraction for high-quality documents"""

    @staticmethod
    def extract_pdf(pdf_path: Path) -> Dict:
        """
        Extract text from text-based PDF

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with extraction results
        """
        logger.info(f"Extracting text (native): {pdf_path.name}")
        start_time = time.time()

        try:
            full_text = ""
            page_results = []

            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n\n"
                        page_results.append({
                            'page': i,
                            'text': text,
                            'word_count': len(text.split()),
                            'char_count': len(text.strip())
                        })

            total_time = time.time() - start_time
            total_pages = len(page_results)
            total_words = len(full_text.split())
            total_chars = len(full_text.strip())

            logger.info(
                f"✓ Extracted {total_words} words from {total_pages} pages "
                f"in {total_time:.2f}s ({total_time/total_pages:.2f}s per page)"
            )

            return {
                'file': pdf_path.name,
                'path': str(pdf_path),
                'method': 'native_text',
                'total_pages': total_pages,
                'total_words': total_words,
                'total_chars': total_chars,
                'total_time': total_time,
                'time_per_page': total_time / total_pages if total_pages > 0 else 0,
                'page_results': page_results,
                'full_text': full_text,
                'success': True
            }

        except Exception as e:
            logger.error(f"Native extraction failed for {pdf_path.name}: {str(e)}")
            return {
                'file': pdf_path.name,
                'path': str(pdf_path),
                'method': 'native_text',
                'success': False,
                'error': str(e),
                'time': time.time() - start_time
            }

    @staticmethod
    def extract_epub(epub_path: Path) -> Dict:
        """
        Extract text from EPUB file

        Args:
            epub_path: Path to EPUB file

        Returns:
            Dictionary with extraction results
        """
        logger.info(f"Extracting text from EPUB: {epub_path.name}")
        start_time = time.time()

        try:
            book = epub.read_epub(str(epub_path))
            full_text = ""
            chapter_count = 0

            for item in book.get_items():
                if item.get_type() == ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text = soup.get_text()
                    if text.strip():
                        full_text += text + "\n\n"
                        chapter_count += 1

            total_time = time.time() - start_time
            total_words = len(full_text.split())
            total_chars = len(full_text.strip())

            logger.info(
                f"✓ Extracted {total_words} words from {chapter_count} chapters "
                f"in {total_time:.2f}s"
            )

            return {
                'file': epub_path.name,
                'path': str(epub_path),
                'method': 'epub',
                'total_chapters': chapter_count,
                'total_words': total_words,
                'total_chars': total_chars,
                'total_time': total_time,
                'full_text': full_text,
                'success': True
            }

        except Exception as e:
            logger.error(f"EPUB extraction failed for {epub_path.name}: {str(e)}")
            return {
                'file': epub_path.name,
                'path': str(epub_path),
                'method': 'epub',
                'success': False,
                'error': str(e),
                'time': time.time() - start_time
            }


class ExtractionPipeline:
    """Main extraction pipeline with automatic quality detection and routing"""

    def __init__(
        self,
        output_dir: Path,
        enable_ocr: bool = True,
        save_debug: bool = False,
        parallel: bool = False,
        max_workers: int = 4
    ):
        """
        Initialize extraction pipeline

        Args:
            output_dir: Directory to save extracted text
            enable_ocr: Enable OCR for scanned documents
            save_debug: Save debug images from OCR preprocessing
            parallel: Enable parallel processing
            max_workers: Maximum number of parallel workers
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.enable_ocr = enable_ocr
        self.save_debug = save_debug
        self.parallel = parallel
        self.max_workers = max_workers

        self.router = OCRRouter()
        self.text_extractor = TextExtractor()

        logger.info(f"Extraction Pipeline initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"OCR enabled: {self.enable_ocr}")
        logger.info(f"Parallel processing: {self.parallel} (workers: {self.max_workers})")

    def extract_document(self, file_path: Path) -> Dict:
        """
        Extract text from a single document with automatic method selection

        Args:
            file_path: Path to document file

        Returns:
            Dictionary with extraction results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {file_path.name}")
        logger.info(f"{'='*80}")

        # Handle EPUB files
        if file_path.suffix.lower() == '.epub':
            result = self.text_extractor.extract_epub(file_path)
            self._save_result(result)
            return result

        # Handle PDF files
        if file_path.suffix.lower() != '.pdf':
            logger.error(f"Unsupported file type: {file_path.suffix}")
            return {
                'file': file_path.name,
                'success': False,
                'error': f"Unsupported file type: {file_path.suffix}"
            }

        # Analyze PDF quality
        config = self.router.get_extraction_config(file_path)

        if not config.get('success', True):
            logger.error(f"Quality analysis failed: {config.get('error')}")
            return config

        # Extract based on quality
        if config['method'] == 'native_text':
            # High quality - use native extraction
            result = self.text_extractor.extract_pdf(file_path)

        elif config['method'] == 'ocr':
            # Medium or poor quality - use OCR
            if not self.enable_ocr:
                logger.warning("OCR disabled - skipping scanned document")
                return {
                    'file': file_path.name,
                    'success': False,
                    'error': 'OCR required but disabled'
                }

            quality_level = config['quality_level']
            logger.info(f"Quality: {quality_level} - Using OCR")

            result = extract_with_ocr(
                file_path,
                quality_level=quality_level,
                save_debug=self.save_debug
            )

        else:
            logger.error(f"Unknown extraction method: {config['method']}")
            return {
                'file': file_path.name,
                'success': False,
                'error': f"Unknown extraction method: {config['method']}"
            }

        # Save result
        self._save_result(result)
        return result

    def _save_result(self, result: Dict) -> None:
        """Save extraction result to output directory"""
        if not result.get('success', False):
            logger.warning(f"Skipping save for failed extraction: {result.get('file')}")
            return

        try:
            # Save text
            text_file = self.output_dir / f"{Path(result['file']).stem}.txt"
            text_file.write_text(result['full_text'])
            logger.info(f"✓ Saved text to: {text_file}")

            # Save metadata
            metadata = {k: v for k, v in result.items() if k != 'full_text'}
            metadata_file = self.output_dir / f"{Path(result['file']).stem}_metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2))
            logger.info(f"✓ Saved metadata to: {metadata_file}")

        except Exception as e:
            logger.error(f"Failed to save result: {str(e)}")

    def extract_collection(self, input_dir: Path) -> Dict:
        """
        Extract text from entire document collection

        Args:
            input_dir: Directory containing documents

        Returns:
            Dictionary with collection extraction results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"EXTRACTING DOCUMENT COLLECTION")
        logger.info(f"{'='*80}")
        logger.info(f"Input: {input_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"{'='*80}\n")

        start_time = time.time()

        # Find all supported files
        pdf_files = list(input_dir.glob("*.pdf"))
        epub_files = list(input_dir.glob("*.epub"))
        all_files = pdf_files + epub_files

        logger.info(f"Found {len(all_files)} files ({len(pdf_files)} PDFs, {len(epub_files)} EPUBs)\n")

        if not all_files:
            logger.warning("No files found to process")
            return {
                'success': False,
                'error': 'No files found'
            }

        # Analyze collection first
        if pdf_files:
            logger.info("Analyzing PDF collection...\n")
            analysis = self.router.analyze_collection(input_dir)
        else:
            analysis = None

        # Extract documents
        results = []

        if self.parallel and len(all_files) > 1:
            logger.info(f"Processing {len(all_files)} files in parallel (workers: {self.max_workers})...\n")
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.extract_document, f): f for f in all_files}

                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Extraction failed for {file_path.name}: {str(e)}")
                        results.append({
                            'file': file_path.name,
                            'success': False,
                            'error': str(e)
                        })
        else:
            logger.info(f"Processing {len(all_files)} files sequentially...\n")
            for file_path in all_files:
                result = self.extract_document(file_path)
                results.append(result)

        # Calculate summary statistics
        total_time = time.time() - start_time
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]

        total_words = sum(r.get('total_words', 0) for r in successful)
        total_chars = sum(r.get('total_chars', 0) for r in successful)
        total_pages = sum(r.get('total_pages', 0) for r in successful)

        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info("EXTRACTION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"\nFiles processed: {len(all_files)}")
        logger.info(f"  Successful: {len(successful)}")
        logger.info(f"  Failed: {len(failed)}")

        if total_pages > 0:
            logger.info(f"\nTotal pages: {total_pages}")

        logger.info(f"\nTotal text extracted:")
        logger.info(f"  Words: {total_words:,}")
        logger.info(f"  Characters: {total_chars:,}")

        logger.info(f"\nProcessing time:")
        logger.info(f"  Total: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        logger.info(f"  Per file: {total_time/len(all_files):.2f}s")

        if failed:
            logger.info(f"\nFailed files:")
            for result in failed:
                logger.info(f"  • {result['file']}: {result.get('error', 'Unknown error')}")

        logger.info(f"\n✓ Results saved to: {self.output_dir}")
        logger.info(f"{'='*80}\n")

        # Save collection summary
        summary = {
            'input_dir': str(input_dir),
            'output_dir': str(self.output_dir),
            'total_files': len(all_files),
            'successful_files': len(successful),
            'failed_files': len(failed),
            'total_pages': total_pages,
            'total_words': total_words,
            'total_chars': total_chars,
            'total_time': total_time,
            'time_per_file': total_time / len(all_files) if all_files else 0,
            'results': results
        }

        summary_file = self.output_dir / "extraction_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        logger.info(f"✓ Summary saved to: {summary_file}\n")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description='Extract text from document collection with automatic OCR detection'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('ramsey_data'),
        help='Input directory containing documents (default: ramsey_data)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/extracted'),
        help='Output directory for extracted text (default: data/extracted)'
    )
    parser.add_argument(
        '--enable-ocr',
        action='store_true',
        default=True,
        help='Enable OCR for scanned documents (default: True)'
    )
    parser.add_argument(
        '--disable-ocr',
        action='store_true',
        help='Disable OCR (skip scanned documents)'
    )
    parser.add_argument(
        '--save-debug',
        action='store_true',
        help='Save preprocessed images for debugging'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--single-file',
        type=Path,
        help='Process single file instead of directory'
    )

    args = parser.parse_args()

    # Handle OCR flags
    enable_ocr = args.enable_ocr and not args.disable_ocr

    # Create pipeline
    pipeline = ExtractionPipeline(
        output_dir=args.output,
        enable_ocr=enable_ocr,
        save_debug=args.save_debug,
        parallel=args.parallel,
        max_workers=args.max_workers
    )

    # Process single file or collection
    if args.single_file:
        if not args.single_file.exists():
            logger.error(f"File not found: {args.single_file}")
            sys.exit(1)

        result = pipeline.extract_document(args.single_file)

        if result.get('success', False):
            logger.info("\n✓ Extraction successful!")
            sys.exit(0)
        else:
            logger.error(f"\n❌ Extraction failed: {result.get('error')}")
            sys.exit(1)

    else:
        if not args.input.exists():
            logger.error(f"Directory not found: {args.input}")
            sys.exit(1)

        if not args.input.is_dir():
            logger.error(f"Not a directory: {args.input}")
            sys.exit(1)

        result = pipeline.extract_collection(args.input)

        if result.get('successful_files', 0) > 0:
            logger.info("\n✓ Extraction complete!")
            sys.exit(0)
        else:
            logger.error("\n❌ No files extracted successfully")
            sys.exit(1)


if __name__ == "__main__":
    main()
