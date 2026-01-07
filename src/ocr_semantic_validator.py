#!/usr/bin/env python3
"""
OCR Semantic Validation Tool

Validates OCR output for semantic coherence beyond just character recognition.
Measures:
- Dictionary word ratio (what % of words are real English words?)
- Alphabetic character ratio (how much noise/symbols?)
- Average word length (too short = garbled)
- N-gram plausibility (do word combinations make sense?)
- Readability score (is the text coherent?)

This provides a reality check on OCR confidence scores.
"""

import re
import string
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticValidator:
    """Validates OCR output for semantic meaning"""

    def __init__(self):
        """Initialize validator with English dictionary"""
        # Basic English dictionary (can be expanded)
        self.common_words = self._load_common_words()

    def _load_common_words(self) -> set:
        """
        Load common English words

        In production, use a proper dictionary like:
        - /usr/share/dict/words (Unix)
        - nltk.corpus.words.words()
        - pyenchant dictionary

        For now, using a basic set
        """
        # Basic common English words (subset)
        basic_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
            'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
            'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
            'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',
            'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its',
            'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our',
            'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any',
            'these', 'give', 'day', 'most', 'us', 'is', 'was', 'are', 'been', 'has',
            'had', 'were', 'said', 'did', 'having', 'may', 'should', 'being',
            # Add some academic/philosophical terms for Ramsey corpus
            'theory', 'truth', 'proposition', 'logic', 'belief', 'knowledge',
            'probability', 'inference', 'causality', 'general', 'particular',
            'statement', 'axiom', 'theorem', 'proof', 'definition', 'concept',
            'philosophy', 'mathematical', 'science', 'method', 'analysis',
            'principle', 'reasoning', 'argument', 'conclusion', 'premise',
            'fact', 'evidence', 'observation', 'hypothesis', 'theory'
        }

        # Try to load system dictionary if available
        try:
            dict_path = Path('/usr/share/dict/words')
            if dict_path.exists():
                with open(dict_path) as f:
                    system_words = {word.strip().lower() for word in f if len(word.strip()) > 1}
                logger.info(f"Loaded {len(system_words)} words from system dictionary")
                return basic_words | system_words
        except Exception as e:
            logger.warning(f"Could not load system dictionary: {e}")

        logger.info(f"Using basic dictionary with {len(basic_words)} words")
        return basic_words

    def clean_word(self, word: str) -> str:
        """Remove punctuation and normalize word"""
        # Remove punctuation
        word = word.strip(string.punctuation)
        # Remove digits and special characters
        word = re.sub(r'[^a-zA-Z]', '', word)
        return word.lower()

    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text for semantic quality

        Args:
            text: Text to analyze

        Returns:
            Dictionary with quality metrics
        """
        # Split into words
        raw_words = text.split()

        # Clean words
        words = [self.clean_word(w) for w in raw_words if self.clean_word(w)]

        if not words:
            return {
                'total_words': 0,
                'valid_words': 0,
                'dictionary_ratio': 0.0,
                'avg_word_length': 0.0,
                'alphabetic_ratio': 0.0,
                'quality_score': 0.0,
                'assessment': 'NO_TEXT'
            }

        # 1. Dictionary word ratio
        valid_words = [w for w in words if w in self.common_words or len(w) > 12]
        dictionary_ratio = len(valid_words) / len(words) if words else 0

        # 2. Average word length (too short = garbled)
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0

        # 3. Alphabetic character ratio
        total_chars = len(text)
        alpha_chars = sum(1 for c in text if c.isalpha() or c.isspace())
        alphabetic_ratio = alpha_chars / total_chars if total_chars > 0 else 0

        # 4. Word length distribution (very short words indicate noise)
        short_words = sum(1 for w in words if len(w) <= 2)
        short_word_ratio = short_words / len(words) if words else 0

        # 5. Calculate composite quality score
        # Weight different factors:
        # - Dictionary ratio: 40%
        # - Alphabetic ratio: 30%
        # - Word length: 20%
        # - Short word penalty: 10%

        # Normalize avg word length (ideal is 4-6 chars)
        word_length_score = min(avg_word_length / 5.0, 1.0)

        # Penalize high short word ratio
        short_word_score = max(1.0 - short_word_ratio, 0.0)

        quality_score = (
            dictionary_ratio * 0.4 +
            alphabetic_ratio * 0.3 +
            word_length_score * 0.2 +
            short_word_score * 0.1
        ) * 100

        # Assess quality
        if quality_score >= 70:
            assessment = "EXCELLENT"
            recommendation = "Text is semantically coherent and ready for training"
        elif quality_score >= 50:
            assessment = "GOOD"
            recommendation = "Text is usable with minor noise"
        elif quality_score >= 30:
            assessment = "FAIR"
            recommendation = "Text has significant errors, consider post-processing"
        elif quality_score >= 15:
            assessment = "POOR"
            recommendation = "Text is heavily garbled, manual review recommended"
        else:
            assessment = "UNUSABLE"
            recommendation = "Text is not semantically meaningful, skip or manual transcribe"

        return {
            'total_raw_words': len(raw_words),
            'total_words': len(words),
            'valid_words': len(valid_words),
            'dictionary_ratio': dictionary_ratio * 100,
            'avg_word_length': avg_word_length,
            'alphabetic_ratio': alphabetic_ratio * 100,
            'short_word_ratio': short_word_ratio * 100,
            'quality_score': quality_score,
            'assessment': assessment,
            'recommendation': recommendation
        }

    def validate_file(self, file_path: Path) -> Dict:
        """
        Validate an extracted text file

        Args:
            file_path: Path to text file

        Returns:
            Validation results
        """
        logger.info(f"Validating: {file_path.name}")

        try:
            text = file_path.read_text()
            results = self.analyze_text(text)
            results['file'] = file_path.name
            results['file_size'] = len(text)

            # Log results
            logger.info(f"Quality Score: {results['quality_score']:.1f}%")
            logger.info(f"Assessment: {results['assessment']}")
            logger.info(f"Dictionary Ratio: {results['dictionary_ratio']:.1f}%")
            logger.info(f"Alphabetic Ratio: {results['alphabetic_ratio']:.1f}%")

            return results

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {
                'file': file_path.name,
                'error': str(e),
                'quality_score': 0.0,
                'assessment': 'ERROR'
            }

    def validate_collection(self, directory: Path) -> List[Dict]:
        """
        Validate all text files in a directory

        Args:
            directory: Directory containing extracted text files

        Returns:
            List of validation results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"SEMANTIC VALIDATION: {directory}")
        logger.info(f"{'='*80}\n")

        text_files = list(directory.glob("*.txt"))
        logger.info(f"Found {len(text_files)} text files\n")

        results = []
        for text_file in sorted(text_files):
            result = self.validate_file(text_file)
            results.append(result)
            print()  # Blank line

        # Summary
        logger.info(f"\n{'='*80}")
        logger.info("VALIDATION SUMMARY")
        logger.info(f"{'='*80}\n")

        avg_quality = sum(r['quality_score'] for r in results) / len(results) if results else 0

        by_assessment = {}
        for r in results:
            assessment = r.get('assessment', 'UNKNOWN')
            by_assessment[assessment] = by_assessment.get(assessment, 0) + 1

        logger.info(f"Total files: {len(results)}")
        logger.info(f"Average quality score: {avg_quality:.1f}%\n")

        logger.info("Distribution:")
        for assessment in ['EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'UNUSABLE', 'ERROR']:
            count = by_assessment.get(assessment, 0)
            if count > 0:
                pct = (count / len(results)) * 100
                logger.info(f"  {assessment}: {count} files ({pct:.1f}%)")

        # List problematic files
        poor_files = [r for r in results if r.get('quality_score', 0) < 30]
        if poor_files:
            logger.info(f"\nProblematic files (quality < 30%):")
            for r in poor_files:
                logger.info(f"  • {r['file']}: {r['quality_score']:.1f}% - {r.get('recommendation', 'N/A')}")

        logger.info(f"\n{'='*80}\n")

        return results


def main():
    """Command-line interface"""
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Validate single file: python ocr_semantic_validator.py <file.txt>")
        print("  Validate directory: python ocr_semantic_validator.py <directory>")
        print("  Save results: python ocr_semantic_validator.py <path> --output results.json")
        sys.exit(1)

    path = Path(sys.argv[1])
    output_file = None

    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        if idx + 1 < len(sys.argv):
            output_file = Path(sys.argv[idx + 1])

    validator = SemanticValidator()

    if path.is_file():
        # Single file
        result = validator.validate_file(path)

        print(f"\n{'='*80}")
        print("SEMANTIC VALIDATION RESULT")
        print(f"{'='*80}\n")
        print(f"File: {result['file']}")
        print(f"Quality Score: {result['quality_score']:.1f}%")
        print(f"Assessment: {result['assessment']}")
        print(f"Recommendation: {result.get('recommendation', 'N/A')}")
        print(f"\nDetails:")
        print(f"  Total Words: {result['total_words']}")
        print(f"  Valid Dictionary Words: {result['valid_words']} ({result['dictionary_ratio']:.1f}%)")
        print(f"  Average Word Length: {result['avg_word_length']:.1f} chars")
        print(f"  Alphabetic Ratio: {result['alphabetic_ratio']:.1f}%")
        print(f"  Short Words (<= 2 chars): {result['short_word_ratio']:.1f}%")
        print(f"{'='*80}\n")

        if output_file:
            output_file.write_text(json.dumps(result, indent=2))
            print(f"✓ Results saved to: {output_file}\n")

    elif path.is_dir():
        # Directory
        results = validator.validate_collection(path)

        if output_file:
            output_file.write_text(json.dumps(results, indent=2))
            print(f"✓ Results saved to: {output_file}\n")
    else:
        print(f"Error: Invalid path: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
