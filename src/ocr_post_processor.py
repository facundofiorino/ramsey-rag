#!/usr/bin/env python3
"""
OCR Post-Processing with Spell Correction and Error Patterns

Improves OCR output by:
1. Fixing common OCR character errors (rn→m, cl→d, etc.)
2. Spell-checking and auto-correction
3. Context-aware word replacement
4. Preserving proper nouns and technical terms
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    print("⚠️  Install pyspellchecker: pip install pyspellchecker")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRPostProcessor:
    """Post-process OCR output to fix common errors"""

    # Common OCR character substitution errors
    OCR_ERROR_PATTERNS = {
        # Letter pairs that look similar
        r'\brn\b': 'm',  # 'rn' often misread as 'm'
        r'\bvv\b': 'w',  # 'vv' often misread as 'w'
        r'\bcl\b': 'd',  # 'cl' often misread as 'd'
        r'\bII\b': 'll', # 'II' (capital i) often should be 'll'
        r'\b0\b': 'o',   # '0' (zero) in middle of word should be 'o'
        r'\b1\b': 'l',   # '1' (one) in middle of word should be 'l'

        # Common word-level errors
        r'\btlie\b': 'the',
        r'\btbe\b': 'the',
        r'\bwhicb\b': 'which',
        r'\bwbich\b': 'which',
        r'\bwitb\b': 'with',
        r'\btbat\b': 'that',
        r'\btbis\b': 'this',
        r'\bfrom\b': 'from',
        r'\bof\b': 'of',
        r'\band\b': 'and',
    }

    # Preserve these words (proper nouns, technical terms)
    PRESERVE_WORDS = {
        'ramsey', 'frank', 'cambridge', 'wittgenstein', 'russell',
        'philosophy', 'logic', 'theorem', 'axiom', 'proposition',
        'truth', 'probability', 'causality', 'inference'
    }

    def __init__(self, use_spellcheck: bool = True, preserve_case: bool = True):
        """
        Initialize post-processor

        Args:
            use_spellcheck: Enable spell-checking (requires pyspellchecker)
            preserve_case: Preserve original case when correcting
        """
        self.use_spellcheck = use_spellcheck and SPELLCHECKER_AVAILABLE
        self.preserve_case = preserve_case

        if self.use_spellcheck:
            self.spell = SpellChecker()
            # Add custom words to dictionary
            self.spell.word_frequency.load_words(self.PRESERVE_WORDS)
            logger.info("Spell checker initialized")
        else:
            self.spell = None
            if use_spellcheck:
                logger.warning("Spell checker requested but pyspellchecker not available")

    def fix_ocr_patterns(self, text: str) -> str:
        """
        Fix common OCR character substitution errors

        Args:
            text: Text to fix

        Returns:
            Text with pattern fixes applied
        """
        fixed = text

        for pattern, replacement in self.OCR_ERROR_PATTERNS.items():
            fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)

        return fixed

    def spell_check_word(self, word: str) -> str:
        """
        Spell-check a single word

        Args:
            word: Word to check

        Returns:
            Corrected word or original if no correction found
        """
        if not self.spell:
            return word

        # Skip if word is in preserve list
        if word.lower() in self.PRESERVE_WORDS:
            return word

        # Skip very short words (often correct or abbreviations)
        if len(word) < 3:
            return word

        # Skip words with numbers
        if any(c.isdigit() for c in word):
            return word

        # Skip words that are all uppercase (likely acronyms)
        if word.isupper() and len(word) > 1:
            return word

        # Check spelling
        word_lower = word.lower()

        # If word is correct, return original
        if word_lower in self.spell:
            return word

        # Get correction
        correction = self.spell.correction(word_lower)

        if correction and correction != word_lower:
            # Preserve original case
            if self.preserve_case:
                if word.isupper():
                    return correction.upper()
                elif word[0].isupper():
                    return correction.capitalize()
            return correction

        return word

    def process_text(self, text: str, apply_patterns: bool = True,
                     apply_spellcheck: bool = True) -> Dict:
        """
        Post-process OCR text

        Args:
            text: OCR output text
            apply_patterns: Apply OCR error pattern fixes
            apply_spellcheck: Apply spell-checking

        Returns:
            Dictionary with processed text and statistics
        """
        original_text = text
        processed_text = text
        corrections_made = 0

        # Step 1: Fix OCR patterns
        if apply_patterns:
            processed_text = self.fix_ocr_patterns(processed_text)
            pattern_changes = sum(1 for a, b in zip(original_text.split(),
                                                     processed_text.split()) if a != b)
            corrections_made += pattern_changes
            logger.info(f"Pattern fixes: {pattern_changes} changes")

        # Step 2: Spell-check
        if apply_spellcheck and self.spell:
            words = processed_text.split()
            corrected_words = []
            spell_corrections = 0

            for word in words:
                # Remove punctuation for checking
                clean_word = word.strip('.,;:!?"\'()[]{}')
                if clean_word:
                    corrected = self.spell_check_word(clean_word)
                    if corrected != clean_word:
                        spell_corrections += 1
                    # Restore punctuation
                    word = word.replace(clean_word, corrected)
                corrected_words.append(word)

            processed_text = ' '.join(corrected_words)
            corrections_made += spell_corrections
            logger.info(f"Spell corrections: {spell_corrections} changes")

        # Calculate improvement
        original_words = len(original_text.split())
        improvement_pct = (corrections_made / original_words * 100) if original_words > 0 else 0

        return {
            'original_text': original_text,
            'processed_text': processed_text,
            'corrections_made': corrections_made,
            'total_words': original_words,
            'improvement_percentage': improvement_pct,
            'pattern_fixes_applied': apply_patterns,
            'spellcheck_applied': apply_spellcheck and self.spell is not None
        }

    def process_file(self, input_path: Path, output_path: Optional[Path] = None) -> Dict:
        """
        Process an OCR output file

        Args:
            input_path: Path to OCR text file
            output_path: Path to save processed text (optional)

        Returns:
            Processing results
        """
        logger.info(f"Processing: {input_path.name}")

        text = input_path.read_text()
        result = self.process_text(text)

        if output_path:
            output_path.write_text(result['processed_text'])
            logger.info(f"✓ Saved to: {output_path}")

        logger.info(f"✓ Made {result['corrections_made']} corrections "
                   f"({result['improvement_percentage']:.1f}% of words)")

        return result


def main():
    """Command-line interface"""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Post-process OCR output')
    parser.add_argument('input', type=Path, help='Input OCR text file')
    parser.add_argument('--output', '-o', type=Path, help='Output file (optional)')
    parser.add_argument('--no-patterns', action='store_true',
                       help='Disable OCR pattern fixes')
    parser.add_argument('--no-spellcheck', action='store_true',
                       help='Disable spell-checking')
    parser.add_argument('--show-diff', action='store_true',
                       help='Show first 500 chars of before/after')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    # Initialize processor
    processor = OCRPostProcessor(
        use_spellcheck=not args.no_spellcheck
    )

    # Process file
    result = processor.process_file(
        args.input,
        args.output
    )

    # Show diff if requested
    if args.show_diff:
        print(f"\n{'='*80}")
        print("BEFORE (first 500 chars):")
        print(f"{'='*80}")
        print(result['original_text'][:500])
        print("\n" + "="*80)
        print("AFTER (first 500 chars):")
        print(f"{'='*80}")
        print(result['processed_text'][:500])
        print(f"\n{'='*80}\n")

    print(f"\n✓ Processing complete!")
    print(f"Corrections made: {result['corrections_made']}")
    print(f"Improvement: {result['improvement_percentage']:.1f}% of words corrected\n")


if __name__ == "__main__":
    main()
