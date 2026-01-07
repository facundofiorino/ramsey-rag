#!/usr/bin/env python3
"""
Local LLM-based OCR Post-Processor

Uses Ollama to intelligently correct OCR errors while preserving technical
terms and philosophical concepts. Free alternative to Claude API.

Recommended models:
- llama3.2:8b - Fast, good quality
- mistral:latest - Good for technical text
- qwen2.5:14b - Best quality (slower)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List
import json

try:
    import requests
except ImportError:
    print("Error: requests library not installed")
    print("Install with: pip install requests")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaOCRCorrector:
    """Uses local Ollama LLM to correct OCR errors"""

    def __init__(self, model: str = "llama3.2:8b", chunk_size: int = 1000):
        """
        Initialize corrector

        Args:
            model: Ollama model to use
            chunk_size: Words per chunk (larger = slower but better context)
        """
        self.model = model
        self.chunk_size = chunk_size
        self.ollama_url = "http://localhost:11434/api/generate"

        # Verify Ollama is running
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code != 200:
                logger.error("Ollama is not running. Start it with: ollama serve")
                sys.exit(1)

            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]

            if model not in model_names:
                logger.warning(f"Model {model} not found. Available: {model_names}")
                logger.info(f"Pulling model... This may take a while.")
                self._pull_model(model)
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Make sure it's running: ollama serve")
            sys.exit(1)

    def _pull_model(self, model: str):
        """Pull model if not available"""
        import subprocess
        subprocess.run(["ollama", "pull", model], check=True)

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks for processing"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size):
            # Add 10% overlap for context
            overlap = int(self.chunk_size * 0.1)
            start = max(0, i - overlap)
            end = min(len(words), i + self.chunk_size + overlap)
            chunk = ' '.join(words[start:end])
            chunks.append((chunk, start, end))

        return chunks

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistent corrections
                        "top_p": 0.9,
                    }
                },
                timeout=600  # 10 minutes for qwen2.5:14b processing
            )

            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return None

    def correct_chunk(self, chunk: str) -> str:
        """Correct OCR errors in a text chunk"""
        prompt = f"""You are an OCR error correction assistant specializing in philosophical texts by Frank Ramsey.

Fix OCR errors in the following text, following these rules:

1. ONLY fix clear OCR mistakes (character recognition errors)
2. PRESERVE all technical terms, proper nouns, and philosophical concepts
3. Do NOT change the meaning or rephrase anything
4. Do NOT add explanations or comments
5. Common OCR errors to fix:
   - rn → m (e.g., "wom" → "worn" is wrong, "tirne" → "time" is right)
   - vv → w (e.g., "vvith" → "with")
   - l → I or i (depending on context)
   - 0 → O (in words, not numbers)
   - 1 → l (in words)
   - Missing or extra spaces

Text to correct:
{chunk}

Output ONLY the corrected text, nothing else:"""

        corrected = self._call_ollama(prompt)
        return corrected if corrected else chunk

    def process_file(self, input_path: Path, output_path: Path = None, show_progress: bool = True) -> dict:
        """
        Process an OCR file and correct errors

        Args:
            input_path: Input file path
            output_path: Output file path (optional)
            show_progress: Show progress updates

        Returns:
            Statistics dictionary
        """
        logger.info(f"Processing: {input_path.name}")
        logger.info(f"Model: {self.model}")
        logger.info(f"Chunk size: {self.chunk_size} words")

        # Read input
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Chunk text
        chunks = self._chunk_text(text)
        logger.info(f"Split into {len(chunks)} chunks")

        # Process chunks
        corrected_chunks = []
        for i, (chunk, start, end) in enumerate(chunks, 1):
            if show_progress:
                logger.info(f"Processing chunk {i}/{len(chunks)} (words {start}-{end})...")

            corrected = self.correct_chunk(chunk)
            corrected_chunks.append(corrected)

        # Merge chunks (simple concatenation - overlaps will be similar)
        # For better merging, could detect overlap and deduplicate
        corrected_text = '\n\n'.join(corrected_chunks)

        # Save output
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(corrected_text)
            logger.info(f"✓ Saved to: {output_path}")

        # Calculate stats
        stats = {
            'input_file': str(input_path),
            'output_file': str(output_path) if output_path else None,
            'input_words': len(text.split()),
            'output_words': len(corrected_text.split()),
            'chunks_processed': len(chunks),
            'model': self.model
        }

        return stats


def main():
    parser = argparse.ArgumentParser(
        description='Correct OCR errors using local LLM (Ollama)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Correct a single file
  python src/llm_ocr_corrector.py input.txt -o corrected.txt

  # Use a different model
  python src/llm_ocr_corrector.py input.txt -m mistral:latest

  # Process larger chunks for better context
  python src/llm_ocr_corrector.py input.txt --chunk-size 2000

Recommended models (in order of quality):
  - llama3.2:8b (default) - Fast, good quality
  - mistral:latest - Good for technical text
  - qwen2.5:14b - Best quality (slower)
  - llama3.2:3b - Fastest (lower quality)
        '''
    )

    parser.add_argument('input', type=Path, help='Input OCR text file')
    parser.add_argument('-o', '--output', type=Path, help='Output file (default: input_corrected.txt)')
    parser.add_argument('-m', '--model', default='llama3.2:8b', help='Ollama model to use')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Words per chunk (default: 1000)')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress messages')

    args = parser.parse_args()

    # Set output path
    if not args.output:
        args.output = args.input.parent / f"{args.input.stem}_llm_corrected.txt"

    # Process
    corrector = OllamaOCRCorrector(model=args.model, chunk_size=args.chunk_size)
    stats = corrector.process_file(args.input, args.output, show_progress=not args.quiet)

    # Print summary
    print("\n" + "="*80)
    print("LLM CORRECTION COMPLETE")
    print("="*80)
    print(f"Model: {stats['model']}")
    print(f"Input words: {stats['input_words']:,}")
    print(f"Output words: {stats['output_words']:,}")
    print(f"Chunks processed: {stats['chunks_processed']}")
    print(f"Output: {stats['output_file']}")
    print("="*80)


if __name__ == '__main__':
    main()
