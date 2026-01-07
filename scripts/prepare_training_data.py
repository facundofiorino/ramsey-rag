#!/usr/bin/env python3
"""
Prepare Frank Ramsey corpus for LLM training

Consolidates all high-quality extracted texts into training-ready formats:
- Single concatenated corpus
- Chunked files for efficient training
- Metadata and statistics
"""

import argparse
from pathlib import Path
from typing import List, Dict
import json


def load_text_files(data_dir: Path) -> List[Dict[str, str]]:
    """Load all final text files"""
    files = [
        # High-quality direct extracts (88-90% quality)
        "[Episteme №16] Frank Plumpton Ramsey (auth.), Nicholas Rescher, Ulrich Majer (e - On Truth_ Original Manuscript Materials (1927–1929) from the Ramsey Collection at the University of Pittsburgh (1991, Springer) [10.1007_978-94- - libgen.li.txt",
        "[History of Analytic Philosophy ] S. J. Methven (auth.) - Frank Ramsey and the Realistic Spirit (2015, Palgrave Macmillan) [10.1057_9781137351081] - libgen.li.txt",
        "[Mind Association occasional series] Lillehammer Hallvard, Mellor D.H. (eds.) - Ramsey's legacy_ [this volume contains revised versions of ten of the thirteen original papers given and discussed at the Frank (2005, Oxford Unive - libgen.li.txt",
        "Frank Ramsey _ a sheer excess of powers -- Misak, Cheryl Jayne -- Impr_ 3, Oxford, United Kingdom, 2020 -- Oxford University Press, USA -- 9780198755357 -- b305d4f92bef056c43aa8a84e7226563 -- Anna's Archive.txt",
        "Karl Sabbagh - Shooting Star_ The Brief and Brilliant Life of Frank Ramsey (2013, Amazon) - libgen.li.txt",
        "Theories -- Ramsey, Frank -- 0 -- 683f433d51c4ad68d760a73469e25be7 -- Anna's Archive.txt",

        # OCR-corrected documents (80-89% quality)
        "truth_and_success_final.txt",
        "general_propositions_final.txt",
    ]

    documents = []
    for filename in files:
        # Try direct path first
        filepath = data_dir / filename
        if not filepath.exists():
            # Try glob matching for Unicode issues
            matches = list(data_dir.glob(filename.replace("_", "?").replace(" ", "*")))
            if matches:
                filepath = matches[0]
                filename = filepath.name

        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                documents.append({
                    'filename': filename,
                    'text': text,
                    'words': len(text.split()),
                    'chars': len(text)
                })
                print(f"✓ Loaded {filename}: {len(text.split()):,} words")
        else:
            print(f"⚠ Missing: {filename}")

    return documents


def create_consolidated_corpus(documents: List[Dict], output_path: Path):
    """Create single consolidated corpus file"""

    # Add document separators for clarity
    separator = "\n\n" + "="*80 + "\n\n"

    corpus_parts = []
    for doc in documents:
        header = f"SOURCE: {doc['filename']}\n"
        corpus_parts.append(header + doc['text'])

    consolidated = separator.join(corpus_parts)

    output_path.write_text(consolidated, encoding='utf-8')
    print(f"\n✓ Consolidated corpus saved: {output_path}")
    print(f"  Total words: {len(consolidated.split()):,}")
    print(f"  Total chars: {len(consolidated):,}")

    return consolidated


def create_chunked_corpus(text: str, output_dir: Path, chunk_size: int = 2048):
    """Split corpus into chunks for training"""

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)

    # Save as JSONL (common format for LLM training)
    jsonl_path = output_dir / "ramsey_corpus_chunks.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            entry = {
                'id': f'ramsey_{i:04d}',
                'text': chunk,
                'source': 'Frank Ramsey Philosophical Works',
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            f.write(json.dumps(entry) + '\n')

    print(f"\n✓ Chunked corpus saved: {jsonl_path}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Chunk size: ~{chunk_size} words")

    return chunks


def create_metadata(documents: List[Dict], corpus: str, chunks: List[str], output_path: Path):
    """Create metadata file"""

    metadata = {
        'corpus_name': 'Frank Ramsey Philosophical Corpus',
        'creation_date': '2025-12-12',
        'total_documents': len(documents),
        'total_words': len(corpus.split()),
        'total_characters': len(corpus),
        'total_chunks': len(chunks),
        'chunk_size': 2048,
        'quality_assessment': {
            'direct_extracts': {
                'documents': 5,
                'quality': '88-90%',
                'words': sum(d['words'] for d in documents[:5])
            },
            'ocr_corrected': {
                'documents': 2,
                'quality': '80-89%',
                'words': sum(d['words'] for d in documents[5:])
            }
        },
        'documents': [
            {
                'filename': d['filename'],
                'words': d['words'],
                'characters': d['chars']
            }
            for d in documents
        ],
        'recommended_use': [
            'Fine-tuning language models on philosophical reasoning',
            'Training models on Frank Ramsey\'s logical and philosophical style',
            'Pre-training data for philosophy-focused LLMs',
            'Question-answering systems about Ramsey\'s philosophy'
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Metadata saved: {output_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description='Prepare Frank Ramsey corpus for LLM training')
    parser.add_argument('--input', type=Path, default=Path('data/extracted'),
                       help='Input directory with extracted texts')
    parser.add_argument('--output', type=Path, default=Path('data/training'),
                       help='Output directory for training data')
    parser.add_argument('--chunk-size', type=int, default=2048,
                       help='Words per training chunk (default: 2048)')

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("PREPARING FRANK RAMSEY CORPUS FOR LLM TRAINING")
    print("="*80)
    print()

    # Load all documents
    documents = load_text_files(args.input)

    if not documents:
        print("\n❌ No documents found!")
        return

    # Create consolidated corpus
    corpus = create_consolidated_corpus(
        documents,
        args.output / 'ramsey_corpus_full.txt'
    )

    # Create chunked version
    chunks = create_chunked_corpus(
        corpus,
        args.output,
        chunk_size=args.chunk_size
    )

    # Create metadata
    metadata = create_metadata(
        documents,
        corpus,
        chunks,
        args.output / 'corpus_metadata.json'
    )

    # Print summary
    print("\n" + "="*80)
    print("CORPUS PREPARATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {args.output}/")
    print(f"\nFiles created:")
    print(f"  - ramsey_corpus_full.txt    (complete corpus)")
    print(f"  - ramsey_corpus_chunks.jsonl (chunked for training)")
    print(f"  - corpus_metadata.json       (statistics and info)")
    print(f"\nReady for:")
    print(f"  - Fine-tuning with Hugging Face Transformers")
    print(f"  - Training with Ollama (modelfile creation)")
    print(f"  - GPT-style model fine-tuning")
    print(f"  - RAG (Retrieval-Augmented Generation) systems")
    print("="*80)


if __name__ == '__main__':
    main()
