#!/usr/bin/env python3
"""
Prepare complete Frank Ramsey corpus for LLM training
Loads ALL .txt files from data/extracted (except intermediates)
"""

import json
from pathlib import Path
from typing import List, Dict

def load_all_final_texts(data_dir: Path) -> List[Dict[str, str]]:
    """Load all final text files, excluding intermediate processing files"""

    # Patterns to exclude (intermediate files)
    exclude_patterns = ['_ocr.txt', '_spell.txt', '_temp.txt', '_corrected.txt', '_metadata.json']

    documents = []

    # Get all .txt files
    for filepath in sorted(data_dir.glob('*.txt')):
        # Skip intermediate files
        if any(pattern in filepath.name for pattern in exclude_patterns):
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            word_count = len(text.split())

            # Skip empty or very small files
            if word_count < 1000:
                print(f"⚠ Skipping {filepath.name}: only {word_count} words")
                continue

            documents.append({
                'filename': filepath.name,
                'text': text,
                'words': word_count,
                'chars': len(text)
            })
            print(f"✓ Loaded {filepath.name}: {word_count:,} words")

    return documents


def create_consolidated_corpus(documents: List[Dict], output_path: Path):
    """Create single consolidated corpus file"""
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
    data_dir = Path('data/extracted')
    output_dir = Path('data/training')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("PREPARING FRANK RAMSEY CORPUS FOR LLM TRAINING")
    print("="*80)
    print()

    # Load all final documents
    documents = load_all_final_texts(data_dir)

    if not documents:
        print("\n❌ No documents found!")
        return

    # Create consolidated corpus
    corpus = create_consolidated_corpus(
        documents,
        output_dir / 'ramsey_corpus_full.txt'
    )

    # Create chunked version
    chunks = create_chunked_corpus(
        corpus,
        output_dir,
        chunk_size=2048
    )

    # Create metadata
    metadata = create_metadata(
        documents,
        corpus,
        chunks,
        output_dir / 'corpus_metadata.json'
    )

    # Print summary
    print("\n" + "="*80)
    print("CORPUS PREPARATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}/")
    print(f"\nFiles created:")
    print(f"  - ramsey_corpus_full.txt     (complete corpus)")
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
