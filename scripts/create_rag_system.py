#!/usr/bin/env python3
"""
Create a RAG (Retrieval-Augmented Generation) system for Frank Ramsey corpus

This is the recommended FREE approach for using your corpus with LLMs.
Uses local Ollama + vector embeddings for intelligent Q&A.
"""

import argparse
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required = {
        'langchain': 'pip install langchain',
        'chromadb': 'pip install chromadb',
        'sentence_transformers': 'pip install sentence-transformers',
    }

    missing = []
    for package, install_cmd in required.items():
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append((package, install_cmd))

    if missing:
        print("Missing required packages:\n")
        for package, cmd in missing:
            print(f"  {package}: {cmd}")
        print("\nInstall all at once:")
        print("  pip install langchain chromadb sentence-transformers")
        return False

    return True


def create_vector_store(corpus_path: Path, output_dir: Path, chunk_size: int = 1000):
    """Create vector store from corpus"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.document_loaders import TextLoader

    print(f"Loading corpus: {corpus_path}")
    loader = TextLoader(str(corpus_path))
    documents = loader.load()

    print(f"Splitting into chunks (size={chunk_size}, overlap=200)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} chunks")

    print("Creating embeddings (this may take a few minutes)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building vector store...")
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=str(output_dir)
    )

    print(f"\n✓ Vector store created: {output_dir}/")
    print(f"  Total chunks: {len(texts)}")
    return vectorstore


def test_query(vectorstore):
    """Test the vector store with sample queries"""
    test_questions = [
        "What did Ramsey think about truth?",
        "How did Ramsey approach probability?",
        "What is Ramsey's view on the foundations of mathematics?",
    ]

    print("\n" + "="*80)
    print("TESTING VECTOR STORE")
    print("="*80)

    for question in test_questions:
        print(f"\nQ: {question}")
        results = vectorstore.similarity_search(question, k=2)
        print(f"Found {len(results)} relevant passages:")
        for i, doc in enumerate(results, 1):
            preview = doc.page_content[:200].replace('\n', ' ')
            print(f"  {i}. {preview}...")


def create_qa_chain():
    """Example code for creating a Q&A chain"""
    example_code = '''
# Example: Use the RAG system with Ollama

from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Load vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    persist_directory="data/vectorstore",
    embedding_function=embeddings
)

# Connect to Ollama
llm = Ollama(model="llama3.2:8b")

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Ask questions
result = qa_chain("What did Frank Ramsey think about truth?")
print(f"Answer: {result['result']}")
print(f"\\nSources: {len(result['source_documents'])} passages used")
'''
    return example_code


def main():
    parser = argparse.ArgumentParser(
        description='Create RAG system for Frank Ramsey corpus',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example usage:
  # Create vector store
  python create_rag_system.py

  # Custom chunk size
  python create_rag_system.py --chunk-size 1500

  # Skip testing
  python create_rag_system.py --no-test
        '''
    )

    parser.add_argument('--corpus', type=Path, default=Path('data/training/ramsey_corpus_full.txt'),
                       help='Path to corpus file')
    parser.add_argument('--output', type=Path, default=Path('data/vectorstore'),
                       help='Output directory for vector store')
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Text chunk size (default: 1000 chars)')
    parser.add_argument('--no-test', action='store_true',
                       help='Skip testing queries')

    args = parser.parse_args()

    print("="*80)
    print("FRANK RAMSEY RAG SYSTEM SETUP")
    print("="*80)
    print()

    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return 1

    # Check corpus exists
    if not args.corpus.exists():
        print(f"\n❌ Corpus not found: {args.corpus}")
        print("Run: python prepare_complete_corpus.py first")
        return 1

    # Create vector store
    args.output.mkdir(parents=True, exist_ok=True)
    vectorstore = create_vector_store(args.corpus, args.output, args.chunk_size)

    # Test queries
    if not args.no_test:
        test_query(vectorstore)

    # Print next steps
    print("\n" + "="*80)
    print("RAG SYSTEM READY")
    print("="*80)
    print(f"\nVector store location: {args.output}/")
    print("\nNext steps:")
    print("1. Save this code to query_ramsey.py:")
    print("\n" + create_qa_chain())
    print("\n2. Run queries:")
    print('   python query_ramsey.py "What did Ramsey think about truth?"')
    print("\n3. Or use in your own code (see example above)")
    print("="*80)

    return 0


if __name__ == '__main__':
    exit(main())
