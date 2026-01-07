#!/usr/bin/env python3
"""
Simple demonstration of Ramsey RAG system

Simpler approach that works reliably with current LangChain versions.
"""

from pathlib import Path


def demo_search():
    """Demonstrate semantic search on Ramsey corpus"""
    print("="*80)
    print("FRANK RAMSEY RAG SYSTEM - DEMO")
    print("="*80)
    print("\nThis demo shows semantic search on your Ramsey corpus.")
    print("For full question-answering, use the complete RAG setup below.\n")

    # Check if vector store exists
    vectorstore_path = Path("data/vectorstore")
    if not vectorstore_path.exists():
        print("❌ Vector store not found. Creating it now...")
        print("\nThis will take a few minutes on first run...\n")
        create_vectorstore()

    print("Loading vector store...")

    try:
        from langchain_chroma import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        print("❌ Missing dependencies")
        print("Run: pip install langchain-chroma langchain-community sentence-transformers")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=str(vectorstore_path),
        embedding_function=embeddings
    )

    # Demo questions
    questions = [
        "What did Ramsey think about truth?",
        "How did Ramsey approach probability?",
        "What is Ramsey's view on causality?",
    ]

    for question in questions:
        print(f"\n{'='*80}")
        print(f"QUESTION: {question}")
        print("="*80)

        # Search for relevant passages
        docs = vectorstore.similarity_search(question, k=3)

        print(f"\nFound {len(docs)} relevant passages:\n")
        for i, doc in enumerate(docs, 1):
            preview = doc.page_content[:400].replace('\n', ' ')
            print(f"{i}. {preview}...\n")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\nTo get AI-generated answers (not just search results), you have two options:\n")
    print("Option 1: Use Ollama directly")
    print("  - Copy relevant passages from search results")
    print("  - Paste into Ollama chat with your question")
    print("  - Example: ollama run llama3.2:8b")
    print()
    print("Option 2: Set up full RAG pipeline")
    print("  - See TRAINING_GUIDE.md for complete setup")
    print("  - Requires langchain-ollama package")
    print("  - Automatically retrieves + generates answers")
    print("="*80)


def create_vectorstore():
    """Create vector store from corpus"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import TextLoader

    corpus_path = Path("data/training/ramsey_corpus_full.txt")
    if not corpus_path.exists():
        print(f"❌ Corpus not found: {corpus_path}")
        print("Run: python prepare_complete_corpus.py")
        return

    print(f"Loading corpus from {corpus_path}...")
    loader = TextLoader(str(corpus_path))
    documents = loader.load()

    print("Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} chunks")

    print("Creating embeddings (this takes a few minutes)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building vector store...")
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="data/vectorstore"
    )

    print(f"\n✓ Vector store created: data/vectorstore/")
    print(f"  Total chunks: {len(texts)}")


if __name__ == '__main__':
    demo_search()
