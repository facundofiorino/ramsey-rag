#!/usr/bin/env python3
"""
Query the Frank Ramsey RAG system

Interactive question-answering using your corpus with Ollama.
"""

import argparse
import sys
from pathlib import Path


def query_ramsey(question: str, model: str = "llama3.2:8b", num_sources: int = 3, show_sources: bool = True):
    """Query the Ramsey corpus using RAG"""
    try:
        from langchain_community.chains import RetrievalQA
        from langchain_community.llms import Ollama
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        # Fallback to old imports
        from langchain.chains import RetrievalQA
        from langchain.llms import Ollama
        from langchain.vectorstores import Chroma
        from langchain.embeddings import HuggingFaceEmbeddings

    # Check vector store exists
    vectorstore_path = Path("data/vectorstore")
    if not vectorstore_path.exists():
        print("❌ Vector store not found!")
        print("Run: python create_rag_system.py")
        return None

    print(f"Loading vector store from {vectorstore_path}...")

    # Load vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=str(vectorstore_path),
        embedding_function=embeddings
    )

    # Connect to Ollama
    print(f"Connecting to Ollama (model: {model})...")
    llm = Ollama(model=model)

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": num_sources}),
        return_source_documents=True
    )

    # Ask question
    print(f"\nQuestion: {question}\n")
    print("Thinking...")
    result = qa_chain(question)

    print("\n" + "="*80)
    print("ANSWER")
    print("="*80)
    print(result['result'])

    if show_sources:
        print("\n" + "="*80)
        print(f"SOURCES ({len(result['source_documents'])} passages used)")
        print("="*80)
        for i, doc in enumerate(result['source_documents'], 1):
            preview = doc.page_content[:300].replace('\n', ' ')
            print(f"\n{i}. {preview}...")

    return result


def interactive_mode(model: str = "llama3.2:8b"):
    """Interactive query mode"""
    print("="*80)
    print("FRANK RAMSEY RAG SYSTEM - Interactive Mode")
    print("="*80)
    print(f"Model: {model}")
    print("Type 'quit' or 'exit' to stop\n")

    while True:
        try:
            question = input("\nYour question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if not question:
                continue

            query_ramsey(question, model=model)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Try another question or type 'quit' to exit")


def main():
    parser = argparse.ArgumentParser(
        description='Query Frank Ramsey corpus using RAG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Ask a single question
  python query_ramsey.py "What did Ramsey think about truth?"

  # Interactive mode
  python query_ramsey.py -i

  # Use different Ollama model
  python query_ramsey.py -i --model qwen2.5:14b

  # More source passages
  python query_ramsey.py "What is Ramsey's theory?" --sources 5
        '''
    )

    parser.add_argument('question', nargs='?', help='Question to ask (omit for interactive mode)')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--model', default='llama3.2:8b', help='Ollama model to use')
    parser.add_argument('--sources', type=int, default=3, help='Number of source passages to retrieve')
    parser.add_argument('--no-show-sources', action='store_true', help='Hide source passages')

    args = parser.parse_args()

    # Check dependencies
    try:
        import langchain
        import chromadb
        import sentence_transformers
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstall with: pip install langchain chromadb sentence-transformers")
        return 1

    # Interactive or single query
    if args.interactive or not args.question:
        interactive_mode(model=args.model)
    else:
        query_ramsey(
            args.question,
            model=args.model,
            num_sources=args.sources,
            show_sources=not args.no_show_sources
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
