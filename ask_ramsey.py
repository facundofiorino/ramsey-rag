#!/usr/bin/env python3
"""
Full RAG Pipeline for Frank Ramsey Corpus

Ask questions and get AI-generated answers based on Ramsey's actual writings.
Combines semantic search + LLM generation using Ollama.
"""

import argparse
import sys
from pathlib import Path


def ask_ramsey(question: str, model: str = "llama3:latest", num_sources: int = 4, verbose: bool = True):
    """
    Ask a question about Ramsey's philosophy and get an AI-generated answer
    based on relevant passages from the corpus.
    """
    try:
        from langchain_ollama import OllamaLLM
        from langchain_chroma import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstall with:")
        print("  pip install langchain-ollama langchain-chroma langchain-community sentence-transformers")
        return None

    # Check vector store exists
    vectorstore_path = Path("data/vectorstore")
    if not vectorstore_path.exists():
        print("❌ Vector store not found!")
        print("Run: python demo_ramsey_rag.py")
        return None

    if verbose:
        print("="*80)
        print("FRANK RAMSEY RAG SYSTEM")
        print("="*80)
        print(f"\n🔍 Question: {question}")
        print(f"🤖 Model: {model}")
        print(f"📚 Searching corpus (584K words)...\n")

    # Load vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=str(vectorstore_path),
        embedding_function=embeddings
    )

    # Create custom prompt
    prompt_template = """You are a philosophical assistant specializing in Frank Ramsey's work.
Answer the question based ONLY on the provided context from Ramsey's writings.
If the context doesn't contain enough information, say so.

Context from Ramsey's writings:
{context}

Question: {question}

Answer (be scholarly but clear, cite specific ideas when possible):"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Initialize Ollama
    llm = OllamaLLM(model=model, temperature=0.3)

    # Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": num_sources}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    # Get answer
    if verbose:
        print("🧠 Generating answer...\n")

    result = qa_chain.invoke({"query": question})

    if verbose:
        print("="*80)
        print("📝 ANSWER")
        print("="*80)
        print(f"\n{result['result']}\n")

        print("="*80)
        print(f"📖 SOURCES ({len(result['source_documents'])} passages used)")
        print("="*80)
        for i, doc in enumerate(result['source_documents'], 1):
            preview = doc.page_content[:300].replace('\n', ' ')
            print(f"\n{i}. {preview}...")

        print("\n" + "="*80)

    return result


def interactive_mode(model: str = "llama3:latest"):
    """Interactive question-answering mode"""
    print("="*80)
    print("FRANK RAMSEY RAG - Interactive Mode")
    print("="*80)
    print(f"Model: {model}")
    print("Ask questions about Ramsey's philosophy")
    print("Type 'quit' or 'exit' to stop\n")

    while True:
        try:
            question = input("\n💭 Your question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if not question:
                continue

            if question == '?':
                print("\nExample questions:")
                print("  - What did Ramsey think about truth?")
                print("  - How did Ramsey approach probability?")
                print("  - What is Ramsey's view on causality?")
                print("  - How did Ramsey's pragmatism differ from others?")
                print("  - What is the Ramsey-Lewis theory?")
                continue

            print()  # Blank line for readability
            ask_ramsey(question, model=model)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Try another question or type 'quit' to exit")


def main():
    parser = argparse.ArgumentParser(
        description='Ask questions about Frank Ramsey using RAG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Ask a single question
  python ask_ramsey.py "What did Ramsey think about truth?"

  # Interactive mode
  python ask_ramsey.py -i

  # Use different model (more powerful but slower)
  python ask_ramsey.py -i --model qwen2.5:14b

  # More source passages for complex questions
  python ask_ramsey.py "Explain Ramsey's theory of universals" --sources 6

Example questions:
  - What did Ramsey think about truth?
  - How did Ramsey approach probability?
  - What is Ramsey's view on causality?
  - How did Ramsey's pragmatism differ from Peirce and James?
  - What is the Ramsey-Lewis theory?
  - What did Ramsey say about the foundations of mathematics?
        '''
    )

    parser.add_argument('question', nargs='?', help='Question to ask (omit for interactive mode)')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--model', default='llama3:latest',
                       help='Ollama model (default: llama3:latest, try: qwen2.5:14b for better quality)')
    parser.add_argument('--sources', type=int, default=4,
                       help='Number of source passages to retrieve (default: 4)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress verbose output (answer only)')

    args = parser.parse_args()

    # Interactive or single query
    if args.interactive or not args.question:
        interactive_mode(model=args.model)
    else:
        result = ask_ramsey(
            args.question,
            model=args.model,
            num_sources=args.sources,
            verbose=not args.quiet
        )

        if result is None:
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
