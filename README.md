# Frank Ramsey RAG System

A complete RAG (Retrieval-Augmented Generation) system for querying Frank Ramsey's philosophical works using semantic search and local LLMs.

## 🎯 Quick Start

### Ask Questions About Ramsey's Philosophy

```bash
# Interactive mode (recommended)
python ask_ramsey.py -i

# Single question
python ask_ramsey.py "What did Ramsey think about truth?"

# Use a better model for complex questions
python ask_ramsey.py -i --model qwen2.5:14b
```

### Example Questions

- "What did Ramsey think about truth?"
- "How did Ramsey approach probability?"
- "What is Ramsey's view on causality?"
- "What did Ramsey say about the foundations of mathematics?"
- "How did Ramsey's pragmatism differ from Peirce and James?"

## 📊 What You Have

**Corpus**: 583,862 words from 8 philosophical texts
**Vector Store**: 4,696 searchable chunks with semantic embeddings
**Quality**: 85-90% average (high-quality PDFs + OCR-corrected scans)
**Cost**: FREE (100% local, no API costs)

## 📁 Project Structure

```
ramsey_training/
├── ask_ramsey.py              # 🎯 MAIN SCRIPT - Full RAG Q&A system
│
├── ramsey_data/               # Source PDFs (8 books, 58MB)
│
├── data/
│   ├── extracted/             # Extracted text files
│   ├── training/              # Training corpus
│   │   ├── ramsey_corpus_full.txt      (584K words)
│   │   ├── ramsey_corpus_chunks.jsonl  (pre-chunked)
│   │   └── corpus_metadata.json        (statistics)
│   └── vectorstore/           # Semantic search database (50MB)
│
├── src/                       # Extraction pipeline code
│   └── data/
│       ├── extract_all.py     # PDF extraction orchestrator
│       ├── processors/        # OCR, spell-check, LLM correction
│       └── extractors/        # PDF text extraction
│
├── scripts/                   # Utility scripts
│   ├── prepare_complete_corpus.py  # Build training corpus
│   ├── create_rag_system.py        # Create vector database
│   ├── demo_ramsey_rag.py          # Semantic search demo
│   ├── run_llm_correction_pipeline.sh
│   ├── utilities/             # Helper scripts
│   └── archive/               # Old one-off scripts
│
├── docs/                      # Documentation
│   ├── FULL_RAG_COMPLETE.md   # Complete guide & examples
│   ├── README_RAG.md          # Quick start guide
│   ├── TRAINING_GUIDE.md      # Advanced training options
│   └── guides/                # OCR & extraction guides
│
├── logs/                      # Processing logs
└── venv/                      # Python virtual environment
```

## 🚀 Installation

### Prerequisites

```bash
# Python 3.8+
python --version

# Ollama (for LLM)
ollama --version
```

### Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Install RAG dependencies (if needed)
pip install langchain-ollama langchain-chroma langchain-community sentence-transformers

# Pull Ollama models (if needed)
ollama pull llama3:latest
ollama pull qwen2.5:14b  # Optional, for better quality
```

### Verify Installation

```bash
# Check vector store exists
ls data/vectorstore/

# Test query
python ask_ramsey.py "What did Ramsey think about truth?"
```

## 📚 How It Works

### 4-Phase Pipeline

**Phase 1: Document Extraction**
```
PDFs → Text Extraction → OCR (if needed) → Spell Check → LLM Correction → Clean Text
```

**Phase 2: Corpus Preparation**
```
8 Text Files → Consolidation → 584K word corpus → Pre-chunked for training
```

**Phase 3: RAG Setup**
```
Corpus → Text Splitting → Vector Embeddings → ChromaDB → Searchable Index
```

**Phase 4: Question Answering**
```
Question → Semantic Search → Top Passages → Ollama LLM → Answer + Citations
```

## 💡 Command-Line Options

```bash
# Interactive mode
python ask_ramsey.py -i

# Single question
python ask_ramsey.py "your question here"

# Use different model (slower but better quality)
python ask_ramsey.py -i --model qwen2.5:14b

# More source passages for complex questions
python ask_ramsey.py "complex question" --sources 6

# Quiet mode (answer only)
python ask_ramsey.py "question" -q
```

## 🤖 Available Models

You have these Ollama models installed:
- **llama3:latest** (default) - Fast, good quality
- **qwen2.5:latest** - Good balance
- **qwen2.5:14b** - Best quality (slower, 9GB)
- **mistral:latest** - Alternative
- **deepseek-r1:latest** - Reasoning-focused

Recommended: Start with **llama3:latest**, use **qwen2.5:14b** for complex philosophical questions.

## 📖 Source Materials (8 Books)

1. **Frank Ramsey: a sheer excess of powers** - 225,555 words
2. **Frank Ramsey and the Realistic Spirit** - 119,687 words
3. **Ramsey's legacy** - 88,564 words
4. **On Truth: Original Manuscripts** - 63,793 words
5. **Truth and Success** (OCR) - 44,765 words
6. **Shooting Star Biography** - 25,112 words
7. **General Propositions** (OCR) - 8,663 words
8. **Theories** - 7,558 words

**Total**: 583,862 words | 4,696 searchable chunks | All 8 PDFs processed ✓

## 🛠️ Troubleshooting

**"Vector store not found"**
```bash
python scripts/demo_ramsey_rag.py  # Creates it automatically
```

**"Model not found"**
```bash
ollama list              # See available models
ollama pull llama3       # Download if needed
```

**Slow responses**
- First query loads models (slow)
- Subsequent queries are faster
- Try smaller model: `--model qwen2.5:latest`

**Poor quality answers**
- Use more sources: `--sources 6`
- Try better model: `--model qwen2.5:14b`
- Make question more specific

## 📝 Documentation

- **docs/FULL_RAG_COMPLETE.md** - Complete usage guide with examples
- **docs/README_RAG.md** - Quick start guide
- **docs/TRAINING_GUIDE.md** - Advanced training options (fine-tuning, etc.)
- **docs/guides/** - OCR extraction pipeline documentation

## 💪 Tips for Better Answers

1. **Be Specific**: "What did Ramsey think about truth?" vs "Tell me about Ramsey"

2. **Use More Sources for Complex Questions**:
   ```bash
   python ask_ramsey.py "Explain Ramsey's theory of universals" --sources 6
   ```

3. **Try Different Models**:
   - llama3:latest - Fast, conversational
   - qwen2.5:14b - Slower, more thorough

4. **Check Sources**: Always review source passages to verify AI's interpretation

## ⚡ Performance

- **Search Speed**: <1 second
- **Answer Generation**: 5-30 seconds (model dependent)
- **Accuracy**: High (grounded in actual Ramsey texts)
- **Cost**: FREE (100% local)

## 🌟 What Makes This Special

- **Grounded**: Answers based on ACTUAL Ramsey texts, not hallucinations
- **Cited**: Every answer shows source passages
- **Free**: No API costs, runs entirely locally
- **Fast**: Semantic search across 584K words in <1 second
- **Quality**: 85-90% text quality from advanced OCR + LLM correction

## 📜 License

For educational and research purposes. Source texts are in the public domain or used under fair use.

## 🙏 Credits

Built on:
- **584,000 words** from 8 philosophical texts
- **OCR**: Tesseract at 800 DPI
- **LLM Correction**: qwen2.5:14b
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector DB**: ChromaDB
- **LLM**: Ollama (local)

---

**Questions?** See `docs/FULL_RAG_COMPLETE.md` for detailed examples and advanced usage.
