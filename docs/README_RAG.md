# Frank Ramsey RAG System - Quick Start

Your Ramsey corpus is now set up with a RAG (Retrieval-Augmented Generation) system!

## What You Have

**Corpus**: 583,862 words (~584K) from 8 philosophical texts
**Vector Store**: 4,696 searchable chunks with semantic embeddings
**Quality**: 85-90% average (high-quality + OCR-corrected)

##Files

- **ramsey_corpus_full.txt** - Complete corpus (3.4MB)
- **ramsey_corpus_chunks.jsonl** - Pre-chunked for training (286 chunks)
- **corpus_metadata.json** - Statistics and info
- **vectorstore/** - Semantic search database

## Quick Start

### Demo (Semantic Search Only)

```bash
python demo_ramsey_rag.py
```

This shows you relevant passages from Ramsey's works based on your questions.

**Example Output:**
```
QUESTION: What did Ramsey think about truth?

Found 3 relevant passages:

1. Let us note here that the same suspicion can be entertained if one
   attributes to Ramsey a simple redundancy conception of truth...

2. The thesis that truth is but an expedient is often associated with
   a third variety of pragmatist views...

3. Facts and Propositions' can be seen as Ramsey's official rejection
   of much of the logical analyst theory...
```

## Using the Results

### Option 1: Manual (Simplest)

1. Run: `python demo_ramsey_rag.py`
2. Copy relevant passages
3. Paste into Ollama chat with your question:
   ```bash
   ollama run llama3.2:8b
   ```
4. Include context: "Based on these passages from Ramsey: [paste], what did he think about [your question]?"

### Option 2: Programmatic

Use the vector store in your own Python code:

```python
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    persist_directory="data/vectorstore",
    embedding_function=embeddings
)

# Search
docs = vectorstore.similarity_search("What did Ramsey think about truth?", k=3)

# Print results
for doc in docs:
    print(doc.page_content)
```

## Example Questions

Try asking:
- "What did Ramsey think about truth?"
- "How did Ramsey approach probability?"
- "What is Ramsey's view on causality?"
- "What did Ramsey say about the foundations of mathematics?"
- "How did Ramsey's pragmatism differ from others?"
- "What is the Ramsey-Lewis theory?"

## How It Works

1. **Your Question** → Converted to vector embedding
2. **Semantic Search** → Finds most similar passages in corpus
3. **Results** → Ranked by relevance, not just keyword matching

The system understands meaning, not just words. Asking "truth" will also find passages about "veracity", "facts", "reality", etc.

## Advanced: Full Question-Answering

For automatic answer generation (not just search), see `TRAINING_GUIDE.md` for:
- Full RAG pipeline with Ollama integration
- Fine-tuning your own Ramsey-specialized model
- Using with cloud APIs (GPT, Claude)

## Files Structure

```
data/
├── training/
│   ├── ramsey_corpus_full.txt       (complete corpus)
│   ├── ramsey_corpus_chunks.jsonl   (pre-chunked)
│   └── corpus_metadata.json         (stats)
└── vectorstore/                     (search database)

Scripts:
├── demo_ramsey_rag.py              (demo semantic search)
├── prepare_complete_corpus.py       (corpus preparation)
├── create_rag_system.py            (advanced RAG setup)
└── TRAINING_GUIDE.md               (full training guide)
```

## Performance

- **Vector Store Size**: ~50MB
- **Search Speed**: <1 second per query
- **Accuracy**: Semantic matching across 584K words
- **Cost**: FREE (all local)

## Corpus Breakdown

1. **Frank Ramsey: a sheer excess of powers** - 225,555 words (89% quality)
2. **Frank Ramsey and the Realistic Spirit** - 119,687 words (90% quality)
3. **Ramsey's legacy** - 88,564 words (88% quality)
4. **On Truth: Original Manuscripts** - 63,793 words (90% quality)
5. **Truth and Success** (OCR) - 44,765 words (89% quality)
6. **Shooting Star Biography** - 25,112 words (88% quality)
7. **General Propositions** (OCR) - 8,663 words (76% quality)
8. **Theories** - 7,558 words (90% quality)

## Troubleshooting

**"Vector store not found"**
- Run: `python demo_ramsey_rag.py` (creates it automatically)
- Or: `rm -rf data/vectorstore && python demo_ramsey_rag.py`

**"Corpus not found"**
- Run: `python prepare_complete_corpus.py` first

**Slow searches**
- First search is slow (loads model)
- Subsequent searches are fast (<1s)

## Next Steps

1. Try the demo with your own questions
2. Read `TRAINING_GUIDE.md` for advanced options
3. See `corpus_metadata.json` for detailed stats
4. Explore other training options (fine-tuning, etc.)

## Credits

Corpus created from OCR extraction + LLM correction pipeline:
- OCR: Tesseract at 800 DPI
- Correction: qwen2.5:14b via Ollama
- Validation: Semantic quality assessment
- Total processing time: ~6-8 hours
