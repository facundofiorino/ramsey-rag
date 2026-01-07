# Frank Ramsey LLM Training Guide

## Current Status

**Corpus Prepared**: 167,165 words (will be ~576K once missing files are located)
**Quality**: 85-90% average
**Location**: `data/training/`

## Training Data Files

- `ramsey_corpus_full.txt` - Complete consolidated corpus
- `ramsey_corpus_chunks.jsonl` - Pre-chunked for training (82 chunks @ 2048 words each)
- `corpus_metadata.json` - Statistics and documentation

## Next Steps for LLM Training

### Option 1: Fine-tune with Ollama (Local, Free)

**Best for**: Creating a local Frank Ramsey-specialized model

```bash
# 1. Create a Modelfile
cat > Modelfile << 'EOF'
FROM llama3.2:8b

# Set temperature for philosophical reasoning
PARAMETER temperature 0.7
PARAMETER top_p 0.9

# System prompt for Ramsey-style responses
SYSTEM """You are an AI assistant trained on the philosophical works of Frank Ramsey.
You provide insights on logic, mathematics, epistemology, and pragmatism in the style
of Ramsey's clear, analytical thinking. You draw from his work on truth, probability,
causality, and the foundations of mathematics."""

EOF

# 2. Create the model with your corpus
ollama create ramsey-philosopher -f Modelfile

# 3. Fine-tune by providing context (Ollama doesn't support true fine-tuning yet)
# Instead, use the corpus in a RAG system (see Option 4)
```

### Option 2: Fine-tune with Hugging Face Transformers

**Best for**: Creating a production-ready model

```bash
# Install dependencies
pip install transformers datasets accelerate torch

# Use the training script below
python train_ramsey_model.py
```

**Training Script** (`train_ramsey_model.py`):

```python
#!/usr/bin/env python3
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch

# Load your corpus
dataset = load_dataset('json', data_files='data/training/ramsey_corpus_chunks.jsonl')

# Choose base model
model_name = "meta-llama/Llama-3.2-8B"  # or "gpt2", "mistralai/Mistral-7B-v0.1"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./ramsey-model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    learning_rate=2e-5,
    warmup_steps=100,
    logging_steps=100,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM, not masked LM
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    data_collator=data_collator,
)

# Train
trainer.train()

# Save
trainer.save_model("./ramsey-philosopher-final")
tokenizer.save_pretrained("./ramsey-philosopher-final")
```

**Estimated Time**: 2-6 hours on GPU, 12-24 hours on CPU
**Hardware**: Recommended 16GB+ RAM, GPU preferred

### Option 3: Fine-tune GPT-style Model (OpenAI/Anthropic)

**Best for**: If you want cloud-hosted fine-tuning

**OpenAI Fine-tuning**:
```bash
# Convert to OpenAI format
python convert_to_openai_format.py

# Upload and fine-tune
openai api fine_tunes.create \
  -t "ramsey_corpus_openai.jsonl" \
  -m gpt-4o-mini \
  --suffix "ramsey-philosopher"
```

**Cost**: ~$8-40 depending on tokens (OpenAI charges per token)

### Option 4: RAG (Retrieval-Augmented Generation) System

**Best for**: Using existing models with your corpus as context

```bash
# Install dependencies
pip install langchain chromadb sentence-transformers

# Create vector store
python create_rag_system.py
```

**RAG Script** (`create_rag_system.py`):

```python
#!/usr/bin/env python3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from pathlib import Path

# Load corpus
loader = TextLoader("data/training/ramsey_corpus_full.txt")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="data/vectorstore"
)

print(f"✓ Vector store created with {len(texts)} chunks")
print(f"  Location: data/vectorstore/")
```

**Then query it**:

```python
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# Connect to Ollama
llm = Ollama(model="llama3.2:8b")

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Ask questions
response = qa_chain.run("What did Ramsey think about truth?")
print(response)
```

**Time**: 30 minutes setup, instant queries
**Cost**: Free (using local Ollama)

## Recommended Approach

Given you prefer **free solutions** and already have **Ollama running**:

### Recommended: RAG System (Option 4)

**Why**:
- Free and fast
- No GPU/expensive training needed
- Works with any LLM (Ollama, GPT, Claude)
- Can update corpus easily
- Production-ready immediately

**Steps**:
1. Install LangChain: `pip install langchain chromadb sentence-transformers`
2. Run the RAG script above
3. Query with any question about Ramsey
4. Responses will cite directly from your corpus

### Alternative: Full Fine-tuning (Option 2)

**If you want a standalone model**:
- Requires GPU (M1/M2 Mac works, or cloud GPU)
- Takes 6-12 hours
- Creates completely independent model
- Better for deployment at scale

## Validation & Testing

After training, test with questions like:

```python
questions = [
    "What is Frank Ramsey's view on truth?",
    "How did Ramsey approach the foundations of mathematics?",
    "What is the Ramsey-Lewis theory?",
    "Explain Ramsey's pragmatist epistemology",
    "What did Ramsey think about causality?"
]

for q in questions:
    response = model.generate(q)
    print(f"Q: {q}\nA: {response}\n")
```

## Next Immediate Action

**Choose your approach**, then:

```bash
# For RAG (recommended for free/fast):
pip install langchain chromadb sentence-transformers
python create_rag_system.py

# For full fine-tuning:
pip install transformers datasets accelerate torch
python train_ramsey_model.py

# For Ollama simple integration:
# Just use ramsey_corpus_full.txt as context in prompts
```

## Resources

- **LangChain docs**: https://python.langchain.com/docs/
- **Hugging Face training**: https://huggingface.co/docs/transformers/training
- **Ollama docs**: https://github.com/ollama/ollama/blob/main/docs/modelfile.md
- **RAG tutorial**: https://python.langchain.com/docs/use_cases/question_answering/

## Cost Comparison

| Approach | Time | Cost | Hardware | Quality |
|----------|------|------|----------|---------|
| RAG System | 30 min | Free | Any | Excellent (cites sources) |
| Ollama Fine-tune | N/A* | Free | Any | Good (limited fine-tuning) |
| HF Transformers | 6-12 hrs | Free** | GPU recommended | Excellent |
| OpenAI Fine-tune | 2-4 hrs | $8-40 | Cloud | Excellent |

\* Ollama doesn't support true fine-tuning yet, only prompt engineering
\*\* Free but may need cloud GPU ($0.50-2/hr) if no local GPU

## Missing Files

NOTE: The preparation script found only 4 of 7 expected files. You mentioned ~576K words,
but we only have ~167K. Missing files:

- Foundations of Mathematics and other Logical Essays
- Frank Ramsey: a sheer excess of powers
- Theories (Ramsey, Frank)

These files should exist based on the extraction logs. Let me know if you want me to locate them.
