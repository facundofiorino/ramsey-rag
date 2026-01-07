# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **LLM training project** that trains custom language models using **Ollama** on domain-specific documents from the `ramsey_data/` folder. The project follows a complete ML pipeline from data extraction through deployment and maintenance.

**Key Technologies:**
- **LLM Framework:** Ollama (configurable models)
- **Language:** Python 3.10.16
- **Source Control:** GitLab
- **Primary Use Case:** Train and deploy domain-specific LLMs for improved performance on Ramsey-related queries

**Documentation:** Comprehensive system specifications are in the `spec/` folder (7 detailed documents covering all aspects of the project).

## Development Environment

**Python Version:** 3.10.16

**Virtual Environment:**
- Activate: `source venv/bin/activate`
- Deactivate: `deactivate`

**Ollama:**
- Install: `curl -fsSL https://ollama.ai/install.sh | sh`
- Start: `ollama serve`
- List models: `ollama list`

## Dependencies

This project uses a comprehensive AI/ML stack including:

- **LangChain Ecosystem:** Core framework (`langchain`, `langchain-core`, `langgraph`) with integrations for Anthropic, OpenAI, Mistral, Ollama, and AWS
- **LLM Providers:** Anthropic Claude, OpenAI, Mistral, Ollama, Nomic
- **Vector Database:** ChromaDB for embeddings and semantic search
- **Web Scraping:** `scrapegraphai`, `scrapegraph_py`, `beautifulsoup4`, `playwright`, `undetected-playwright`
- **Document Processing:** PyPDF2, pdfplumber, python-docx, pdfminer.six, ebooklib for PDF/EPUB extraction
- **OCR:** Tesseract (v5.5.1), pytesseract, pdf2image, poppler, OpenCV for scanned document processing
- **NLP:** spaCy for natural language processing
- **ML/DL:** PyTorch, transformers, accelerate for deep learning
- **API Framework:** FastAPI, Flask for web services
- **Data Processing:** pandas, numpy for data manipulation
- **Geo/Maps:** folium, geopy for geographic visualization
- **Testing:** pytest, pytest-cov for testing and coverage

## Project Structure

```
ramsey_training/
├── src/                    # All source code
│   ├── processors/         # OCR and extraction processors
│   ├── extract_all.py      # Main extraction pipeline
│   ├── ocr_detector.py     # Automatic OCR quality detection
│   ├── ocr_semantic_validator.py  # Text quality validation
│   ├── ocr_post_processor.py      # Spell correction & cleanup
│   ├── ocr_optimizer_workflow.py  # LangGraph OCR optimizer
│   ├── models/             # Model training and evaluation
│   ├── deployment/         # Deployment and API code
│   └── utils/              # Shared utilities
├── ramsey_data/           # Training documents (PDFs, DOCX, etc.)
├── spec/                  # System documentation (7 detailed specs)
├── tests/                 # Test suite
├── configs/               # Configuration files
├── data/                  # Generated data (extracted/, processed/)
├── models/                # Trained models and checkpoints
├── logs/                  # Application logs
└── venv/                  # Python virtual environment
```

## Common Commands

**Setup:**
```bash
# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt  # Once created

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
```

**Data Pipeline:**
```bash
# Analyze document collection (detect OCR needs)
python src/ocr_detector.py ramsey_data/

# Extract data from documents (automatic OCR detection)
python src/extract_all.py \
  --input ramsey_data/ \
  --output data/extracted/ \
  --enable-ocr

# Extract with parallel processing (faster)
python src/extract_all.py \
  --input ramsey_data/ \
  --output data/extracted/ \
  --enable-ocr \
  --parallel \
  --max-workers 4

# Extract single file
python src/extract_all.py \
  --single-file ramsey_data/document.pdf \
  --output data/extracted/

# Manual OCR processing (for testing/debugging)
python src/processors/ocr_processor.py document.pdf poor

# Preprocess data
python src/data/preprocessing_pipeline.py \
  --input data/extracted/ \
  --output data/processed/ \
  --config configs/preprocessing_config.yaml

# Generate preprocessing statistics
python src/data/quality/generate_stats.py --input data/processed/
```

**Model Training:**
```bash
# Train model
python src/models/train.py \
  --config configs/model_config.yaml \
  --data data/processed/ \
  --output models/trained/

# Resume from checkpoint
python src/models/train.py \
  --config configs/model_config.yaml \
  --resume checkpoints/epoch_2.pt

# Hyperparameter search
python src/models/hyperparameter_search.py \
  --config configs/hyperparameter_search.yaml
```

**Model Evaluation:**
```bash
# Evaluate model
python src/models/evaluate.py \
  --model models/trained/best_model.pt \
  --config configs/evaluation_config.yaml

# Compare with baseline
python src/models/compare_baselines.py \
  --models models/trained/best_model.pt ollama:llama3.2:8b

# Error analysis
python src/models/error_analysis.py \
  --model models/trained/best_model.pt \
  --test-data data/processed/test.jsonl
```

**Deployment:**
```bash
# Export model to Ollama
python src/models/create_modelfile.py \
  --checkpoint models/trained/best_model.pt \
  --output models/ollama/

# Create Ollama model
ollama create ramsey-model -f models/ollama/Modelfile

# Test model
ollama run ramsey-model "Test prompt"

# Start API server
python src/deployment/api.py
# Or with uvicorn:
uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000

# Docker deployment
docker-compose -f docker/docker-compose.yml up -d
```

**Testing:**
```bash
pytest                           # Run all tests
pytest tests/test_file.py       # Run specific test file
pytest -v                        # Verbose output
pytest --cov                     # With coverage report
pytest tests/test_file.py::test_function_name  # Single test
```

## Architecture Overview

### ML Pipeline Stages

1. **Data Extraction** (spec/2-data-extraction.md)
   - Extract text from PDFs, EPUBs, and other document formats in `ramsey_data/`
   - **Automatic OCR Detection:** Analyzes text density to route documents
     - High quality (>100 chars/page): Native text extraction (fast)
     - Medium quality (50-100 chars/page): OCR at 600 DPI
     - Poor quality (<50 chars/page): OCR at 800 DPI with ultra preprocessing
   - **OCR Quality:** 90% improvement achieved (20% → 38-87% confidence)
   - **Tools:**
     - Native: PyPDF2, pdfplumber, ebooklib
     - OCR: Tesseract (LSTM), pytesseract, pdf2image, OpenCV
   - **Components:**
     - `src/ocr_detector.py` - Automatic quality detection
     - `src/processors/ocr_processor.py` - OCR engine with optimal settings
     - `src/extract_all.py` - Unified extraction pipeline
   - Output: Text files + metadata in `data/extracted/`

2. **Data Preprocessing** (spec/3-data-preprocessing.md)
   - Clean, normalize, and chunk extracted text
   - Feature engineering and quality filtering
   - Create train/validation/test splits (80/10/10)
   - Output: Processed JSONL files in `data/processed/`

3. **Model Training** (spec/4-model-training.md)
   - Fine-tune Ollama models (Llama, Mistral, Gemma, etc.)
   - Configurable model selection via `configs/model_config.yaml`
   - LoRA for efficient fine-tuning
   - Checkpoint management and experiment tracking

4. **Evaluation** (spec/5-evaluation-metrics.md)
   - Perplexity, BLEU, ROUGE, BERTScore
   - Human evaluation framework
   - Error analysis and baseline comparison
   - Statistical significance testing

5. **Deployment** (spec/6-deployment-strategy.md)
   - FastAPI REST API wrapping Ollama
   - Docker containerization
   - Load balancing and high availability
   - Monitoring with Prometheus/Grafana

6. **Maintenance** (spec/7-maintenance-plan.md)
   - Model drift detection and retraining
   - User feedback integration
   - Incident management
   - Continuous improvement

### Key Design Patterns

- **Configurable Ollama Models:** Model selection via YAML config, not hardcoded
- **Modular Pipeline:** Each stage is independent and can be run separately
- **Reproducibility:** All configs version-controlled, random seeds set
- **Monitoring First:** Comprehensive logging and metrics collection
- **GitLab Workflow:** Feature branches, merge requests, CI/CD pipeline

## Configuration Management

All project configurations are in `configs/`:
- `model_config.yaml` - Model training parameters, Ollama model selection
- `preprocessing_config.yaml` - Data preprocessing settings
- `evaluation_config.yaml` - Evaluation metrics and thresholds
- `hyperparameter_search.yaml` - Hyperparameter search space
- `model_registry.yaml` - Available Ollama models catalog

**Key Configuration Pattern:**
- Environment-specific configs (dev, staging, prod)
- Secrets via environment variables, never committed
- Version control all non-secret configs

## Development Workflow

**GitLab Workflow:**
1. Create feature branch from `main`
2. Implement changes in `src/`
3. Write tests in `tests/`
4. Update documentation if needed
5. Create merge request
6. CI/CD runs tests and linting
7. Code review and approval
8. Merge to `main`

**Package Management:**
```bash
pip freeze > requirements.txt    # Save current dependencies
pip install <package>            # Add new package
pip-audit                        # Security scan
```

**Python Path:** Ensure project root is in PYTHONPATH if using relative imports:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Detailed Documentation

Comprehensive specifications are in `spec/`:
1. **spec/1-project-overview.md** - Project goals, timeline, technology stack
2. **spec/2-data-extraction.md** - Document processing pipeline details
3. **spec/3-data-preprocessing.md** - Data cleaning and transformation
4. **spec/4-model-training.md** - Ollama integration and training procedures
5. **spec/5-evaluation-metrics.md** - Metrics, validation, and error analysis
6. **spec/6-deployment-strategy.md** - Production deployment and infrastructure
7. **spec/7-maintenance-plan.md** - Ongoing maintenance and updates

**When to consult specs:**
- Implementing new features: Read relevant spec first
- Debugging: Check spec for expected behavior
- Onboarding: Start with 1-project-overview.md
- Operations: Refer to 6-deployment-strategy.md and 7-maintenance-plan.md

## Important Notes

- **Ollama Models:** Always configurable via `configs/model_config.yaml`, never hardcoded
- **Data Privacy:** Documents in `ramsey_data/` may contain sensitive information
- **GitLab:** All source code stored in GitLab (not GitHub)
- **Testing:** Maintain >80% test coverage for all code in `src/`
- **Logging:** Use structured logging (JSON format) with appropriate log levels
