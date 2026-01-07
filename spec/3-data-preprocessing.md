# Data Preprocessing Steps

## 1. Overview

### 1.1 Purpose
- Transform raw extracted data into clean, structured format suitable for LLM training
- Standardize data representation across different document sources
- Enhance data quality and remove noise
- Prepare datasets for efficient model training

### 1.2 Preprocessing Goals
- Consistent text normalization
- Removal of irrelevant content and artifacts
- Structured data organization
- Optimal tokenization for Ollama models
- Creation of training, validation, and test splits

## 2. Data Cleaning

### 2.1 Text Normalization

#### 2.1.1 Character-Level Cleaning
- **Location:** src/data/cleaning/text_normalizer.py
- Unicode normalization (NFC/NFKC)
- Remove control characters and non-printable characters
- Fix encoding artifacts (e.g., â€™ → ')
- Standardize whitespace (spaces, tabs, newlines)
- Handle special characters and symbols

#### 2.1.2 Encoding Standardization
- Convert all text to UTF-8
- Handle mixed encodings within documents
- Detect and fix mojibake
- Preserve intentional special characters

#### 2.1.3 Case Handling
- Document case preservation strategy
- Lowercasing considerations for model
- Proper noun preservation
- Acronym handling

### 2.2 Noise Removal

#### 2.2.1 Boilerplate Content
- **Location:** src/data/cleaning/boilerplate_remover.py
- Remove headers and footers
- Strip page numbers
- Remove watermarks and stamps
- Eliminate repeated disclaimers

#### 2.2.2 Extraction Artifacts
- Remove PDF extraction errors (e.g., garbled text)
- Clean up table parsing artifacts
- Fix hyphenation issues from line breaks
- Merge incorrectly split words

#### 2.2.3 Irrelevant Content
- Remove advertisement text
- Strip navigation elements (from HTML sources)
- Eliminate boilerplate legal text
- Remove excessive whitespace and blank lines

### 2.3 Data Validation

#### 2.3.1 Quality Checks
- **Location:** src/data/cleaning/validator.py
- Minimum text length requirements
- Language detection and filtering
- Gibberish detection
- Duplicate content detection

#### 2.3.2 Structural Validation
- Verify document structure integrity
- Check for missing sections
- Validate metadata completeness
- Ensure consistent formatting

#### 2.3.3 Content Filtering
- Remove documents below quality threshold
- Filter out non-target language content
- Exclude overly short or long documents
- Remove duplicates and near-duplicates

## 3. Data Transformation

### 3.1 Text Segmentation

#### 3.1.1 Sentence Segmentation
- **Tool:** spaCy for sentence boundary detection
- **Location:** src/data/transform/segmenter.py
- Handle edge cases (abbreviations, ellipsis)
- Preserve sentence context
- Maintain document structure markers

#### 3.1.2 Paragraph Detection
- Identify paragraph boundaries
- Preserve semantic groupings
- Handle multi-level document structures

#### 3.1.3 Section Detection
- Extract document sections and subsections
- Preserve hierarchical relationships
- Label sections with metadata

### 3.2 Chunking Strategy

#### 3.2.1 Chunk Size Determination
- **Tool:** semchunk for semantic chunking
- Target chunk size for Ollama context window
- Balance between context and efficiency
- Overlap strategy for continuity

#### 3.2.2 Semantic Chunking
- **Location:** src/data/transform/chunker.py
- Chunk by semantic boundaries (paragraphs, sections)
- Preserve complete thoughts and concepts
- Maintain context across chunks

#### 3.2.3 Metadata Preservation
- Attach source document ID to each chunk
- Preserve section/page information
- Include chunk position in original document

### 3.3 Text Enrichment

#### 3.3.1 Metadata Augmentation
- **Location:** src/data/transform/enricher.py
- Add document type labels
- Include temporal information
- Add topic/category tags (if available)
- Assign quality scores

#### 3.3.2 Entity Recognition (Optional)
- **Tool:** spaCy NER
- Extract and label named entities
- Normalize entity mentions
- Create entity index for reference

#### 3.3.3 Relationship Extraction (Optional)
- Document-to-document relationships
- Section-to-section links
- Citation and reference mapping

## 4. Feature Engineering

### 4.1 Statistical Features

#### 4.1.1 Document-Level Features
- **Location:** src/data/features/statistical.py
- Document length (words, tokens, characters)
- Sentence count and average length
- Vocabulary richness (unique tokens ratio)
- Readability scores (Flesch-Kincaid, etc.)

#### 4.1.2 Chunk-Level Features
- Chunk position in document (normalized)
- Relative importance score
- Information density metrics
- Topic coherence scores

### 4.2 Linguistic Features

#### 4.2.1 Part-of-Speech Statistics
- **Tool:** spaCy
- POS tag distributions
- Syntactic complexity measures
- Dependency parsing depth

#### 4.2.2 Semantic Features
- **Tool:** Sentence transformers or spaCy embeddings
- Semantic similarity to document centroid
- Topic distribution (if using topic modeling)
- Domain-specific terminology density

### 4.3 Token-Level Preprocessing

#### 4.3.1 Tokenization Strategy
- **Consideration:** Ollama model's tokenizer compatibility
- Pre-tokenization for analysis
- Token count estimation
- Special token handling

#### 4.3.2 Vocabulary Analysis
- Build vocabulary statistics
- Identify rare and common tokens
- Domain-specific terminology extraction
- Create vocabulary mapping

### 4.4 Feature Selection
- Identify most relevant features for training
- Correlation analysis
- Feature importance ranking
- Dimensionality reduction (if applicable)

## 5. Data Splitting

### 5.1 Split Strategy

#### 5.1.1 Split Ratios
- **Training Set:** 80% of data
- **Validation Set:** 10% of data
- **Test Set:** 10% of data

#### 5.1.2 Splitting Method
- **Location:** src/data/split/splitter.py
- Random splitting with stratification
- Ensure representation across document types
- Maintain temporal distribution (if time-sensitive)
- Prevent data leakage

### 5.2 Document-Level Splitting
- Split at document level (not chunk level)
- Prevent information leakage between splits
- All chunks from same document in same split
- Validate split distributions

### 5.3 Stratification Considerations
- Stratify by document type
- Stratify by document length categories
- Stratify by source (if multiple sources)
- Ensure balanced representation

### 5.4 Validation of Splits
- Verify no data leakage
- Check distribution similarity across splits
- Validate size ratios
- Statistical comparison of splits

## 6. Output Data Format

### 6.1 Processed Data Structure
```python
{
    "chunk_id": "doc123_chunk_005",
    "document_id": "doc123",
    "source_file": "ramsey_data/original.pdf",
    "text": "preprocessed chunk content",
    "metadata": {
        "document_type": "pdf",
        "section": "Chapter 3",
        "page_range": [5, 6],
        "position": 0.15,  # Relative position in document
        "word_count": 256,
        "token_count": 312,
        "language": "en"
    },
    "features": {
        "readability_score": 65.2,
        "vocabulary_richness": 0.68,
        "avg_sentence_length": 18.5
    },
    "split": "train",  # or "validation" or "test"
    "preprocessing_version": "1.0"
}
```

### 6.2 Storage Format
- **Format:** JSONL (one JSON object per line)
- **Location:**
  - data/processed/train.jsonl
  - data/processed/validation.jsonl
  - data/processed/test.jsonl
- Compressed with gzip for efficiency
- Separate files for each split

### 6.3 Supporting Artifacts
- **Vocabulary:** data/processed/vocabulary.json
- **Statistics:** data/processed/statistics.json
- **Split Manifest:** data/processed/split_manifest.json
- **Preprocessing Config:** data/processed/preprocessing_config.yaml

## 7. Pipeline Implementation

### 7.1 Pipeline Architecture
```
Extracted Data (data/extracted/)
    ↓
Data Cleaning
    ↓
Text Normalization
    ↓
Segmentation & Chunking
    ↓
Feature Engineering
    ↓
Quality Filtering
    ↓
Data Splitting
    ↓
Processed Data (data/processed/)
```

### 7.2 Preprocessing Modules

#### 7.2.1 Main Pipeline Controller
- **Location:** src/data/preprocessing_pipeline.py
- Orchestrates all preprocessing steps
- Handles checkpointing for resumability
- Manages parallel processing
- Generates preprocessing report

#### 7.2.2 Configuration Management
- **Location:** configs/preprocessing_config.yaml
- Configurable parameters for each step
- Version control for reproducibility
- Environment-specific overrides

#### 7.2.3 Monitoring and Logging
- Progress tracking with tqdm
- Detailed logging at each stage
- Error handling and recovery
- Statistics collection

## 8. Quality Assurance

### 8.1 Preprocessing Quality Metrics
- Data loss rate (target: <5%)
- Chunk quality score distribution
- Text cleanliness metrics
- Feature distribution validation

### 8.2 Automated Checks
- **Location:** src/data/quality/qa_checks.py
- Schema validation
- Range checks for numerical features
- Text quality heuristics
- Duplicate detection

### 8.3 Manual Review Process
- Sample review workflow
- Quality issue tracking
- Feedback incorporation
- Continuous improvement

## 9. Performance Optimization

### 9.1 Parallel Processing
- **Tool:** mpire for parallel execution
- Process documents in parallel
- Optimal worker count configuration
- Memory-efficient batching

### 9.2 Caching and Checkpointing
- Cache intermediate results
- Checkpoint after major stages
- Resume from checkpoint on failure
- Version-based cache invalidation

### 9.3 Resource Management
- Memory profiling and optimization
- Disk I/O optimization
- Processing queue management
- Garbage collection tuning

## 10. Execution Commands

### 10.1 Full Preprocessing Pipeline
```bash
python src/data/preprocessing_pipeline.py \
  --input data/extracted/ \
  --output data/processed/ \
  --config configs/preprocessing_config.yaml
```

### 10.2 Individual Steps
```bash
# Data cleaning only
python src/data/cleaning/clean_data.py --input data/extracted/ --output data/cleaned/

# Chunking only
python src/data/transform/chunker.py --input data/cleaned/ --output data/chunked/

# Data splitting
python src/data/split/splitter.py --input data/chunked/ --output data/processed/
```

### 10.3 Validation and Statistics
```bash
# Generate preprocessing statistics
python src/data/quality/generate_stats.py --input data/processed/ --output reports/preprocessing_stats.html

# Run quality checks
python src/data/quality/qa_checks.py --input data/processed/
```

## 11. Testing

### 11.1 Unit Tests
- **Location:** tests/data/test_preprocessing.py
- Test each preprocessing function independently
- Mock data generation
- Edge case coverage

### 11.2 Integration Tests
- **Location:** tests/data/test_preprocessing_pipeline.py
- End-to-end pipeline testing
- Sample document processing
- Output validation

### 11.3 Regression Tests
- Detect preprocessing changes
- Validate consistency across versions
- Performance regression detection

## 12. Monitoring and Debugging

### 12.1 Logging Strategy
- **Location:** logs/preprocessing/
- Debug, info, warning, error levels
- Structured logging with JSON format
- Timestamped log files

### 12.2 Progress Tracking
- Real-time progress bars (tqdm)
- Stage completion notifications
- Estimated time remaining
- Throughput metrics

### 12.3 Error Handling
- Graceful degradation on errors
- Error categorization and logging
- Retry mechanisms for transient errors
- Error report generation

## 13. Reproducibility

### 13.1 Version Control
- Track preprocessing code versions
- Version configuration files
- Document dependency versions
- Git commit hashes in metadata

### 13.2 Deterministic Processing
- Set random seeds for splitting
- Deterministic ordering where possible
- Document any non-deterministic steps
- Reproducible environment setup

### 13.3 Audit Trail
- Log all preprocessing decisions
- Record data transformations
- Track data lineage
- Generate reproducibility report

## 14. Future Enhancements

### 14.1 Planned Improvements
- Advanced deduplication using embeddings
- Active learning for quality filtering
- Automated data augmentation
- Real-time preprocessing for streaming data

### 14.2 Advanced Feature Engineering
- Domain-specific feature extraction
- Transfer learning for feature generation
- Graph-based document relationships
- Multi-modal feature integration (if applicable)
