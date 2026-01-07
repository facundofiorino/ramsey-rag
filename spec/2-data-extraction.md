# Data Extraction Process

## 1. Overview

### 1.1 Purpose
- Extract structured and unstructured data from documents in ramsey_data folder
- Transform raw documents into machine-readable format
- Ensure data integrity and completeness during extraction

### 1.2 Data Extraction Goals
- Comprehensive extraction of all relevant information
- Maintain document structure and context
- Preserve metadata and document relationships
- Handle multiple document formats uniformly

## 2. Data Sources

### 2.1 Source Location
- **Primary Data Source:** ramsey_data/ folder
- Directory structure and organization
- Document naming conventions

### 2.2 Document Inventory

**Current Ramsey Data Collection:**
- **Total Documents:** 8 files
- **Total Size:** ~58 MB
- **Total Pages:** ~1,341 pages

**Document Types Breakdown:**
- **PDF documents:** 7 files
  - Text-based PDFs: 5 files (71%) - 1,183 pages
  - Scanned/Image PDFs: 2 files (29%) - 158 pages **[REQUIRES OCR]**
- **EPUB documents:** 1 file (173 KB)
- Word documents (DOCX): 0 files
- HTML/Web content: 0 files

**OCR Requirements:**
- **29% of PDFs require OCR processing** (2 out of 7 files)
- Scanned documents identified by text density < 100 chars/page
- Estimated OCR processing time: ~2-5 minutes per document

### 2.3 Document Characteristics

**Ramsey Data Analysis:**
- **Average PDF size:** 8.3 MB (range: 1.07 MB to 28.5 MB)
- **Average document length:** 192 pages (range: 14 to 552 pages)
- **Language:** English (academic/philosophical content)
- **Document structure:**
  - Structured academic books with chapters, sections, references
  - Original manuscripts and papers
  - Mix of modern digitally-created PDFs and older scanned works
- **Special considerations:**
  - Academic citations and footnotes
  - Mathematical/logical notation in some documents
  - Older documents (1927-1929 manuscripts) may have degraded scan quality

### 2.4 Data Volume Estimates

**Current Collection:**
- **Total data size:** 58.2 MB
- **Total pages:** ~1,341 pages
- **Estimated tokens:** ~500K-800K tokens (assuming ~300-600 tokens per page)
- **Character count:** ~2-3 million characters

**Processing Estimates:**
- Text-based extraction: ~5-10 seconds per document
- OCR processing: ~2-5 minutes per scanned document
- Total extraction time: ~15-20 minutes for entire collection

**Expected Growth:**
- Additional documents may be added over time
- Maintain extraction scripts for incremental updates

## 3. Extraction Methods

### 3.1 PDF Extraction

PDF extraction requires a **hybrid approach** to handle both text-based and scanned documents:

#### 3.1.1 Text-Based PDF Extraction
- **Tools:** PyPDF2, pdfplumber, pdfminer.six
- **Use case:** Modern, digitally-created PDFs with embedded text
- **Detection:** Text density > 100 characters per page
- **Strategy:**
  1. Attempt extraction with pdfplumber (best for complex layouts)
  2. Fallback to PyPDF2 for simple PDFs
  3. Use pdfminer.six for low-level analysis if needed
- **Table extraction:** pdfplumber's table detection
- **Image handling:** Skip or extract metadata only

#### 3.1.2 Scanned PDF Extraction (OCR Required)
- **Tools:** Tesseract OCR, pytesseract, pdf2image, pypdfium2
- **Use case:** Scanned documents, old manuscripts, image-based PDFs
- **Detection:** Text density < 100 characters per page
- **Strategy:**
  1. Convert PDF pages to images (pdf2image or pypdfium2)
  2. Preprocess images (deskew, denoise, contrast enhancement)
  3. Apply Tesseract OCR with appropriate language models
  4. Post-process OCR output (spell checking, confidence filtering)
  5. Reconstruct document structure

#### 3.1.3 Hybrid PDF Extraction
- **Use case:** PDFs with mixed content (text + scanned pages)
- **Strategy:**
  1. Analyze each page individually for text density
  2. Extract text from text-based pages
  3. Apply OCR to scanned pages
  4. Merge results maintaining page order
  5. Flag pages with low OCR confidence for review

### 3.2 Word Document Extraction
- **Tool:** python-docx
- Text and formatting extraction
- Table and embedded object handling
- Metadata extraction

### 3.3 HTML/Web Content Extraction
- **Tools:** BeautifulSoup4, playwright
- HTML parsing strategies
- JavaScript-rendered content handling
- Link and reference extraction

### 3.4 Plain Text Extraction
- Encoding detection and handling
- Line break and formatting preservation
- Special character handling

### 3.5 OCR Processing Pipeline

#### 3.5.1 OCR Detection
- **Location:** src/data/ocr_detector.py
- Automatic detection of scanned PDFs
- Calculate text density per page
- Flag documents/pages requiring OCR
- **Threshold:** < 100 characters per page

#### 3.5.2 Image Preprocessing
- **Location:** src/data/image_preprocessor.py
- **Deskewing:** Correct rotated scans
- **Denoising:** Remove artifacts and noise
- **Contrast Enhancement:** Improve text clarity
- **Binarization:** Convert to black/white for better OCR
- **Resolution:** Ensure 300+ DPI for optimal OCR

#### 3.5.3 OCR Execution
- **Tool:** Tesseract OCR 4.x or 5.x
- **Language:** English (eng) with optional specialized models
- **Engine Mode:** LSTM neural network (--oem 1)
- **Page Segmentation:** Auto detect with orientation (--psm 3)
- **Confidence Scoring:** Track per-word and per-page confidence
- **Output Format:** Plain text with optional hOCR for positioning

#### 3.5.4 OCR Post-Processing
- **Confidence Filtering:** Flag low-confidence words (< 60%)
- **Spell Checking:** Correct common OCR errors
- **Pattern Recognition:** Fix common substitutions (0/O, 1/l, etc.)
- **Structure Recovery:** Identify paragraphs, headings, footnotes
- **Quality Metrics:** Calculate OCR quality score

### 3.6 Fallback Strategies
- Handling corrupted or unreadable documents
- Multi-tool approach for difficult documents
- Try alternative OCR engines (EasyOCR, PaddleOCR) if Tesseract fails
- Manual review process for edge cases
- Document extraction failures for future improvement

## 4. Tools and Technologies Used

### 4.1 Python Libraries

#### 4.1.1 Text-Based Extraction
- **PyPDF2** (v3.0.1): Basic PDF text extraction
- **pdfplumber** (v0.11.0): Advanced PDF parsing with table support
- **pdfminer.six** (v20231228): Low-level PDF analysis
- **python-docx** (v1.1.0): Word document processing
- **BeautifulSoup4** (v4.12.3): HTML parsing
- **playwright** (v1.55.0): Browser automation for dynamic content

#### 4.1.2 OCR and Image Processing
- **pytesseract** (v0.3.10): Python wrapper for Tesseract OCR
- **pdf2image** (v1.16.3): Convert PDF pages to PIL images
- **pypdfium2** (v4.30.0): Alternative PDF to image converter (faster)
- **Pillow (PIL)** (v10.4.0): Image processing and manipulation
- **opencv-python** (optional): Advanced image preprocessing
- **numpy** (v1.26.4): Numerical operations for image arrays

**Note:** Tesseract OCR must be installed separately on the system:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# macOS
brew install tesseract

# Windows
# Download installer from GitHub: github.com/UB-Mannheim/tesseract/wiki
```

### 4.2 Supporting Libraries
- **chardet** (v5.2.0): Character encoding detection
- **lxml** (v6.0.2): Fast XML/HTML processing
- **pandas** (v2.1.4): Data structuring and manipulation

### 4.3 Custom Extraction Utilities
- **Document type detection** utility (src/data/format_detector.py)
- **OCR detection** utility (src/data/ocr_detector.py)
- **Image preprocessor** (src/data/image_preprocessor.py)
- **Batch processing** framework with parallel OCR support
- **Error handling** and logging system
- **Progress tracking** for large-scale extraction with OCR time estimates

## 5. Extraction Pipeline Architecture

### 5.1 Pipeline Stages
```
Document Discovery
    ↓
Format Detection
    ↓
Text Density Analysis (OCR Detection)
    ↓
Extraction Strategy Selection
    ├─→ Text-Based Path → Direct Extraction
    └─→ OCR Path → Image Conversion → Preprocessing → OCR → Post-Processing
    ↓
Content Extraction
    ↓
Metadata Extraction
    ↓
Quality Validation
    ↓
Storage
```

### 5.2 Document Discovery Module
- **Location:** src/data/discovery.py
- Recursive directory traversal
- File filtering and selection
- Duplicate detection
- **Output:** List of document paths with metadata

### 5.3 Format Detection Module
- **Location:** src/data/format_detector.py
- MIME type detection
- Magic number validation
- Extension verification
- **Output:** Document type classification

### 5.3.1 OCR Detection Module
- **Location:** src/data/ocr_detector.py
- Sample first 3 pages for text density analysis
- Calculate characters per page metric
- Classify as text-based (>100 chars/page) or scanned (<100 chars/page)
- Generate OCR requirements report
- **Output:** Boolean flag and text density score per document/page

### 5.4 Extraction Dispatcher
- **Location:** src/data/extractor.py
- Route documents to appropriate extractor
- Parallel processing coordination
- Error handling and retry logic
- **Output:** Raw extracted content

### 5.5 Content Processors
- **Location:** src/data/processors/
- PDF processor: `pdf_processor.py` (handles text-based PDFs)
- OCR processor: `ocr_processor.py` (handles scanned PDFs)
- DOCX processor: `docx_processor.py`
- HTML processor: `html_processor.py`
- Text processor: `text_processor.py`
- **Output:** Structured content objects

### 5.5.1 OCR Processor Details
- **Location:** src/data/processors/ocr_processor.py
- **Image Conversion:** Convert PDF pages to images (300 DPI minimum)
- **Preprocessing:** Apply image enhancement techniques
- **OCR Engine:** Tesseract with English language model
- **Confidence Tracking:** Record OCR confidence per word/page
- **Error Handling:** Retry failed pages with different settings
- **Output:** Extracted text with confidence scores and quality metrics

### 5.6 Metadata Extraction
- **Location:** src/data/metadata.py
- Document creation/modification dates
- Author information (if available)
- Document title and subject
- Page/word counts
- **Output:** Metadata dictionary

### 5.7 Validation Module
- **Location:** src/data/validator.py
- Completeness checks (all text extracted)
- Encoding validation
- Structure verification
- Quality metrics calculation
- **Output:** Validation report

## 6. Data Quality Assurance

### 6.1 Extraction Accuracy

#### 6.1.1 Text-Based Extraction
- Text accuracy verification methods
- Sampling strategy for manual review
- Automated accuracy metrics
- Comparison with ground truth (if available)

#### 6.1.2 OCR Accuracy
- **Per-Word Confidence:** Track Tesseract confidence scores
  - High confidence: >80%
  - Medium confidence: 60-80%
  - Low confidence: <60% (flag for review)
- **Per-Page Quality Score:** Average confidence across page
- **Character Error Rate (CER):** Estimated error rate for OCR
- **Word Error Rate (WER):** Estimated word-level errors
- **Manual Validation:** Sample 5-10% of OCR pages for human review
- **Confidence Thresholds:**
  - Accept: Average page confidence >70%
  - Review: Average page confidence 50-70%
  - Reject/Retry: Average page confidence <50%

### 6.2 Completeness Checks
- Verify all documents processed
- Check for missing pages or sections
- Validate metadata completeness
- Track extraction failures

### 6.3 Consistency Validation
- Encoding consistency across documents
- Format standardization checks
- Naming convention adherence
- Structural consistency validation

### 6.4 Error Handling
- **Error Types:**
  - Corrupted documents
  - Unsupported formats
  - Encoding issues
  - Missing permissions
  - Extraction timeouts
  - **OCR-specific errors:**
    - Poor image quality (low resolution, skewed, noisy)
    - Tesseract not installed or misconfigured
    - Out of memory during image processing
    - Unsupported image formats
    - Low OCR confidence across entire document

- **Handling Strategy:**
  - Log all errors with context
  - Retry with alternative methods
  - For OCR failures:
    - Retry with different image preprocessing
    - Try alternative DPI settings (300, 400, 600)
    - Attempt alternative OCR engines (EasyOCR, PaddleOCR)
    - Flag for manual review if all methods fail
  - Document error patterns for future improvements

### 6.5 Quality Metrics
- **Overall Extraction:**
  - Extraction success rate (target: >95%)
  - Average processing time per document
  - Error rate by document type
  - Data completeness score

- **OCR-Specific Metrics:**
  - OCR success rate (target: >90%)
  - Average OCR confidence score (target: >75%)
  - Low-confidence page percentage (target: <10%)
  - OCR processing time per page
  - Image preprocessing success rate
  - Character/word error rates (estimated)

## 7. Output Format

### 7.1 Extracted Data Structure
```python
{
    "document_id": "unique_identifier",
    "source_file": "path/to/original.pdf",
    "document_type": "pdf",
    "extracted_date": "2024-11-24T10:00:00",
    "content": {
        "text": "full extracted text",
        "pages": [
            {"page_num": 1, "text": "page 1 content"},
            {"page_num": 2, "text": "page 2 content"}
        ],
        "tables": [...],  # if applicable
        "images": [...]   # if applicable
    },
    "metadata": {
        "title": "Document Title",
        "author": "Author Name",
        "created": "2024-01-01",
        "pages": 10,
        "word_count": 5000,
        "language": "en"
    },
    "extraction_info": {
        "tool": "pdfplumber",  // or "tesseract_ocr"
        "version": "0.11.0",
        "processing_time": 2.5,
        "success": true,
        "warnings": [],
        "extraction_method": "text",  // or "ocr" or "hybrid"
        "ocr_required": false,
        "ocr_confidence": null  // or average confidence score if OCR used
    }
}
}
```

### 7.2 Storage Format
- **Format:** JSONL (JSON Lines) for efficient processing
- **Location:** data/extracted/
- One JSON object per line
- Compressed with gzip for storage efficiency

### 7.3 Backup and Versioning
- Original documents preserved in ramsey_data/
- Extracted data versioned by timestamp
- Extraction logs retained for audit trail

## 8. Performance Optimization

### 8.1 Parallel Processing
- Multi-threaded document processing
- **Tool:** mpire for parallel execution
- Optimal worker count based on CPU cores
- Memory management for large documents

### 8.2 Caching Strategy
- Cache extraction results
- Skip re-extraction if unchanged
- Hash-based change detection

### 8.3 Resource Management
- Memory limits per worker
- Disk I/O optimization
- Processing queue management

## 9. Monitoring and Logging

### 9.1 Extraction Logs
- **Location:** logs/extraction/
- Timestamp and document identification
- Processing duration
- Warnings and errors
- Quality metrics

### 9.2 Progress Tracking
- Real-time progress dashboard (optional)
- Estimated completion time
- Success/failure statistics
- Bottleneck identification

### 9.3 Alerting
- Notification on high error rates
- Alert on processing failures
- Resource utilization warnings

## 10. Testing and Validation

### 10.1 Unit Tests
- **Location:** tests/data/test_extraction.py
- Test each extractor independently
- Mock document samples
- Edge case handling

### 10.2 Integration Tests
- End-to-end extraction pipeline tests
- Real document sample testing
- Performance benchmarking

### 10.3 Sample Documents
- **Location:** tests/data/samples/
- Representative samples of each document type
- Edge cases and challenging documents
- Ground truth for validation

## 11. Execution Commands

### 11.1 Analyze Documents (OCR Detection)
```bash
# Analyze documents to identify OCR requirements
python src/data/analyze_documents.py --input ramsey_data/ --output reports/document_analysis.json

# Generate OCR requirements report
python src/data/ocr_detector.py --input ramsey_data/ --report
```

### 11.2 Full Extraction with OCR Support
```bash
# Extract all documents (auto-detect OCR needs)
python src/data/extract_all.py --input ramsey_data/ --output data/extracted/ --enable-ocr

# Extract with specific DPI for OCR
python src/data/extract_all.py --input ramsey_data/ --output data/extracted/ --enable-ocr --ocr-dpi 400
```

### 11.3 Single Document Extraction
```bash
# Extract single text-based PDF
python src/data/extract_single.py --file ramsey_data/document.pdf --output data/extracted/

# Extract single scanned PDF with OCR
python src/data/extract_single.py --file ramsey_data/scanned.pdf --output data/extracted/ --force-ocr

# Extract with OCR confidence reporting
python src/data/extract_single.py --file ramsey_data/scanned.pdf --output data/extracted/ --force-ocr --report-confidence
```

### 11.4 OCR-Only Processing
```bash
# Process only scanned documents
python src/data/extract_all.py --input ramsey_data/ --output data/extracted/ --ocr-only

# Retry failed OCR extractions
python src/data/retry_ocr.py --input data/extracted/ --failed-only
```

### 11.5 Re-extraction with Force
```bash
python src/data/extract_all.py --input ramsey_data/ --force --output data/extracted/ --enable-ocr
```

### 11.6 Extraction with Validation
```bash
# Validate extraction quality including OCR confidence
python src/data/extract_all.py --input ramsey_data/ --validate --output data/extracted/ --enable-ocr

# Generate quality report
python src/data/quality_report.py --input data/extracted/ --output reports/extraction_quality.html
```

## 12. Troubleshooting

### 12.1 Common Issues

#### 12.1.1 General Extraction Issues
- **Encoding errors:** Solution approaches
- **Memory overflow:** Resource management strategies
- **Timeout errors:** Processing optimization
- **Corrupted documents:** Recovery procedures

#### 12.1.2 OCR-Specific Issues

**Problem: Tesseract not found**
```bash
# Solution: Install Tesseract
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# Verify installation
tesseract --version
```

**Problem: Poor OCR quality / Low confidence scores**
```bash
# Solution 1: Increase DPI
python src/data/extract_single.py --file document.pdf --force-ocr --ocr-dpi 600

# Solution 2: Apply image preprocessing
python src/data/extract_single.py --file document.pdf --force-ocr --preprocess aggressive

# Solution 3: Try different page segmentation mode
python src/data/extract_single.py --file document.pdf --force-ocr --psm 6
```

**Problem: OCR taking too long**
```bash
# Solution 1: Reduce DPI (trade quality for speed)
python src/data/extract_all.py --input ramsey_data/ --enable-ocr --ocr-dpi 200

# Solution 2: Use faster PDF-to-image converter
python src/data/extract_all.py --input ramsey_data/ --enable-ocr --use-pypdfium2

# Solution 3: Parallel processing
python src/data/extract_all.py --input ramsey_data/ --enable-ocr --workers 4
```

**Problem: Out of memory during OCR**
```bash
# Solution: Process one page at a time
python src/data/extract_all.py --input ramsey_data/ --enable-ocr --sequential-pages
```

**Problem: Skewed or rotated scans**
```bash
# Solution: Enable automatic deskewing
python src/data/extract_single.py --file document.pdf --force-ocr --deskew
```

### 12.2 Debug Mode
- Enable verbose logging
- Step-by-step extraction with checkpoints
- Intermediate result inspection (save preprocessed images)
- OCR confidence visualization
```bash
# Debug mode with image saving
python src/data/extract_single.py \
  --file document.pdf \
  --force-ocr \
  --debug \
  --save-images \
  --output-dir debug/
```

## 13. Future Enhancements

### 13.1 Planned OCR Improvements
- **Multiple OCR Engines:** Ensemble approach using Tesseract + EasyOCR + PaddleOCR
- **Deep Learning OCR:** Integration with modern OCR models (TrOCR, DocTR)
- **Layout Analysis:** Advanced document layout understanding (LayoutLM)
- **Handwriting Recognition:** Support for handwritten manuscripts
- **Quality-Based Routing:** Automatically select best OCR engine per document
- **Active Learning:** Improve OCR on failure cases through human feedback

### 13.2 Advanced Processing
- **Advanced table structure recognition** in scanned documents
- **Mathematical formula recognition** (LaTeX extraction from images)
- **Multi-language support** enhancements (currently English only)
- **Real-time extraction monitoring** dashboard with OCR progress
- **Adaptive DPI selection** based on document characteristics

### 13.3 Scalability Considerations
- **Distributed OCR processing** across multiple machines
- **GPU-accelerated OCR** for faster processing
- **Cloud OCR services** integration (Google Vision API, AWS Textract) as fallback
- **Cloud storage** integration for large document collections
- **Stream processing** for continuous document ingestion
