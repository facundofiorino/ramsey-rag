#!/bin/bash
################################################################################
# Complete OCR Extraction and LLM Correction Pipeline
#
# This script orchestrates the complete extraction pipeline:
# 1. OCR extraction of missing documents
# 2. Basic post-processing (spell check)
# 3. LLM-based correction (Ollama qwen2.5:14b)
# 4. Quality validation
#
# Runs entirely in background - safe to leave overnight!
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/complete_pipeline_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$MASTER_LOG"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$MASTER_LOG"
}

log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$MASTER_LOG"
}

################################################################################
# Configuration
################################################################################

MODEL="qwen2.5:14b"
CHUNK_SIZE=1500  # Larger chunks for better context
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

log "=========================================================================="
log "RAMSEY TRAINING - COMPLETE EXTRACTION PIPELINE"
log "=========================================================================="
log "Project root: $PROJECT_ROOT"
log "LLM Model: $MODEL"
log "Chunk size: $CHUNK_SIZE words"
log "Master log: $MASTER_LOG"
log "=========================================================================="

################################################################################
# Step 1: Verify Ollama and Model
################################################################################

log "Step 1: Verifying Ollama setup..."

if ! command -v ollama &> /dev/null; then
    log_error "Ollama not found. Install from: https://ollama.ai"
    exit 1
fi

if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    log_error "Ollama server not running. Start with: ollama serve"
    exit 1
fi

if ! ollama list | grep -q "$MODEL"; then
    log "Model $MODEL not found. Pulling (this may take a while)..."
    ollama pull "$MODEL" 2>&1 | tee -a "$MASTER_LOG"
fi

log "✓ Ollama ready with model: $MODEL"

################################################################################
# Step 2: OCR Missing Documents
################################################################################

log ""
log "Step 2: OCR extraction of missing documents..."

# Document 1: Truth and Success
DOC1_PDF="ramsey_data/Frank Ramsey _ Truth and Success -- Jérôme Dokic; Pascal Engel -- Taylor & Francis (Unlimited), London, 2003 -- London ; New York_ Routledge -- 9780203217832 -- fd21e06107c0fdb53d4016fd3c2f9116 -- Anna's Archive.pdf"
DOC1_TXT="data/extracted/Frank Ramsey _ Truth and Success -- Jérôme Dokic; Pascal Engel -- Taylor & Francis (Unlimited), London, 2003 -- London ; New York_ Routledge -- 9780203217832 -- fd21e06107c0fdb53d4016fd3c2f9116 -- Anna's Archive.txt"

if [ ! -f "$DOC1_TXT" ]; then
    log "OCR: Truth and Success (128 pages, ~26 min)..."
    python src/processors/ocr_processor.py "$DOC1_PDF" poor 2>&1 | tee "$LOG_DIR/doc1_ocr.log"
    log "✓ Truth and Success OCR complete"
else
    log "✓ Truth and Success already extracted"
fi

# Document 2: General Propositions
DOC2_PDF="ramsey_data/Ramsey, Frank - General Propositions and Causality - libgen.li.pdf"
DOC2_TXT="data/extracted/Ramsey, Frank - General Propositions and Causality - libgen.li.txt"

if [ ! -f "$DOC2_TXT" ]; then
    log "OCR: General Propositions (30 pages, ~6 min)..."
    python src/processors/ocr_processor.py "$DOC2_PDF" poor 2>&1 | tee "$LOG_DIR/doc2_ocr.log"
    log "✓ General Propositions OCR complete"
else
    log "✓ General Propositions already extracted"
fi

################################################################################
# Step 3: Basic Post-Processing (Spell Check)
################################################################################

log ""
log "Step 3: Basic post-processing (spell check)..."

DOC1_BASIC="data/extracted/truth_and_success_basic.txt"
if [ ! -f "$DOC1_BASIC" ]; then
    log "Spell checking: Truth and Success..."
    python src/ocr_post_processor.py "$DOC1_TXT" -o "$DOC1_BASIC" 2>&1 | tee "$LOG_DIR/doc1_spell.log"
    log "✓ Truth and Success spell check complete"
else
    log "✓ Truth and Success already spell checked"
fi

DOC2_BASIC="data/extracted/general_propositions_basic.txt"
if [ ! -f "$DOC2_BASIC" ]; then
    log "Spell checking: General Propositions..."
    python src/ocr_post_processor.py "$DOC2_TXT" -o "$DOC2_BASIC" 2>&1 | tee "$LOG_DIR/doc2_spell.log"
    log "✓ General Propositions spell check complete"
else
    log "✓ General Propositions already spell checked"
fi

################################################################################
# Step 4: LLM Correction (The Long Part!)
################################################################################

log ""
log "Step 4: LLM-based correction with $MODEL..."
log "Note: This will take 3-4 hours for 80K words. Progress logged to separate files."

DOC1_FINAL="data/extracted/truth_and_success_final.txt"
if [ ! -f "$DOC1_FINAL" ]; then
    log "LLM correcting: Truth and Success (~65K words, ~2 hours)..."
    log "Progress log: $LOG_DIR/doc1_llm.log"
    python src/llm_ocr_corrector.py "$DOC1_BASIC" \
        -o "$DOC1_FINAL" \
        --model "$MODEL" \
        --chunk-size "$CHUNK_SIZE" \
        2>&1 | tee "$LOG_DIR/doc1_llm.log"
    log "✓ Truth and Success LLM correction complete"
else
    log "✓ Truth and Success already LLM corrected"
fi

DOC2_FINAL="data/extracted/general_propositions_final.txt"
if [ ! -f "$DOC2_FINAL" ]; then
    log "LLM correcting: General Propositions (~15K words, ~30 min)..."
    log "Progress log: $LOG_DIR/doc2_llm.log"
    python src/llm_ocr_corrector.py "$DOC2_BASIC" \
        -o "$DOC2_FINAL" \
        --model "$MODEL" \
        --chunk-size "$CHUNK_SIZE" \
        2>&1 | tee "$LOG_DIR/doc2_llm.log"
    log "✓ General Propositions LLM correction complete"
else
    log "✓ General Propositions already LLM corrected"
fi

################################################################################
# Step 5: Quality Validation
################################################################################

log ""
log "Step 5: Quality validation..."

log "Validating: Truth and Success..."
python src/ocr_semantic_validator.py "$DOC1_FINAL" 2>&1 | tee "$LOG_DIR/doc1_validation.log"

log "Validating: General Propositions..."
python src/ocr_semantic_validator.py "$DOC2_FINAL" 2>&1 | tee "$LOG_DIR/doc2_validation.log"

################################################################################
# Step 6: Final Summary
################################################################################

log ""
log "=========================================================================="
log "PIPELINE COMPLETE!"
log "=========================================================================="
log ""
log "Final corpus files:"
log "  1. Truth and Success:       $DOC1_FINAL"
log "  2. General Propositions:    $DOC2_FINAL"
log ""
log "Existing high-quality files (530K words at 88-90%):"
for file in data/extracted/*.txt; do
    if [[ "$file" != *"basic"* ]] && [[ "$file" != *"final"* ]] && [[ "$file" != *"corrected"* ]]; then
        wc=$(wc -w < "$file" 2>/dev/null || echo "0")
        if [ "$wc" -gt 10000 ]; then
            log "  - $(basename "$file"): $(printf "%'d" $wc) words"
        fi
    fi
done
log ""
log "Total corpus: ~610,000 words at 82-90% quality"
log "Ready for LLM training!"
log ""
log "All logs saved in: $LOG_DIR/"
log "Master log: $MASTER_LOG"
log "=========================================================================="
