# LangGraph OCR Optimizer Workflow

## Overview

This workflow uses **LangGraph** to implement an **evaluator-optimizer pattern** that iteratively finds the best OCR parameters for each document. Instead of using fixed settings (800 DPI + ultra), it automatically tests different combinations and selects the optimal configuration.

---

## Architecture

### Workflow Graph

```
┌─────────────┐
│   START     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  EXTRACT    │ ◄─────┐
│             │       │
│ Run OCR     │       │
│ with params │       │
└──────┬──────┘       │
       │              │
       ▼              │
┌─────────────┐       │
│  EVALUATE   │       │
│             │       │
│ Check       │       │
│ quality     │       │
└──────┬──────┘       │
       │              │
       ▼              │
   ┌───────┐          │
   │Decision│          │
   └───┬───┘          │
       │              │
  ┌────┴────┐         │
  │         │         │
Quality   Quality     │
 Good?     Poor?      │
  │         │         │
  ▼         ▼         │
┌───┐   ┌──────────┐ │
│END│   │ OPTIMIZE │ │
└───┘   │          │ │
        │ Adjust   │ │
        │ params   │─┘
        └──────────┘
```

### State

The workflow maintains state that flows through nodes:

```python
class OCROptimizerState:
    # Input
    pdf_path: str
    page_number: int
    target_quality: float  # e.g., 0.70 for 70% dictionary ratio

    # Tracking
    attempt: int
    max_attempts: int

    # Current parameters
    dpi: int
    preprocessing: str

    # Results
    ocr_text: str
    ocr_confidence: float
    dictionary_ratio: float
    quality_score: float

    # History
    attempts_history: List[dict]

    # Control
    should_continue: bool
    reason: str
```

### Nodes

#### 1. Extract Node
**Purpose:** Run OCR with current parameters

**Actions:**
- Initialize OCRProcessor with current DPI and preprocessing mode
- Extract text from specified page
- Record OCR confidence and text length

**Output:** Updated state with OCR results

#### 2. Evaluate Node
**Purpose:** Assess quality of extracted text

**Actions:**
- Run semantic validation on extracted text
- Calculate dictionary ratio and quality score
- Record attempt in history
- Determine if we should continue or stop

**Decision Logic:**
```python
if dictionary_ratio >= target_quality:
    STOP - "Target quality achieved"
elif attempt >= max_attempts:
    STOP - "Max attempts reached"
else:
    CONTINUE - "Try next parameter combination"
```

#### 3. Optimize Node
**Purpose:** Select next parameters to try

**Strategies:**

**A. Heuristic Optimizer** (default)
- Predefined search sequence based on empirical testing:
  1. 800 DPI + ultra (best overall)
  2. 600 DPI + adaptive (good for clean docs)
  3. 800 DPI + adaptive (less aggressive)
  4. 1200 DPI + ultra (brute force)
  5. 600 DPI + ultra (alternative)
  6. 800 DPI + gentle (minimal processing)

**B. LLM Optimizer** (future)
- Use Claude to intelligently analyze results
- Select next parameters based on:
  - Current quality metrics
  - Previous attempt outcomes
  - Document characteristics
  - Domain knowledge about OCR

### Edges

- `START → EXTRACT`: Begin workflow
- `EXTRACT → EVALUATE`: Always evaluate after extraction
- `EVALUATE → END`: If quality is good or max attempts reached
- `EVALUATE → OPTIMIZE`: If quality is poor and attempts remain
- `OPTIMIZE → EXTRACT`: Try again with new parameters

---

## Heuristic Optimization Strategy

### Search Sequence Rationale

Based on extensive testing documented in `FINAL_OCR_RESULTS.md` and `DPI_QUALITY_COMPARISON.md`:

**Attempt 1: 800 DPI + ultra**
- **Why:** Best overall performance in testing (38.1% confidence)
- **Best for:** Degraded historical documents
- **Expected:** Highest success rate

**Attempt 2: 600 DPI + adaptive**
- **Why:** Faster, less noise amplification
- **Best for:** Clean modern documents
- **Expected:** Better for documents where 800 DPI over-samples

**Attempt 3: 800 DPI + adaptive**
- **Why:** Same resolution, less aggressive preprocessing
- **Best for:** Documents with minor degradation
- **Expected:** Helps if ultra preprocessing created artifacts

**Attempt 4: 1200 DPI + ultra**
- **Why:** Maximum detail capture
- **Best for:** Perfect quality scans with fine text
- **Expected:** Rarely needed, but worth trying for very small text

**Attempts 5-6:** Alternative combinations
- Exploration of remaining parameter space

### Parameter Space

**DPI Options:**
- 600: Fast, good for clean documents
- 800: Optimal for most historical documents
- 1200: High detail, but noise amplification risk

**Preprocessing Options:**
- `gentle`: Minimal processing, preserves original quality
- `adaptive`: Balanced enhancement
- `ultra`: Aggressive denoising and enhancement

**Total Combinations:** 3 DPI × 3 Preprocessing = 9 possible configurations

---

## Usage

### Basic Usage

```bash
# Optimize OCR for a specific page
python src/data/ocr_optimizer_workflow.py "ramsey_data/document.pdf" --page 5

# Set custom target quality (70% dictionary ratio)
python src/data/ocr_optimizer_workflow.py "ramsey_data/document.pdf" --target 0.70

# Allow more optimization attempts
python src/data/ocr_optimizer_workflow.py "ramsey_data/document.pdf" --max-attempts 9
```

### Programmatic Usage

```python
from src.data.ocr_optimizer_workflow import OCROptimizerWorkflow

# Create optimizer
optimizer = OCROptimizerWorkflow(
    target_quality=0.70,  # 70% dictionary ratio
    max_attempts=6
)

# Run optimization
result = optimizer.run(
    pdf_path="ramsey_data/document.pdf",
    page_number=5
)

# Get best parameters
print(f"Best DPI: {result['dpi']}")
print(f"Best preprocessing: {result['preprocessing']}")
print(f"Quality achieved: {result['dictionary_ratio']:.1%}")
```

### Example Output

```
================================================================================
🚀 Starting OCR Optimization Workflow
================================================================================
PDF: General Propositions and Causality.pdf
Page: 0
Target quality: 70.0%
Max attempts: 6

🔍 Attempt 1/6: DPI=800, Preprocessing=ultra
  ✓ Extracted 1523 chars, OCR confidence: 38.1%
📊 Evaluating quality...
  📈 Dictionary ratio: 45.7%
  📈 Quality score: 63.1%
  🔄 Quality 45.7% < target 70.0%
⚙️  Optimizing parameters...
  → Next: DPI=600, Preprocessing=adaptive

🔍 Attempt 2/6: DPI=600, Preprocessing=adaptive
  ✓ Extracted 1445 chars, OCR confidence: 32.9%
📊 Evaluating quality...
  📈 Dictionary ratio: 42.3%
  📈 Quality score: 58.5%
  🔄 Quality 42.3% < target 70.0%
⚙️  Optimizing parameters...
  → Next: DPI=800, Preprocessing=adaptive

[... continues ...]

================================================================================
📋 Optimization Summary
================================================================================

Final Result: ⛔ Max attempts reached (6)
Total attempts: 6

Best Parameters:
  DPI: 800
  Preprocessing: ultra
  Dictionary Ratio: 45.7%
  Quality Score: 63.1%
  OCR Confidence: 38.1%

================================================================================
📊 All Attempts
================================================================================

#    DPI    Preproc    Dict%    Quality%   OCR%
------------------------------------------------------------
1    800    ultra      45.7     63.1       38.1
2    600    adaptive   42.3     58.5       32.9
3    800    adaptive   43.1     59.8       35.2
4    1200   ultra      41.8     57.3       32.7
5    600    ultra      42.5     58.9       33.8
6    800    gentle     44.2     61.5       36.5

✨ Best: Attempt #1 - 800 DPI + ultra = 45.7%
================================================================================

🎯 Recommended settings for this document:
   DPI: 800
   Preprocessing: ultra
   Expected quality: 45.7%
```

---

## Advantages over Fixed Parameters

### 1. Document-Specific Optimization

**Problem:** Different documents need different settings
- Modern PDFs: 600 DPI sufficient
- Degraded manuscripts: 800 DPI optimal
- Very clean scans: 1200 DPI might help

**Solution:** Workflow finds best settings per document

### 2. Automatic Quality Assessment

**Problem:** Manual testing is time-consuming
- Must run multiple extractions
- Compare results manually
- Track which settings worked best

**Solution:** Workflow automatically tests and compares

### 3. Exploration of Parameter Space

**Problem:** May not know optimal settings in advance
- Our testing found 800 DPI best, but maybe not for all docs
- Preprocessing mode depends on document condition

**Solution:** Systematic exploration finds edge cases

### 4. Data-Driven Decisions

**Problem:** Guessing parameters wastes time
- Running wrong settings = wasted processing
- May miss optimal configuration

**Solution:** Evidence-based parameter selection

---

## Extending to LLM-Based Optimization

### Future Enhancement: Claude as Optimizer

Instead of heuristic search, use Claude to intelligently select next parameters:

```python
def _llm_optimize(self, state: OCROptimizerState) -> OCROptimizerState:
    """Use Claude to select next parameters"""

    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model="claude-3-haiku-20240307")

    # Build context from previous attempts
    history = "\n".join([
        f"Attempt {a['attempt']}: {a['dpi']} DPI + {a['preprocessing']} "
        f"= {a['dictionary_ratio']:.1%} quality"
        for a in state['attempts_history']
    ])

    prompt = f"""You are an OCR optimization expert. Based on previous attempts,
    select the best parameters to try next.

    Previous attempts:
    {history}

    Current quality: {state['dictionary_ratio']:.1%}
    Target quality: {state['target_quality']:.1%}

    Available options:
    DPI: 600, 800, 1200
    Preprocessing: gentle, adaptive, ultra

    Reasoning:
    - 600 DPI: Fast, good for clean documents
    - 800 DPI: Optimal for historical documents
    - 1200 DPI: Maximum detail, but may amplify noise

    - gentle: Minimal processing
    - adaptive: Balanced enhancement
    - ultra: Aggressive denoising

    What parameters should we try next to improve quality?
    Respond with JSON: {{"dpi": <int>, "preprocessing": "<str>", "reasoning": "<str>"}}
    """

    response = llm.invoke(prompt)

    # Parse LLM response
    import json
    result = json.loads(response.content)

    logger.info(f"  🤖 LLM suggests: {result['dpi']} DPI + {result['preprocessing']}")
    logger.info(f"  💭 Reasoning: {result['reasoning']}")

    state['dpi'] = result['dpi']
    state['preprocessing'] = result['preprocessing']

    return state
```

### Benefits of LLM Optimizer

1. **Intelligent reasoning** about parameter relationships
2. **Learns from patterns** in previous attempts
3. **Adapts strategy** based on document characteristics
4. **Explains decisions** for transparency

---

## Integration with Existing Pipeline

### Option 1: Replace Fixed Parameters

```python
# Old approach (extract_all.py)
if quality_level == "poor":
    dpi = 800
    preprocessing = "ultra"

# New approach with optimizer
optimizer = OCROptimizerWorkflow(target_quality=0.70)
result = optimizer.run(pdf_path, page_number=0)

dpi = result['dpi']
preprocessing = result['preprocessing']
# Use these for all pages in document
```

### Option 2: Hybrid Approach

```python
# Use optimizer for first page, then apply to all
result = optimizer.run(pdf_path, page_number=0)

if result['dictionary_ratio'] >= 0.70:
    # First page good, use same settings for rest
    extract_pdf(pdf_path, dpi=result['dpi'], preprocessing=result['preprocessing'])
else:
    # First page poor even after optimization
    # Either skip document or use best settings found
    if result['dictionary_ratio'] >= 0.40:
        extract_pdf(pdf_path, dpi=result['dpi'], preprocessing=result['preprocessing'])
    else:
        logger.warning(f"Document quality too poor ({result['dictionary_ratio']:.1%}), skipping")
```

---

## Performance Considerations

### Time Cost

**Per Optimization Run:**
- 1 attempt: ~12 seconds (800 DPI + ultra, 1 page)
- 6 attempts: ~72 seconds = 1.2 minutes
- Cost: 6x slower than single extraction

**Mitigation:**
- Only optimize first page, apply settings to rest
- Cache optimal settings for similar documents
- Use parallel processing for multiple pages

### Resource Usage

**Memory:**
- Stores 6 different OCR results
- Adds ~30-50 MB per optimization run
- Minimal compared to image processing

**Storage:**
- No additional storage (results in memory only)
- Could save optimization history for analysis

---

## Recommendations

### When to Use Optimizer

✅ **Use optimizer for:**
- Unknown document quality
- Critical documents needing best quality
- First document of a new type
- Benchmarking and testing

❌ **Skip optimizer for:**
- Documents with known good settings
- Batch processing of similar documents
- Time-critical extractions
- Documents already tested

### Optimal Workflow

1. **Sample and optimize** first document of each type
2. **Cache best parameters** for that document type
3. **Apply cached settings** to similar documents
4. **Re-optimize** if quality drops below threshold

---

## Testing and Validation

### Test the Workflow

```bash
# Test on degraded manuscript
python src/data/ocr_optimizer_workflow.py \
  "ramsey_data/Ramsey, Frank - General Propositions and Causality - libgen.li.pdf" \
  --page 0 \
  --target 0.70 \
  --max-attempts 6

# Test on modern PDF
python src/data/ocr_optimizer_workflow.py \
  "ramsey_data/[History of Analytic Philosophy] *.pdf" \
  --page 0 \
  --target 0.85 \
  --max-attempts 3
```

### Expected Outcomes

**Degraded Manuscript:**
- Should try multiple combinations
- Likely settle on 800 DPI + ultra
- May not reach 70% target (document limitation)
- Shows clear progression in attempts

**Modern PDF:**
- Should succeed quickly (1-2 attempts)
- Likely 600 DPI + adaptive sufficient
- Easily exceeds 85% target
- Demonstrates efficiency on easy documents

---

## Future Enhancements

### 1. Multi-Page Optimization
- Test multiple pages, average results
- Handle documents with varying quality

### 2. Dynamic Target Quality
- Adjust target based on document age
- Lower threshold for very old documents

### 3. Cost-Benefit Analysis
- Balance quality vs processing time
- Stop early if improvement is marginal

### 4. Parallel Exploration
- Run multiple parameter combinations simultaneously
- Use Ray or multiprocessing

### 5. Learning from History
- Build database of optimal settings by document type
- Use ML to predict best parameters

---

## Conclusion

The LangGraph OCR Optimizer Workflow provides:

✅ **Automatic parameter optimization**
✅ **Data-driven quality improvement**
✅ **Systematic exploration of parameter space**
✅ **Extensible to LLM-based intelligence**
✅ **Production-ready with existing OCR pipeline**

This transforms OCR from a fixed-parameter process to an **adaptive, intelligent system** that finds optimal settings for each unique document! 🚀
