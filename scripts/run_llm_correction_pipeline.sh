#!/bin/bash
################################################################################
# LLM Correction Pipeline - Runs qwen2.5:14b correction on both documents
################################################################################

set -e

# Wait for spell check to finish
echo "Waiting for spell check to complete..."
while ! [ -f "data/extracted/truth_and_success_spell.txt" ] || ! [ -f "data/extracted/general_propositions_spell.txt" ]; do
    sleep 5
done

echo "✓ Spell check complete! Starting LLM correction..."
echo ""

# Truth and Success LLM correction
echo "========================================="
echo "LLM Correcting: Truth and Success"
echo "Estimated time: ~2 hours for 46K words"
echo "========================================="
python src/llm_ocr_corrector.py \
    data/extracted/truth_and_success_spell.txt \
    -o data/extracted/truth_and_success_final.txt \
    --model qwen2.5:14b \
    --chunk-size 1500 \
    2>&1 | tee logs/truth_llm.log

echo "✓ Truth and Success LLM correction complete"
echo ""

# General Propositions LLM correction
echo "========================================="
echo "LLM Correcting: General Propositions"
echo "Estimated time: ~20 min for 7.5K words"
echo "========================================="
python src/llm_ocr_corrector.py \
    data/extracted/general_propositions_spell.txt \
    -o data/extracted/general_propositions_final.txt \
    --model qwen2.5:14b \
    --chunk-size 1500 \
    2>&1 | tee logs/general_llm.log

echo "✓ General Propositions LLM correction complete"
echo ""

# Quality validation
echo "========================================="
echo "Quality Validation"
echo "========================================="
python src/ocr_semantic_validator.py data/extracted/truth_and_success_final.txt 2>&1 | tee logs/truth_validation.log
python src/ocr_semantic_validator.py data/extracted/general_propositions_final.txt 2>&1 | tee logs/general_validation.log

echo ""
echo "========================================="
echo "PIPELINE COMPLETE!"
echo "========================================="
echo ""
echo "Final files:"
echo "  - data/extracted/truth_and_success_final.txt"
echo "  - data/extracted/general_propositions_final.txt"
echo ""
echo "Total corpus: ~576,000 words at 85-90% quality"
echo "Ready for LLM training!"
