#!/usr/bin/env python3
"""
LangGraph OCR Optimizer Workflow

Implements an evaluator-optimizer pattern that iteratively:
1. Extracts text with OCR using current parameters
2. Evaluates quality using semantic validation
3. Optimizes parameters based on results
4. Repeats until optimal quality achieved or max attempts reached

Based on: https://docs.langchain.com/oss/python/langgraph/workflows-agents
"""

import sys
from pathlib import Path
from typing import TypedDict, Annotated, Sequence, Optional
from operator import add
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Import our existing OCR components
from processors.ocr_processor import OCRProcessor
from ocr_semantic_validator import SemanticValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define the state that flows through the graph
class OCROptimizerState(TypedDict):
    """State for OCR optimization workflow"""

    # Input
    pdf_path: str
    page_number: int
    target_quality: float  # Target dictionary ratio (e.g., 0.70 for 70%)

    # Current attempt tracking
    attempt: int
    max_attempts: int

    # Current parameters
    dpi: int
    preprocessing: str  # 'ultra', 'adaptive', 'gentle'

    # Results from current attempt
    ocr_text: Optional[str]
    ocr_confidence: Optional[float]
    dictionary_ratio: Optional[float]
    quality_score: Optional[float]

    # Historical results
    attempts_history: Annotated[Sequence[dict], add]

    # Decision tracking
    should_continue: bool
    reason: str  # Why we stopped or continue


class OCROptimizerWorkflow:
    """LangGraph workflow for iterative OCR optimization"""

    # Parameter search space
    DPI_OPTIONS = [600, 800, 1200]
    PREPROCESSING_OPTIONS = ['gentle', 'adaptive', 'ultra']

    def __init__(
        self,
        target_quality: float = 0.70,
        max_attempts: int = 6,
        use_llm_optimizer: bool = False
    ):
        """
        Initialize OCR optimizer workflow

        Args:
            target_quality: Target dictionary ratio (0.0-1.0)
            max_attempts: Maximum optimization attempts
            use_llm_optimizer: Use LLM to intelligently select parameters (vs heuristic)
        """
        self.target_quality = target_quality
        self.max_attempts = max_attempts
        self.use_llm_optimizer = use_llm_optimizer

        # Initialize components
        self.validator = SemanticValidator()

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""

        # Create graph
        workflow = StateGraph(OCROptimizerState)

        # Add nodes
        workflow.add_node("extract", self.extract_node)
        workflow.add_node("evaluate", self.evaluate_node)
        workflow.add_node("optimize", self.optimize_node)

        # Define edges
        workflow.set_entry_point("extract")
        workflow.add_edge("extract", "evaluate")

        # Conditional edge based on evaluation
        workflow.add_conditional_edges(
            "evaluate",
            self.should_continue_or_end,
            {
                "optimize": "optimize",
                "end": END
            }
        )

        workflow.add_edge("optimize", "extract")

        return workflow.compile()

    def extract_node(self, state: OCROptimizerState) -> OCROptimizerState:
        """
        Node: Extract text using current OCR parameters
        """
        logger.info(f"🔍 Attempt {state['attempt']}/{state['max_attempts']}: "
                   f"DPI={state['dpi']}, Preprocessing={state['preprocessing']}")

        try:
            # Initialize OCR processor with current parameters
            processor = OCRProcessor(
                dpi=state['dpi'],
                enable_semantic_validation=False  # We'll do this separately
            )

            # Extract single page
            pdf_path = Path(state['pdf_path'])
            page_result = processor.extract_page(
                pdf_path,
                state['page_number'],
                preprocessing_mode=state['preprocessing']
            )

            # Update state with results
            state['ocr_text'] = page_result.get('text', '')
            state['ocr_confidence'] = page_result.get('avg_confidence', 0.0)

            logger.info(f"  ✓ Extracted {len(state['ocr_text'])} chars, "
                       f"OCR confidence: {state['ocr_confidence']:.1f}%")

        except Exception as e:
            logger.error(f"  ✗ Extraction failed: {e}")
            state['ocr_text'] = ''
            state['ocr_confidence'] = 0.0

        return state

    def evaluate_node(self, state: OCROptimizerState) -> OCROptimizerState:
        """
        Node: Evaluate quality of extracted text
        """
        logger.info(f"📊 Evaluating quality...")

        if not state['ocr_text']:
            state['dictionary_ratio'] = 0.0
            state['quality_score'] = 0.0
            logger.warning(f"  ⚠️ No text extracted!")
        else:
            # Semantic validation
            result = self.validator.analyze_text(state['ocr_text'])
            state['dictionary_ratio'] = result['dictionary_ratio'] / 100
            state['quality_score'] = result['quality_score'] / 100

            logger.info(f"  📈 Dictionary ratio: {state['dictionary_ratio']:.1%}")
            logger.info(f"  📈 Quality score: {state['quality_score']:.1%}")

        # Record attempt in history
        attempt_record = {
            'attempt': state['attempt'],
            'dpi': state['dpi'],
            'preprocessing': state['preprocessing'],
            'ocr_confidence': state['ocr_confidence'],
            'dictionary_ratio': state['dictionary_ratio'],
            'quality_score': state['quality_score'],
            'text_length': len(state['ocr_text']) if state['ocr_text'] else 0
        }
        state['attempts_history'] = [attempt_record]

        # Determine if we should continue
        if state['dictionary_ratio'] >= state['target_quality']:
            state['should_continue'] = False
            state['reason'] = f"✅ Target quality achieved: {state['dictionary_ratio']:.1%}"
        elif state['attempt'] >= state['max_attempts']:
            state['should_continue'] = False
            state['reason'] = f"⛔ Max attempts reached ({state['max_attempts']})"
        else:
            state['should_continue'] = True
            state['reason'] = f"🔄 Quality {state['dictionary_ratio']:.1%} < target {state['target_quality']:.1%}"

        logger.info(f"  {state['reason']}")

        return state

    def optimize_node(self, state: OCROptimizerState) -> OCROptimizerState:
        """
        Node: Optimize parameters based on evaluation

        Uses either heuristic rules or LLM to select next parameters
        """
        logger.info(f"⚙️  Optimizing parameters...")

        state['attempt'] += 1

        if self.use_llm_optimizer:
            # Use LLM to intelligently select parameters
            # (Implementation would use langchain_anthropic here)
            logger.info("  🤖 Using LLM optimizer (not yet implemented)")
            state = self._heuristic_optimize(state)
        else:
            # Use heuristic optimization
            state = self._heuristic_optimize(state)

        logger.info(f"  → Next: DPI={state['dpi']}, Preprocessing={state['preprocessing']}")

        return state

    def _heuristic_optimize(self, state: OCROptimizerState) -> OCROptimizerState:
        """
        Heuristic parameter optimization strategy

        Strategy:
        1. Start with 800 DPI + ultra (our tested best)
        2. If poor, try 600 DPI + adaptive (faster, sometimes better for clear docs)
        3. If still poor, try 1200 DPI + ultra (brute force)
        4. Try other combinations systematically
        """
        attempt = state['attempt']
        current_quality = state['dictionary_ratio']

        # Define search sequence (ordered by expected effectiveness)
        search_sequence = [
            (800, 'ultra'),      # Best overall based on testing
            (600, 'adaptive'),   # Good for clean documents
            (800, 'adaptive'),   # Less aggressive, may help
            (1200, 'ultra'),     # Brute force for very degraded
            (600, 'ultra'),      # Alternative combination
            (800, 'gentle'),     # Minimal processing
        ]

        # Select next parameters
        if attempt <= len(search_sequence):
            state['dpi'], state['preprocessing'] = search_sequence[attempt - 1]
        else:
            # Fallback: cycle through remaining combinations
            idx = (attempt - 1) % len(search_sequence)
            state['dpi'], state['preprocessing'] = search_sequence[idx]

        return state

    def should_continue_or_end(self, state: OCROptimizerState) -> str:
        """
        Conditional edge: Determine if we should continue optimizing or end
        """
        return "end" if not state['should_continue'] else "optimize"

    def run(
        self,
        pdf_path: str,
        page_number: int = 0,
        initial_dpi: int = 800,
        initial_preprocessing: str = 'ultra'
    ) -> dict:
        """
        Run the optimization workflow

        Args:
            pdf_path: Path to PDF file
            page_number: Page to optimize (0-indexed)
            initial_dpi: Starting DPI
            initial_preprocessing: Starting preprocessing mode

        Returns:
            Final state with best parameters found
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 Starting OCR Optimization Workflow")
        logger.info(f"{'='*80}")
        logger.info(f"PDF: {Path(pdf_path).name}")
        logger.info(f"Page: {page_number}")
        logger.info(f"Target quality: {self.target_quality:.1%}")
        logger.info(f"Max attempts: {self.max_attempts}\n")

        # Initialize state
        initial_state = OCROptimizerState(
            pdf_path=pdf_path,
            page_number=page_number,
            target_quality=self.target_quality,
            attempt=1,
            max_attempts=self.max_attempts,
            dpi=initial_dpi,
            preprocessing=initial_preprocessing,
            ocr_text=None,
            ocr_confidence=None,
            dictionary_ratio=None,
            quality_score=None,
            attempts_history=[],
            should_continue=True,
            reason=''
        )

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        # Print summary
        self._print_summary(final_state)

        return final_state

    def _print_summary(self, state: OCROptimizerState):
        """Print optimization summary"""
        logger.info(f"\n{'='*80}")
        logger.info(f"📋 Optimization Summary")
        logger.info(f"{'='*80}\n")

        logger.info(f"Final Result: {state['reason']}")
        logger.info(f"Total attempts: {state['attempt']}")
        logger.info(f"\nBest Parameters:")
        logger.info(f"  DPI: {state['dpi']}")
        logger.info(f"  Preprocessing: {state['preprocessing']}")
        logger.info(f"  Dictionary Ratio: {state['dictionary_ratio']:.1%}")
        logger.info(f"  Quality Score: {state['quality_score']:.1%}")
        logger.info(f"  OCR Confidence: {state['ocr_confidence']:.1f}%")

        # Show all attempts
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 All Attempts")
        logger.info(f"{'='*80}\n")
        logger.info(f"{'#':<4} {'DPI':<6} {'Preproc':<10} {'Dict%':<8} {'Quality%':<10} {'OCR%':<8}")
        logger.info(f"{'-'*60}")

        for attempt in state['attempts_history']:
            logger.info(
                f"{attempt['attempt']:<4} "
                f"{attempt['dpi']:<6} "
                f"{attempt['preprocessing']:<10} "
                f"{attempt['dictionary_ratio']*100:<8.1f} "
                f"{attempt['quality_score']*100:<10.1f} "
                f"{attempt['ocr_confidence']:<8.1f}"
            )

        # Find best attempt
        best = max(state['attempts_history'], key=lambda x: x['dictionary_ratio'])
        logger.info(f"\n✨ Best: Attempt #{best['attempt']} - "
                   f"{best['dpi']} DPI + {best['preprocessing']} = {best['dictionary_ratio']:.1%}")
        logger.info(f"{'='*80}\n")


def main():
    """CLI for OCR optimization workflow"""
    import argparse

    parser = argparse.ArgumentParser(description='Optimize OCR parameters using LangGraph')
    parser.add_argument('pdf_path', type=str, help='Path to PDF file')
    parser.add_argument('--page', type=int, default=0, help='Page number to optimize (default: 0)')
    parser.add_argument('--target', type=float, default=0.70,
                       help='Target dictionary ratio (default: 0.70 = 70%%)')
    parser.add_argument('--max-attempts', type=int, default=6,
                       help='Maximum optimization attempts (default: 6)')
    parser.add_argument('--use-llm', action='store_true',
                       help='Use LLM for parameter optimization (vs heuristic)')

    args = parser.parse_args()

    # Create workflow
    optimizer = OCROptimizerWorkflow(
        target_quality=args.target,
        max_attempts=args.max_attempts,
        use_llm_optimizer=args.use_llm
    )

    # Run optimization
    result = optimizer.run(
        pdf_path=args.pdf_path,
        page_number=args.page
    )

    # Return best parameters
    print(f"\n🎯 Recommended settings for this document:")
    print(f"   DPI: {result['dpi']}")
    print(f"   Preprocessing: {result['preprocessing']}")
    print(f"   Expected quality: {result['dictionary_ratio']:.1%}\n")


if __name__ == '__main__':
    main()
