"""Task B: evidence-span extraction, dual candidates, judge selection, micro-adjustments."""

from .pipeline import GenerationConfig, GenerationPipeline, Models
from .selection import ExtractivenessBand, SelectionWeights, selection_scores
from .text import detect_question_type, extractiveness

__all__ = ["ExtractivenessBand", "GenerationConfig", "GenerationPipeline", "Models", "SelectionWeights",
           "detect_question_type", "extractiveness", "selection_scores"]
