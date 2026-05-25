"""Spurious-token filters used by the UniRefiner training method.

All filters use the same mask convention: `True` marks a token as clean and
eligible for refinement supervision; `False` marks a token as rejected by the
corresponding spurious-token criterion.
"""

from .adaptive_register import analyze_adaptive_spurious_detection, filter_by_adaptive_register
from .attention_hijack import analyze_attention_hijacking, filter_attention_hijackees
from .fp_gp import analyze_fp_gp_similarity, filter_by_fp_gp_similarity
from .sampling import safe_multinomial

__all__ = [
    "analyze_adaptive_spurious_detection",
    "analyze_attention_hijacking",
    "analyze_fp_gp_similarity",
    "filter_attention_hijackees",
    "filter_by_adaptive_register",
    "filter_by_fp_gp_similarity",
    "safe_multinomial",
]
