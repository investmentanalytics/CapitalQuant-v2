"""Aletheia Regime Engine — detector de régimen basado en microestados."""

from .microstate_classifier import MicrostateConfig, classify_microstates
from .pipeline import RegimeEngineResult, run_pipeline
from .regime_detector import DEFAULT_RULES, RegimeRule, classify_regimes
from .segmenter import SegmenterConfig, segment_chain
from .state_chain import StateChain, build_state_chain
from .transition_matrix import build_transition_matrix, find_frequent_sequences, what_follows

__all__ = [
    "MicrostateConfig",
    "classify_microstates",
    "RegimeEngineResult",
    "run_pipeline",
    "DEFAULT_RULES",
    "RegimeRule",
    "classify_regimes",
    "SegmenterConfig",
    "segment_chain",
    "StateChain",
    "build_state_chain",
    "build_transition_matrix",
    "find_frequent_sequences",
    "what_follows",
]

__version__ = "0.1.0"
