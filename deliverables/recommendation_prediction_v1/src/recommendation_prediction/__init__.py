"""Digikala recommendation-status prediction component."""

from .preprocessing import build_model_text, normalize_text

__all__ = ["RecommendationPredictor", "build_model_text", "normalize_text"]


def __getattr__(name):
    if name == "RecommendationPredictor":
        from .predictor import RecommendationPredictor

        return RecommendationPredictor
    raise AttributeError(name)
