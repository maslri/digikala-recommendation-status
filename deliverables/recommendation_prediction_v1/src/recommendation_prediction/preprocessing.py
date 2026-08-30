"""Frozen fa_light_v1 preprocessing and label contract."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any


VALID_LABELS = ("recommended", "not_recommended", "no_idea")
ID2LABEL = {index: label for index, label in enumerate(VALID_LABELS)}
LABEL2ID = {label: index for index, label in ID2LABEL.items()}
NULL_TOKENS = {"", "nan", "none", "null", "na", "n/a"}
ARABIC_TO_PERSIAN = str.maketrans(
    {"\u064a": "\u06cc", "\u0649": "\u06cc", "\u0643": "\u06a9"}
)
PREPROCESSING_VERSION = "fa_light_v1"


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (list, tuple)):
        raise TypeError(
            "advantages/disadvantages and all text fields must be strings in schema v1; "
            "JSON arrays would change the training-time representation"
        )
    return str(value)


def normalize_text(value: Any) -> str:
    """Apply the exact fa_light_v1 normalization used during training."""

    text = _coerce_text(value)
    if text.strip().lower() in NULL_TOKENS:
        return ""
    text = unicodedata.normalize("NFKC", text).translate(ARABIC_TO_PERSIAN)
    text = text.replace("\ufeff", "")
    return re.sub(r"\s+", " ", text).strip()


def build_model_text(
    *,
    title: Any = "",
    body: Any = "",
    advantages: Any = "",
    disadvantages: Any = "",
) -> str:
    """Build the tagged input text in the same order used for fine-tuning."""

    fields = (
        ("[TITLE]", normalize_text(title)),
        ("[BODY]", normalize_text(body)),
        ("[ADVANTAGES]", normalize_text(advantages)),
        ("[DISADVANTAGES]", normalize_text(disadvantages)),
    )
    text = " ".join(f"{tag} {value}" for tag, value in fields if value)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("At least one of title/body/advantages/disadvantages must be non-empty")
    return text
