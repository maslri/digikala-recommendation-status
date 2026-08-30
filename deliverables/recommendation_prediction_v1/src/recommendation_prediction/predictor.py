"""Offline inference for the released Digikala recommendation classifier.

The implementation intentionally mirrors the preprocessing and label order used
in the training and final-evaluation notebooks. Softmax scores are model scores,
not calibrated probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .preprocessing import (
    ID2LABEL,
    PREPROCESSING_VERSION,
    VALID_LABELS,
    build_model_text,
)


DEFAULT_MODEL_VERSION = "digikala-rec-xlm-roberta-2pct-v1.0.0"
REQUIRED_MODEL_FILES = ("config.json", "inference_config.json")


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _find_weight_file(model_dir: Path) -> Path:
    candidates = [
        model_dir / "model.safetensors",
        model_dir / "pytorch_model.bin",
    ]
    candidates.extend(sorted(model_dir.glob("model-*.safetensors")))
    candidates.extend(sorted(model_dir.glob("pytorch_model-*.bin")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No model weight file found under {model_dir}")


class RecommendationPredictor:
    """Load the fine-tuned encoder and expose a stable JSON-compatible API."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str | None = None,
        model_version: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        for filename in REQUIRED_MODEL_FILES:
            if not (self.model_dir / filename).is_file():
                raise FileNotFoundError(self.model_dir / filename)

        self.inference_config = json.loads(
            (self.model_dir / "inference_config.json").read_text(encoding="utf-8")
        )
        self._validate_inference_config()
        self.max_length = int(self.inference_config["max_length"])
        self.model_version = (
            model_version
            or self.inference_config.get("model_version")
            or DEFAULT_MODEL_VERSION
        )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            use_fast=True,
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        self._validate_model_config()

        weight_file = _find_weight_file(self.model_dir)
        self.artifact_sha256 = _sha256(weight_file)

    def _validate_inference_config(self) -> None:
        expected = {
            "max_length": 128,
            "labels": list(VALID_LABELS),
            "normalization_version": PREPROCESSING_VERSION,
            "text_column": "text_full",
        }
        mismatches = {
            key: {"expected": value, "actual": self.inference_config.get(key)}
            for key, value in expected.items()
            if self.inference_config.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Incompatible inference_config.json: {mismatches}")

    def _validate_model_config(self) -> None:
        if getattr(self.model.config, "model_type", None) != "xlm-roberta":
            raise ValueError(f"Expected xlm-roberta, got {self.model.config.model_type!r}")
        if int(self.model.config.num_labels) != len(VALID_LABELS):
            raise ValueError(f"Expected 3 labels, got {self.model.config.num_labels}")

        actual_id2label = {
            int(key): value for key, value in dict(self.model.config.id2label).items()
        }
        if actual_id2label != ID2LABEL:
            raise ValueError(f"Unsafe label mapping: expected {ID2LABEL}, got {actual_id2label}")

    def _infer(self, texts: Sequence[str]) -> tuple[torch.Tensor, float]:
        started = time.perf_counter()
        encoded = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=self.max_length,
            padding=True,
            pad_to_multiple_of=8 if self.device.type == "cuda" else None,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(**encoded).logits
            else:
                logits = self.model(**encoded).logits
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return torch.softmax(logits.float(), dim=-1).cpu(), elapsed_ms

    def predict_batch(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        rows = list(records)
        if not rows:
            return []
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        outputs: list[dict[str, Any]] = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            texts = [
                build_model_text(
                    title=row.get("title", ""),
                    body=row.get("body", ""),
                    advantages=row.get("advantages", ""),
                    disadvantages=row.get("disadvantages", ""),
                )
                for row in batch
            ]
            probabilities, elapsed_ms = self._infer(texts)
            per_item_latency = elapsed_ms / len(batch)
            for row, scores_tensor in zip(batch, probabilities):
                scores_list = [float(value) for value in scores_tensor.tolist()]
                ordered = sorted(scores_list, reverse=True)
                prediction_id = max(range(len(scores_list)), key=scores_list.__getitem__)
                outputs.append(
                    {
                        "component": "recommendation_prediction",
                        "schema_version": "1.0.0",
                        "comment_id": row.get("comment_id"),
                        "product_id": row.get("product_id"),
                        "label": ID2LABEL[prediction_id],
                        "scores": {
                            ID2LABEL[index]: score for index, score in enumerate(scores_list)
                        },
                        "confidence_score": ordered[0],
                        "score_margin": ordered[0] - ordered[1],
                        "scores_are_calibrated_probabilities": False,
                        "model_version": self.model_version,
                        "artifact_sha256": self.artifact_sha256,
                        "preprocessing_version": PREPROCESSING_VERSION,
                        "source": "model_prediction",
                        "latency_ms": per_item_latency,
                    }
                )
        return outputs

    def predict_one(self, **record: Any) -> dict[str, Any]:
        return self.predict_batch([record], batch_size=1)[0]

    def resolve(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Preserve an observed label; predict only when the label is missing."""

        observed = record.get("recommendation_status")
        if observed not in (None, ""):
            if observed not in VALID_LABELS:
                raise ValueError(f"Invalid observed recommendation_status: {observed!r}")
            return {
                "component": "recommendation_prediction",
                "schema_version": "1.0.0",
                "comment_id": record.get("comment_id"),
                "product_id": record.get("product_id"),
                "label": observed,
                "scores": None,
                "confidence_score": None,
                "score_margin": None,
                "scores_are_calibrated_probabilities": False,
                "model_version": self.model_version,
                "artifact_sha256": self.artifact_sha256,
                "preprocessing_version": PREPROCESSING_VERSION,
                "source": "observed",
                "latency_ms": 0.0,
            }
        return self.predict_batch([record], batch_size=1)[0]

    def health_check(self) -> dict[str, Any]:
        result = self.predict_one(
            title="\u0622\u0632\u0645\u0627\u06cc\u0634 \u0633\u0644\u0627\u0645\u062a",
            body="\u06a9\u06cc\u0641\u06cc\u062a \u0645\u062d\u0635\u0648\u0644 \u062e\u0648\u0628 \u0628\u0648\u062f",
        )
        return {
            "status": "ok",
            "device": str(self.device),
            "model_version": self.model_version,
            "artifact_sha256": self.artifact_sha256,
            "sample_label": result["label"],
        }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Digikala recommendation prediction")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-json", required=True, help="A JSON object or a list of objects")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else [payload]
    predictor = RecommendationPredictor(args.model_dir, device=args.device)
    print(json.dumps(predictor.predict_batch(records), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
