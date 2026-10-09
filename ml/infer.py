"""
Production-grade ONNX inference engine for multilingual smishing detection.

Features
--------
- Config-driven (model file, threshold, max_length, input/output names, class index)
- Fast tokenization via the lightweight `tokenizers` library when tokenizer.json
  exists, with automatic fallback to `transformers.AutoTokenizer`
- Length-sorted micro-batching (bounded memory, minimal padding waste)
- Strict input validation, character clipping, optional Unicode normalization
- Explicit ORT thread control, warm-up, thread-safe tokenization
- Binary softmax or single-logit sigmoid heads
- Model version / variant in every result, structured logging

Requires Python 3.10+, numpy, onnxruntime, and either `tokenizers` or `transformers`.

Optional keys in inference_config.json (all have safe defaults):
    "output_name":            "logits"
    "positive_class_index":   1
    "model_version":          "unknown"
    "unicode_normalization":  "NFKC" | "NFC" | null
    "intra_op_num_threads":   int
    "pad_token":              "[PAD]"
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnxruntime as ort

logger = logging.getLogger("smish.inference")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "smish_model" / "onnx"

DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_CHARS = 2000


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ModelLoadError(RuntimeError):
    """Raised when the model, config, or tokenizer cannot be loaded."""


class InvalidInputError(ValueError):
    """Raised when predict() receives unusable input."""


# --------------------------------------------------------------------------- #
# Tokenizer backends
# --------------------------------------------------------------------------- #
class _FastTokenizerBackend:
    """Uses the standalone `tokenizers` library (small, fast cold start)."""

    _PAD_FALLBACKS = ("[PAD]", "<pad>", "<PAD>")

    def __init__(self, model_dir: Path, max_length: int, pad_token: str | None):
        from tokenizers import Tokenizer  # lazy import

        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tok.no_padding()
        self._tok.enable_truncation(max_length=max_length)
        self.pad_id = self._resolve_pad_id(model_dir, pad_token)

    def _resolve_pad_id(self, model_dir: Path, pad_token: str | None) -> int:
        candidates: list[str] = []
        if pad_token:
            candidates.append(pad_token)

        cfg_path = model_dir / "tokenizer_config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                pt = cfg.get("pad_token")
                if isinstance(pt, dict):
                    pt = pt.get("content")
                if isinstance(pt, str):
                    candidates.append(pt)
            except (OSError, json.JSONDecodeError):
                pass

        candidates.extend(self._PAD_FALLBACKS)
        for tok in candidates:
            tid = self._tok.token_to_id(tok)
            if tid is not None:
                return int(tid)
        raise ModelLoadError(
            "Could not determine the pad token id. Set 'pad_token' in "
            "inference_config.json."
        )

    def encode(self, texts: list[str]) -> dict[str, np.ndarray]:
        encs = self._tok.encode_batch(texts)
        n = len(encs)
        longest = max((len(e.ids) for e in encs), default=1)
        ids = np.full((n, longest), self.pad_id, dtype=np.int64)
        mask = np.zeros((n, longest), dtype=np.int64)
        types = np.zeros((n, longest), dtype=np.int64)
        for i, e in enumerate(encs):
            length = len(e.ids)
            ids[i, :length] = e.ids
            mask[i, :length] = 1
            types[i, :length] = e.type_ids
        return {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}


class _TransformersBackend:
    """Fallback for models without a tokenizer.json."""

    def __init__(self, model_dir: Path, max_length: int):
        try:
            from transformers import AutoTokenizer  # lazy import

            self._tok = AutoTokenizer.from_pretrained(str(model_dir))
        except ValueError:
            from transformers import AlbertTokenizerFast  # fallback for IndicBERT

            self._tok = AlbertTokenizerFast.from_pretrained(str(model_dir))

        self._max_length = max_length

    def encode(self, texts: list[str]) -> dict[str, np.ndarray]:
        enc = self._tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="np",
        )
        return {k: np.asarray(v, dtype=np.int64) for k, v in enc.items()}


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
class SmishClassifier:
    """Thread-safe ONNX smishing classifier."""

    def __init__(
        self,
        model_dir: Path | str = DEFAULT_MODEL_DIR,
        use_int8: bool = True,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_chars: int = DEFAULT_MAX_CHARS,
        intra_op_threads: int | None = None,
        inter_op_threads: int = 1,
        warmup: bool = True,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_chars < 1:
            raise ValueError("max_chars must be >= 1")

        self.model_dir = Path(model_dir)
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.variant = "int8" if use_int8 else "fp32"

        self.config = self._load_config()
        self.max_length = int(self._require("max_length"))
        self.input_names = set(self._require("input_names"))
        self.model_version = str(self.config.get("model_version", "unknown"))
        self.positive_idx = int(self.config.get("positive_class_index", 1))
        self._norm_form = self.config.get("unicode_normalization")
        if self._norm_form not in (None, "NFC", "NFD", "NFKC", "NFKD"):
            raise ModelLoadError(
                f"Invalid unicode_normalization: {self._norm_form!r}"
            )

        self.threshold = self._resolve_threshold(use_int8)
        self.session = self._build_session(intra_op_threads, inter_op_threads)
        self.output_name = self.config.get(
            "output_name", self.session.get_outputs()[0].name
        )
        self._validate_session_inputs()

        self._tokenizer = self._build_tokenizer()
        # Tokenizers are not guaranteed thread-safe; ORT sessions are.
        self._tok_lock = threading.Lock()

        if warmup:
            self._warmup()

        logger.info(
            "Loaded smish model version=%s variant=%s threshold=%.4f "
            "max_length=%d tokenizer=%s",
            self.model_version,
            self.variant,
            self.threshold,
            self.max_length,
            type(self._tokenizer).__name__,
        )

    # ------------------------------ loading ------------------------------- #
    def _load_config(self) -> dict[str, Any]:
        path = self.model_dir / "inference_config.json"
        if not path.is_file():
            raise ModelLoadError(f"Config not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            raise ModelLoadError(f"Invalid JSON in {path}: {exc}") from exc

    def _require(self, key: str) -> Any:
        if key not in self.config:
            raise ModelLoadError(f"Missing '{key}' in inference_config.json")
        return self.config[key]

    def _resolve_threshold(self, use_int8: bool) -> float:
        try:
            thresholds = self._require("thresholds")
            key = "recommended_for_int8" if use_int8 else "fp32"
            threshold = float(thresholds[key])
        except KeyError as exc:
            raise ModelLoadError(f"Missing threshold key: {exc}") from exc
        if not 0.0 <= threshold <= 1.0:
            raise ModelLoadError(f"Threshold {threshold} outside [0, 1]")
        return threshold

    def _build_session(
        self, intra: int | None, inter: int
    ) -> ort.InferenceSession:
        try:
            model_file = self._require("models")[self.variant]["file"]
        except KeyError as exc:
            raise ModelLoadError(
                f"Config has no model entry for variant '{self.variant}': {exc}"
            ) from exc

        model_path = self.model_dir / model_file
        if not model_path.is_file():
            raise ModelLoadError(f"Model file not found: {model_path}")

        if intra is None:
            intra = int(
                self.config.get(
                    "intra_op_num_threads", min(4, os.cpu_count() or 1)
                )
            )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = intra
        opts.inter_op_num_threads = inter
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        try:
            return ort.InferenceSession(
                str(model_path), opts, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # onnxruntime raises several types
            raise ModelLoadError(f"Failed to load {model_path}: {exc}") from exc

    def _validate_session_inputs(self) -> None:
        graph_inputs = {i.name for i in self.session.get_inputs()}
        unknown = self.input_names - graph_inputs
        if unknown:
            raise ModelLoadError(
                f"Config input_names {sorted(unknown)} not in model inputs "
                f"{sorted(graph_inputs)}"
            )
        # Only feed what the graph actually accepts.
        self.input_names &= graph_inputs

    def _build_tokenizer(self):
        fast_path = self.model_dir / "tokenizer.json"
        try:
            if fast_path.is_file():
                try:
                    return _FastTokenizerBackend(
                        self.model_dir,
                        self.max_length,
                        self.config.get("pad_token"),
                    )
                except ImportError:
                    logger.info("`tokenizers` not installed; using transformers")
            return _TransformersBackend(self.model_dir, self.max_length)
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"Failed to load tokenizer: {exc}") from exc

    def _warmup(self) -> None:
        start = time.perf_counter()
        self._predict_proba(["warm up"], batch_size=1)
        logger.debug("Warm-up took %.1f ms", (time.perf_counter() - start) * 1e3)

    # ----------------------------- inference ------------------------------ #
    def _prepare(self, texts: str | Sequence[str]) -> list[str]:
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, (list, tuple)):
            raise InvalidInputError(
                f"Expected str or list[str], got {type(texts).__name__}"
            )

        prepared: list[str] = []
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise InvalidInputError(
                    f"Item {i} must be str, got {type(t).__name__}"
                )
            t = t[: self.max_chars]
            if self._norm_form:
                t = unicodedata.normalize(self._norm_form, t)
            prepared.append(t)
        return prepared

    def _logits_to_probs(self, logits: Any) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim != 2:
            raise RuntimeError(f"Unexpected logits shape {logits.shape}")

        if logits.shape[1] == 1:  # single-logit sigmoid head
            return 1.0 / (1.0 + np.exp(-logits[:, 0]))

        if not 0 <= self.positive_idx < logits.shape[1]:
            raise RuntimeError(
                f"positive_class_index {self.positive_idx} invalid for "
                f"{logits.shape[1]} classes"
            )
        z = logits - logits.max(axis=1, keepdims=True)  # stable softmax
        e = np.exp(z)
        return e[:, self.positive_idx] / e.sum(axis=1)

    def _predict_proba(
        self, texts: list[str], batch_size: int | None = None
    ) -> np.ndarray:
        bs = batch_size or self.batch_size
        probs = np.empty(len(texts), dtype=np.float64)

        # Sort by length so each batch pads to a similar size.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))

        for start in range(0, len(order), bs):
            idx = order[start : start + bs]
            batch = [texts[i] for i in idx]

            with self._tok_lock:
                enc = self._tokenizer.encode(batch)

            feed = {k: v for k, v in enc.items() if k in self.input_names}
            logits = np.asarray(self.session.run([self.output_name], feed)[0])
            probs[idx] = self._logits_to_probs(logits)

        return probs

    def predict(
        self,
        texts: str | Sequence[str],
        *,
        batch_size: int | None = None,
        echo_text: bool = False,
    ) -> list[dict]:
        """Classify one or many messages. Output order matches input order."""
        prepared = self._prepare(texts)
        if not prepared:
            return []

        start = time.perf_counter()
        probs = self._predict_proba(prepared, batch_size)
        elapsed_ms = (time.perf_counter() - start) * 1e3
        logger.debug("Predicted %d texts in %.1f ms", len(prepared), elapsed_ms)

        results: list[dict] = []
        for i, p in enumerate(probs):
            p = float(p)
            item = {
                "is_smishing": p >= self.threshold,
                "spam_probability": round(p, 4),
                "threshold": self.threshold,
                "model_version": self.model_version,
                "variant": self.variant,
            }
            if echo_text:
                item["text"] = prepared[i]
            results.append(item)
        return results

    def predict_one(self, text: str) -> dict:
        return self.predict(text)[0]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Smishing classifier")
    parser.add_argument("texts", nargs="*", help="Messages to classify")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--fp32", action="store_true", help="Use fp32 model")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    texts = args.texts or [
        "Free recharge Rs 299 active now! Click http://bit.ly/recharge-now",
        "Dinner at 8 pm?",
    ]

    clf = SmishClassifier(args.model_dir, use_int8=not args.fp32)
    for text, res in zip(texts, clf.predict(texts)):
        print(f"{text!r}\n  -> {json.dumps(res, ensure_ascii=False)}")


if __name__ == "__main__":
    _main()