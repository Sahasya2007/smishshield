#!/usr/bin/env python3
"""
Fine-tune a multilingual transformer for smishing / spam SMS detection.
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

MODEL_PRESETS = {
    "mbert": "distilbert/distilbert-base-multilingual-cased",
    "muril": "google/muril-base-cased",
    "xlmr": "xlm-roberta-base",
    "indic": "ai4bharat/IndicBERTv2-MLM-only",
}

ID2LABEL = {0: "ham", 1: "spam"}
LABEL2ID = {"ham": 0, "spam": 1}

TEST_FILES = {
    "unseen_template_test": "unseen_template_test.csv",
    "uci_test": "uci_test.csv",
    "real_test": "real_test.csv",
}

_EVAL_PARAM = (
    "eval_strategy"
    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
    else "evaluation_strategy"
)


# =========================================================
# DATA
# =========================================================


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    for col in ("text", "label"):
        if col not in df.columns:
            raise ValueError(f"{path.name} is missing required column '{col}'")
    if "language" not in df.columns:
        df["language"] = "unknown"
    if "source" not in df.columns:
        df["source"] = "unknown"
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    if not set(df["label"]).issubset({0, 1}):
        raise ValueError(f"{path.name}: labels must be 0/1")
    return df.reset_index(drop=True)


class EncodedDataset(Dataset):
    """Pre-tokenised dataset; padding happens per batch in the collator."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = int(self.labels[idx])
        return item


def encode(tokenizer, df: pd.DataFrame, max_len: int) -> EncodedDataset:
    enc = tokenizer(list(df["text"]), truncation=True, max_length=max_len)
    return EncodedDataset(enc, df["label"].values)


# =========================================================
# TOKENIZER COVERAGE
# =========================================================


def tokenizer_coverage(tokenizer, df: pd.DataFrame, max_len: int):
    """Per-language UNK rate, fragmentation and truncation."""
    unk_id = tokenizer.unk_token_id
    rows, warnings = [], []

    for lang, g in df.groupby("language"):
        texts = list(g["text"])
        ids = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
        n_tok = max(sum(len(x) for x in ids), 1)
        n_unk = sum(1 for x in ids for i in x if i == unk_id) if unk_id is not None else 0
        n_words = max(sum(len(t.split()) for t in texts), 1)
        trunc = float(np.mean([len(x) + 2 > max_len for x in ids]))
        row = {
            "language": lang,
            "rows": len(g),
            "unk_rate": n_unk / n_tok,
            "tokens_per_word": n_tok / n_words,
            "truncated": trunc,
        }
        rows.append(row)

        if row["unk_rate"] > 0.01:
            warnings.append(
                f"[{lang}] {row['unk_rate']:.1%} of tokens are [UNK]; poor script coverage."
            )
        if row["tokens_per_word"] > 4.0:
            warnings.append(
                f"[{lang}] {row['tokens_per_word']:.1f} tokens/word; heavy fragmentation."
            )
        if trunc > 0.01:
            warnings.append(
                f"[{lang}] {trunc:.1%} of messages exceed max_length={max_len}."
            )

    return pd.DataFrame(rows), warnings


# =========================================================
# METRICS
# =========================================================


def softmax_spam(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e[:, 1] / e.sum(axis=1)).astype(np.float64)


def binary_metrics(labels, probs, thr=0.5):
    labels = np.asarray(labels)
    preds = (probs >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    out = {
        "n": int(len(labels)),
        "spam_rate": float(labels.mean()) if len(labels) else None,
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "pr_auc": None,
    }
    if len(set(labels.tolist())) == 2:
        out["pr_auc"] = float(average_precision_score(labels, probs))
    return out


def grouped_metrics(df: pd.DataFrame, probs: np.ndarray, thr: float):
    result = {"overall": binary_metrics(df["label"].values, probs, thr), "by_language": {}}
    for lang, g in df.groupby("language"):
        idx = g.index.values
        result["by_language"][lang] = binary_metrics(df["label"].values[idx], probs[idx], thr)
    return result


def make_compute_metrics(languages):
    languages = np.asarray(languages)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = softmax_spam(np.asarray(logits))
        overall = binary_metrics(labels, probs, 0.5)

        lang_f1 = []
        for lang in np.unique(languages):
            m = languages == lang
            lang_f1.append(binary_metrics(labels[m], probs[m], 0.5)["f1"])

        return {
            "accuracy": overall["accuracy"],
            "precision": overall["precision"],
            "recall": overall["recall"],
            "f1": overall["f1"],
            "pr_auc": overall["pr_auc"] if overall["pr_auc"] is not None else 0.0,
            "lang_macro_f1": float(np.mean(lang_f1)),
        }

    return compute_metrics


def choose_threshold(probs, labels, target_recall: float):
    precision, recall, thr = precision_recall_curve(labels, probs)
    precision, recall = precision[:-1], recall[:-1]
    ok = np.where(recall >= target_recall)[0]
    if len(ok):
        best = ok[np.argmax(precision[ok])]
        return float(thr[best]), f"recall>={target_recall:.2f}"
    f1s = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return float(thr[int(np.argmax(f1s))]), "max_f1 (recall target unreachable)"


# =========================================================
# TRAINER WITH CLASS WEIGHTS
# =========================================================


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device, dtype=logits.dtype)
        loss = F.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


# =========================================================
# ONE TRAINING RUN
# =========================================================


def run_one_seed(seed: int, cfg: dict, tokenizer, data: dict, test_data: dict):
    set_seed(seed)
    run_dir = cfg["out_dir"] / "runs" / f"seed_{seed}"
    final_dir = cfg["out_dir"] / f"seed_{seed}"
    if run_dir.exists():
        shutil.rmtree(run_dir)

    train_ds = encode(tokenizer, data["train"], cfg["max_length"])
    val_ds = encode(tokenizer, data["val"], cfg["max_length"])

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )

    class_weights = None
    if cfg["class_weights"]:
        counts = np.bincount(data["train"]["label"].values, minlength=2).astype(float)
        class_weights = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1.0)), dtype=torch.float)
        print(f"       class weights (ham, spam): {class_weights.tolist()}")

    use_fp16 = torch.cuda.is_available()

    training_kwargs: dict[str, Any] = {
        "output_dir": str(run_dir),
        _EVAL_PARAM: "epoch",
        "save_strategy": "epoch",
        "learning_rate": cfg["lr"],
        "per_device_train_batch_size": cfg["batch_size"],
        "per_device_eval_batch_size": cfg["batch_size"] * 2,
        "num_train_epochs": cfg["epochs"],
        "weight_decay": 0.01,
        "warmup_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "lang_macro_f1",
        "greater_is_better": True,
        "logging_steps": 50,
        "save_total_limit": 1,
        "report_to": "none",
        "seed": seed,
        "data_seed": seed,
        "fp16": use_fp16,
    }

    training_args = cast(Any, TrainingArguments)(**training_kwargs)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if use_fp16 else None),
        compute_metrics=make_compute_metrics(data["val"]["language"].values),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["patience"])],
        class_weights=class_weights,
    )

    trainer.train()
    best_val_metric = trainer.state.best_metric

    val_out = trainer.predict(val_ds)
    val_preds_raw = cast(Any, val_out.predictions)
    val_probs = softmax_spam(np.asarray(val_preds_raw))
    thr, thr_rule = choose_threshold(val_probs, data["val"]["label"].values, cfg["target_recall"])

    result = {
        "seed": seed,
        "best_val_lang_macro_f1": best_val_metric,
        "threshold": thr,
        "threshold_rule": thr_rule,
        "val": {
            "at_0.5": grouped_metrics(data["val"], val_probs, 0.5),
            "at_tuned": grouped_metrics(data["val"], val_probs, thr),
        },
        "tests": {},
    }
    preds_store = {}

    for name, df in test_data.items():
        ds = encode(tokenizer, df, cfg["max_length"])
        test_out = trainer.predict(ds)
        test_preds_raw = cast(Any, test_out.predictions)
        probs = softmax_spam(np.asarray(test_preds_raw))
        result["tests"][name] = {
            "at_0.5": grouped_metrics(df, probs, 0.5),
            "at_tuned": grouped_metrics(df, probs, thr),
        }
        preds_store[name] = probs

    if final_dir.exists():
        shutil.rmtree(final_dir)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    (final_dir / "threshold.json").write_text(
        json.dumps({"threshold": thr, "rule": thr_rule, "id2label": ID2LABEL}, indent=2),
        encoding="utf-8",
    )

    shutil.rmtree(run_dir, ignore_errors=True)
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, preds_store


# =========================================================
# REPORTING
# =========================================================


def aggregate(results: list[dict]) -> dict:
    summary = {}
    names = ["val"] + list(results[0]["tests"].keys())
    for name in names:
        summary[name] = {}
        for mode in ("at_0.5", "at_tuned"):
            summary[name][mode] = {}
            for metric in ("precision", "recall", "f1", "pr_auc"):
                vals = []
                for r in results:
                    block = r["val"] if name == "val" else r["tests"][name]
                    v = block[mode]["overall"][metric]
                    if v is not None:
                        vals.append(v)
                if vals:
                    summary[name][mode][metric] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                    }
    return summary


def print_summary(summary: dict, results: list[dict]):
    print("\n" + "=" * 72)
    print(f"RESULTS (mean +/- std over {len(results)} seed(s))")
    print("=" * 72)
    for name, modes in summary.items():
        tag = " (used for model selection / threshold)" if name == "val" else ""
        print(f"\n{name}{tag}")
        for mode, metrics in modes.items():
            parts = [f"{m}={v['mean']:.3f}+/-{v['std']:.3f}" for m, v in metrics.items()]
            print(f"  {mode:<9} " + "  ".join(parts))

    print("\nPer-language F1 at tuned threshold (best seed):")
    best = max(results, key=lambda r: r["best_val_lang_macro_f1"])
    for name, block in best["tests"].items():
        print(f"  {name}")
        for lang, m in sorted(block["at_tuned"]["by_language"].items()):
            pr = f"{m['pr_auc']:.3f}" if m["pr_auc"] is not None else "n/a"
            print(
                f"    {lang:<12} n={m['n']:<5} P={m['precision']:.3f} "
                f"R={m['recall']:.3f} F1={m['f1']:.3f} PR-AUC={pr}"
            )


def export_errors(out_dir: Path, test_data: dict, preds_store: dict, thr: float):
    err_dir = out_dir / "errors"
    err_dir.mkdir(parents=True, exist_ok=True)
    for name, df in test_data.items():
        probs = preds_store[name]
        pred = (probs >= thr).astype(int)
        wrong = df.copy()
        wrong["spam_prob"] = probs
        wrong["pred"] = pred
        wrong = wrong[wrong["pred"] != wrong["label"]]
        wrong.to_csv(err_dir / f"{name}_errors.csv", index=False, encoding="utf-8")


def read_dataset_manifest(data_dir: Path):
    path = data_dir / "manifest.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("files")
        except Exception:
            return None
    return None


# =========================================================
# MAIN
# =========================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="indic", help=f"preset {list(MODEL_PRESETS)} or a HF model id")
    p.add_argument("--data-dir", default=str(BASE_DIR / "datasets"))
    p.add_argument("--out-dir", default=str(BASE_DIR / "models" / "smish_model"))
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--target-recall", type=float, default=0.95)
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--coverage-only", action="store_true", help="report tokenizer coverage and exit")
    args, _ = p.parse_known_args(argv)
    return args


def main():
    args = parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    model_name = MODEL_PRESETS.get(args.model, args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading datasets from {data_dir}...")
    data = {
        "train": load_split(data_dir / "train.csv"),
        "val": load_split(data_dir / "val.csv"),
    }
    test_data = {
        name: load_split(data_dir / fname)
        for name, fname in TEST_FILES.items()
        if (data_dir / fname).exists()
    }
    print(f"       train={len(data['train'])}  val={len(data['val'])}")
    for name, df in test_data.items():
        print(f"       {name}={len(df)}")
    if "real_test" not in test_data:
        print("       WARNING: no real_test.csv; reported scores do not measure real-world performance.")

    print(f"[2/6] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("[3/6] Tokenizer coverage (train set)")
    cov, cov_warnings = tokenizer_coverage(tokenizer, data["train"], args.max_length)
    print(cov.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    for w in cov_warnings:
        print(f"       WARNING: {w}")
    cov.to_csv(out_dir / "tokenizer_coverage.csv", index=False)

    if args.coverage_only:
        return

    cfg = {
        "model_name": model_name,
        "out_dir": out_dir,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "patience": args.patience,
        "target_recall": args.target_recall,
        "class_weights": not args.no_class_weights,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[4/6] Training on {device.upper()} with seeds {args.seeds}")

    results, all_preds = [], {}
    for seed in args.seeds:
        print(f"\n--- seed {seed} ---")
        res, preds = run_one_seed(seed, cfg, tokenizer, data, test_data)
        results.append(res)
        all_preds[seed] = preds
        print(f"       best val lang_macro_f1={res['best_val_lang_macro_f1']:.4f}  "
              f"threshold={res['threshold']:.3f} ({res['threshold_rule']})")

    print("\n[5/6] Aggregating results...")
    best = max(results, key=lambda r: r["best_val_lang_macro_f1"])
    summary = aggregate(results)
    print_summary(summary, results)

    print("\n[6/6] Saving artifacts...")
    best_dir = out_dir / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(out_dir / f"seed_{best['seed']}", best_dir)
    export_errors(out_dir, test_data, all_preds[best["seed"]], best["threshold"])

    run_info = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()},
        "seeds": args.seeds,
        "best_seed": best["seed"],
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "dataset_files": read_dataset_manifest(data_dir),
        "tokenizer_warnings": cov_warnings,
        "summary": summary,
        "per_seed": results,
    }
    (out_dir / "results.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    print(f"Best model (seed {best['seed']}) : {best_dir}")
    print(f"Results JSON                    : {out_dir / 'results.json'}")
    print(f"Misclassified examples          : {out_dir / 'errors'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        raise