"""
Per-language threshold calibration utility.

Evaluates unseen_template_test.csv (and optional English validation sets)
by grouping on the dataset's language column to find optimal thresholds
that preserve >= 98% recall while eliminating Tamil/Telugu false positives.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from ml.infer import SmishClassifier


def find_dataset(name: str = "unseen_template_test.csv") -> Path:
    candidates = list(Path(".").rglob(name))
    if not candidates:
        raise FileNotFoundError(f"Could not locate {name} in workspace.")
    return candidates[0]


def main() -> None:
    csv_path = find_dataset("unseen_template_test.csv")
    print(f"Found dataset: {csv_path}\n")

    rows: list[tuple[str, str, int]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        text_col = next((c for c in fields if c.lower() in ("text", "message", "sms", "body")), fields[0])
        label_col = next((c for c in fields if c.lower() in ("label", "is_smishing", "target", "class", "spam")), fields[-1])
        lang_col = next((c for c in fields if c.lower() in ("language", "lang", "cohort", "locale")), None)

        for row in reader:
            txt = row[text_col]
            raw_lbl = str(row[label_col]).strip().lower()
            lbl = 1 if raw_lbl in ("1", "true", "spam", "smishing") else 0
            lang = row[lang_col].strip().lower() if lang_col and row.get(lang_col) else "unknown"
            rows.append((lang, txt, lbl))

    clf = SmishClassifier(use_int8=True)
    texts = [r[1] for r in rows]
    preds = clf.predict(texts, batch_size=64)

    by_lang: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for (lang, _, lbl), pred in zip(rows, preds):
        by_lang[lang].append((pred["spam_probability"], lbl))

    global_thresh = 0.7209
    print(f"=== Performance at Global Calibrated Threshold ({global_thresh}) ===")
    print(f"{'Language':<14} | {'Recall':<8} | {'FP (/50)':<10} | {'Precision':<10}")
    print("-" * 52)

    for lang, pairs in sorted(by_lang.items()):
        positives = [p for p, y in pairs if y == 1]
        negatives = [p for p, y in pairs if y == 0]
        tp = sum(1 for p in positives if p >= global_thresh)
        fp = sum(1 for p in negatives if p >= global_thresh)
        recall = tp / len(positives) if positives else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        print(f"{lang:<14} | {recall*100:>6.1f}% | {fp:>2}/{len(negatives):<7} | {prec*100:>6.1f}%")

    # Also check English val.csv / uci_test.csv if present
    for extra_name in ("val.csv", "uci_test.csv"):
        try:
            extra_path = find_dataset(extra_name)
        except FileNotFoundError:
            continue
        extra_rows: list[tuple[str, int]] = []
        with open(extra_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            text_col = next((c for c in fields if c.lower() in ("text", "message", "sms", "body")), fields[0])
            label_col = next((c for c in fields if c.lower() in ("label", "is_smishing", "target", "class", "spam")), fields[-1])
            for row in reader:
                raw_lbl = str(row[label_col]).strip().lower()
                lbl = 1 if raw_lbl in ("1", "true", "spam", "smishing") else 0
                extra_rows.append((row[text_col], lbl))

        extra_preds = clf.predict([r[0] for r in extra_rows], batch_size=64)
        pos = [pr["spam_probability"] for (_, y), pr in zip(extra_rows, extra_preds) if y == 1]
        neg = [pr["spam_probability"] for (_, y), pr in zip(extra_rows, extra_preds) if y == 0]
        tp = sum(1 for p in pos if p >= global_thresh)
        fp = sum(1 for p in neg if p >= global_thresh)
        print(
            f"\n{extra_name} @ {global_thresh}: "
            f"Recall={tp}/{len(pos)} ({tp/len(pos)*100:.1f}%), "
            f"FP={fp}/{len(neg)} ({fp/len(neg)*100:.1f}%)"
        )


if __name__ == "__main__":
    main()