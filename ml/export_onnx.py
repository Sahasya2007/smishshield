#!/usr/bin/env python3
"""
Export the fine-tuned smishing classifier to ONNX (FP32) and INT8 (dynamic quantization),
verify both, and write everything a standalone runtime needs (v2).

What changed vs v1
------------------
* Logits-only wrapper module, so the exported graph has a stable `logits` output.
* Inputs are chosen from the tokenizer (token_type_ids only for models that use them).
* Traced with a variable-length, multi-row batch; old exporter path pinned on new torch.
* Hard parity check: PyTorch vs ONNX FP32 across several batch sizes / lengths.
* ONNX checker, optional pre-processing before quantization, optional per-channel INT8.
* INT8 vs FP32 evaluation on your held-out CSVs at the saved threshold, with regression
  warnings (overall and per-language) and a re-tuned INT8 threshold from validation data.
* Full tokenizer saved next to the models; `inference_config.json` records max_length,
  input names, thresholds, hashes, versions, sizes and latency.
* argparse paths (no hard-coded cwd), size + latency benchmark, optional `--predict`.

Usage
-----
    pip install onnx onnxruntime
    python export_onnx.py --model-dir models/smish_model/best --data-dir datasets
    python export_onnx.py --per-channel --target-recall 0.97
    python export_onnx.py --predict "URGENT: your account is blocked, verify at bit.ly/x1"
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import transformers
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:  # notebooks
    BASE_DIR = Path.cwd()

EVAL_FILES = {
    "val": "val.csv",
    "unseen_template_test": "unseen_template_test.csv",
    "uci_test": "uci_test.csv",
    "real_test": "real_test.csv",
}

# Short multilingual probes for parity testing (mixed scripts, one long to hit truncation).
PARITY_TEXTS = [
    "URGENT: Your SBI account is suspended due to pending KYC. Update now at bit.ly/kyc-verify-123",
    "Your one time password is 482913. Do not share with anyone.",
    "प्रिय ग्राहक, आपका SBI खाता ब्लॉक कर दिया गया है। तुरंत पैन लिंक करें bit.ly/kyc-91",
    "Aapka OTP 123456 hai. Kisi ke sath share na karein.",
    "ଆପଣଙ୍କ SBI ଖାତା ବନ୍ଦ ହୋଇଯାଇଛି। ତୁରନ୍ତ KYC ଅପଡେଟ କରନ୍ତୁ bit.ly/kyc-12",
    "உங்கள் வங்கி KYC புதுப்பிக்கப்படவில்லை. உடனடியாக சரிபார்க்கவும் bit.ly/kyc-5",
    "Mee bank KYC pending undi. Account block kakamundey verify cheyandi bit.ly/kyc-77",
    "Final notice: Electricity service may be disconnected unless payment is completed at bit.ly/quick-pay-9 " * 12,
]


# =========================================================
# HELPERS
# =========================================================


def softmax_spam(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e[:, 1] / e.sum(axis=1)).astype(np.float64)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def size_mb(path):
    return Path(path).stat().st_size / (1024 * 1024)


def metrics_at(labels, probs, thr):
    preds = (probs >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def choose_threshold(probs, labels, target_recall):
    precision, recall, thr = precision_recall_curve(labels, probs)
    precision, recall = precision[:-1], recall[:-1]
    ok = np.where(recall >= target_recall)[0]
    if len(ok):
        return float(thr[ok[np.argmax(precision[ok])]]), f"recall>={target_recall:.2f}"
    f1s = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return float(thr[int(np.argmax(f1s))]), "max_f1 (recall target unreachable)"


def load_eval_csv(path):
    df = pd.read_csv(path, encoding="utf-8")
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path.name} needs 'text' and 'label' columns")
    if "language" not in df.columns:
        df["language"] = "unknown"
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


# =========================================================
# EXPORT
# =========================================================


class LogitsWrapper(torch.nn.Module):
    """Returns a plain logits tensor so the ONNX output is unambiguous."""

    def __init__(self, model, use_token_type_ids):
        super().__init__()
        self.model = model
        self.use_token_type_ids = use_token_type_ids

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "return_dict": False}
        if self.use_token_type_ids:
            kwargs["token_type_ids"] = token_type_ids
        return self.model(**kwargs)[0]


def load_torch_model(model_dir):
    try:  # eager attention traces most reliably
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, attn_implementation="eager")
    except (TypeError, ValueError):
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model.eval().cpu()


def export_fp32(model, tokenizer, path, max_length, opset):
    input_names = [n for n in ("input_ids", "attention_mask", "token_type_ids")
                   if n in tokenizer.model_input_names]
    use_tt = "token_type_ids" in input_names

    # Variable-length, multi-row example so the traced graph does not bake in shapes.
    enc = tokenizer(
        [PARITY_TEXTS[0], "Hi"],
        padding=True, truncation=True, max_length=max_length, return_tensors="pt",
    )
    example = tuple(enc[n] for n in input_names)

    wrapper = LogitsWrapper(model, use_tt).eval()
    dynamic_axes = {n: {0: "batch_size", 1: "sequence_length"} for n in input_names}
    dynamic_axes["logits"] = {0: "batch_size"}

    export_kwargs = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False  # keep the stable TorchScript exporter path

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example,
            str(path),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            **export_kwargs,
        )

    onnx.checker.check_model(str(path))
    return input_names


def quantize_int8(fp32_path, int8_path, per_channel):
    source = fp32_path
    pre_path = fp32_path.with_name("model_preprocessed.onnx")
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process

        quant_pre_process(str(fp32_path), str(pre_path))
        source = pre_path
    except Exception as exc:  # pre-processing is optional
        print(f"       (skipping quantization pre-process: {exc})")

    quantize_dynamic(
        model_input=str(source),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
    )
    if pre_path.exists():
        pre_path.unlink()
    onnx.checker.check_model(str(int8_path))


# =========================================================
# INFERENCE / VERIFICATION
# =========================================================


def make_session(path):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def run_ort(session, tokenizer, texts, max_length, batch_size=32):
    names = {i.name for i in session.get_inputs()}
    chunks = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            list(texts[i:i + batch_size]),
            padding=True, truncation=True, max_length=max_length, return_tensors="np",
        )
        feed = {k: v.astype(np.int64) for k, v in enc.items() if k in names}
        chunks.append(session.run(["logits"], feed)[0])
    return np.concatenate(chunks) if chunks else np.zeros((0, 2), dtype=np.float32)


def run_torch(model, tokenizer, texts, max_length):
    enc = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    allowed = set(inspect.signature(model.forward).parameters)
    feed = {k: v for k, v in enc.items() if k in allowed}
    with torch.no_grad():
        out = model(**feed)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return logits.detach().cpu().numpy()


def parity_check(model, tokenizer, fp32_sess, int8_sess, max_length, atol=1e-3):
    """PyTorch vs ONNX FP32 must match across batch sizes and padding lengths."""
    worst = 0.0
    for bs in (1, 3, len(PARITY_TEXTS)):
        texts = PARITY_TEXTS[:bs]
        ref = run_torch(model, tokenizer, texts, max_length)
        got = run_ort(fp32_sess, tokenizer, texts, max_length, batch_size=bs)
        worst = max(worst, float(np.abs(ref - got).max()))
    if worst > atol:
        raise RuntimeError(
            f"FP32 ONNX does not match PyTorch (max |diff|={worst:.2e} > {atol}). Do not deploy."
        )

    ref_p = softmax_spam(run_ort(fp32_sess, tokenizer, PARITY_TEXTS, max_length))
    q_p = softmax_spam(run_ort(int8_sess, tokenizer, PARITY_TEXTS, max_length))
    return {
        "fp32_vs_torch_max_abs_logit_diff": worst,
        "int8_vs_fp32_max_abs_prob_diff": float(np.abs(ref_p - q_p).max()),
        "int8_vs_fp32_label_agreement": float(np.mean((ref_p >= 0.5) == (q_p >= 0.5))),
    }


def benchmark(session, tokenizer, max_length, batch, runs=50):
    enc = tokenizer(
        [PARITY_TEXTS[0]] * batch, padding=True, truncation=True,
        max_length=max_length, return_tensors="np",
    )
    names = {i.name for i in session.get_inputs()}
    feed = {k: v.astype(np.int64) for k, v in enc.items() if k in names}
    for _ in range(5):
        session.run(["logits"], feed)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(["logits"], feed)
        times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def evaluate_quantization(fp32_sess, int8_sess, tokenizer, data_dir, max_length,
                          fp32_thr, target_recall):
    """FP32 vs INT8 on held-out data at the saved threshold, plus a re-tuned INT8 threshold."""
    report, warnings, int8_thr, int8_rule = {}, [], None, None
    if not data_dir.exists():
        return report, warnings, int8_thr, int8_rule

    for name, fname in EVAL_FILES.items():
        path = data_dir / fname
        if not path.exists():
            continue
        df = load_eval_csv(path)
        texts, labels = df["text"].tolist(), df["label"].values
        p32 = softmax_spam(run_ort(fp32_sess, tokenizer, texts, max_length))
        p8 = softmax_spam(run_ort(int8_sess, tokenizer, texts, max_length))

        entry = {
            "n": len(df),
            "label_agreement": float(np.mean((p32 >= fp32_thr) == (p8 >= fp32_thr))),
            "fp32": metrics_at(labels, p32, fp32_thr),
            "int8_at_fp32_threshold": metrics_at(labels, p8, fp32_thr),
            "by_language": {},
        }
        if name == "val" and len(set(labels.tolist())) == 2:
            int8_thr, int8_rule = choose_threshold(p8, labels, target_recall)
            entry["int8_at_retuned_threshold"] = metrics_at(labels, p8, int8_thr)

        drop = entry["fp32"]["recall"] - entry["int8_at_fp32_threshold"]["recall"]
        if drop > 0.01:
            warnings.append(f"[{name}] INT8 recall is {drop:.3f} lower than FP32 at the saved threshold.")

        for lang, g in df.groupby("language"):
            if len(g) < 20 or len(set(g["label"])) < 2:
                continue
            idx = g.index.values
            r32 = metrics_at(labels[idx], p32[idx], fp32_thr)["recall"]
            r8 = metrics_at(labels[idx], p8[idx], fp32_thr)["recall"]
            entry["by_language"][lang] = {"n": len(g), "recall_fp32": r32, "recall_int8": r8}
            if r32 - r8 > 0.02:
                warnings.append(f"[{name}/{lang}] INT8 recall drop {r32 - r8:.3f} (n={len(g)}).")
        report[name] = entry

    return report, warnings, int8_thr, int8_rule


# =========================================================
# MAIN
# =========================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=str(BASE_DIR / "models" / "smish_model" / "best"))
    p.add_argument("--out-dir", default=None, help="default: <model-dir>/../onnx")
    p.add_argument("--data-dir", default=str(BASE_DIR / "datasets"))
    p.add_argument("--max-length", type=int, default=None, help="default: from training results.json, else 128")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--per-channel", action="store_true", help="per-channel INT8 weights (often more accurate)")
    p.add_argument("--target-recall", type=float, default=0.95)
    p.add_argument("--no-eval", action="store_true", help="skip INT8 vs FP32 dataset evaluation")
    p.add_argument("--predict", nargs="*", default=None, help="run the INT8 model on these texts")
    args, _ = p.parse_known_args(argv)
    return args


def resolve_max_length(model_dir, cli_value):
    if cli_value:
        return cli_value
    results = model_dir.parent / "results.json"
    if results.exists():
        try:
            return int(json.loads(results.read_text(encoding="utf-8"))["config"]["max_length"])
        except Exception:
            pass
    return 128


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir) if args.out_dir else model_dir.parent / "onnx"
    data_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = out_dir / "model.onnx"
    int8_path = out_dir / "model_int8.onnx"
    max_length = resolve_max_length(model_dir, args.max_length)

    thr_file = model_dir / "threshold.json"
    fp32_thr = 0.5
    if thr_file.exists():
        fp32_thr = float(json.loads(thr_file.read_text(encoding="utf-8")).get("threshold", 0.5))
    else:
        print("WARNING: threshold.json not found; using 0.5.")

    print(f"[1/6] Loading checkpoint from {model_dir} (max_length={max_length})")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = load_torch_model(model_dir)

    print(f"[2/6] Exporting FP32 ONNX (opset {args.opset})")
    input_names = export_fp32(model, tokenizer, fp32_path, max_length, args.opset)
    print(f"       inputs={input_names}  size={size_mb(fp32_path):.1f} MB")

    print(f"[3/6] INT8 dynamic quantization (per_channel={args.per_channel})")
    quantize_int8(fp32_path, int8_path, args.per_channel)
    print(f"       size={size_mb(int8_path):.1f} MB")

    print("[4/6] Verifying outputs")
    fp32_sess, int8_sess = make_session(fp32_path), make_session(int8_path)
    parity = parity_check(model, tokenizer, fp32_sess, int8_sess, max_length)
    print(f"       torch vs FP32 max |logit diff| : {parity['fp32_vs_torch_max_abs_logit_diff']:.2e}")
    print(f"       FP32 vs INT8 max |prob diff|   : {parity['int8_vs_fp32_max_abs_prob_diff']:.4f}")
    print(f"       FP32 vs INT8 label agreement   : {parity['int8_vs_fp32_label_agreement']:.3f}")

    eval_report, warnings, int8_thr, int8_rule = {}, [], None, None
    if not args.no_eval:
        print("[5/6] Evaluating INT8 vs FP32 on held-out data")
        eval_report, warnings, int8_thr, int8_rule = evaluate_quantization(
            fp32_sess, int8_sess, tokenizer, data_dir, max_length, fp32_thr, args.target_recall
        )
        if not eval_report:
            print(f"       no evaluation CSVs found in {data_dir}")
        for name, e in eval_report.items():
            a, b = e["fp32"], e["int8_at_fp32_threshold"]
            print(
                f"       {name:<22} n={e['n']:<6} "
                f"FP32 P/R/F1={a['precision']:.3f}/{a['recall']:.3f}/{a['f1']:.3f}  "
                f"INT8 P/R/F1={b['precision']:.3f}/{b['recall']:.3f}/{b['f1']:.3f}  "
                f"agree={e['label_agreement']:.3f}"
            )
        for w in warnings:
            print(f"       WARNING: {w}")
        if int8_thr is not None:
            print(f"       re-tuned INT8 threshold from val: {int8_thr:.4f} ({int8_rule}) "
                  f"vs FP32 threshold {fp32_thr:.4f}")
    else:
        print("[5/6] Skipping dataset evaluation (--no-eval)")

    print("[6/6] Benchmarking and writing runtime files")
    bench = {
        name: {
            f"batch_{b}": benchmark(sess, tokenizer, max_length, b)
            for b in (1, 16)
        }
        for name, sess in (("fp32", fp32_sess), ("int8", int8_sess))
    }
    for b in ("batch_1", "batch_16"):
        print(f"       {b:<9} FP32 {bench['fp32'][b]['median_ms']:.1f} ms | "
              f"INT8 {bench['int8'][b]['median_ms']:.1f} ms (median)")

    tokenizer.save_pretrained(str(out_dir))
    for fname in ("config.json", "threshold.json"):
        src = model_dir / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)

    recommended = int8_thr if int8_thr is not None else fp32_thr
    config = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "max_length": max_length,
        "input_names": input_names,
        "output_names": ["logits"],
        "id2label": {"0": "ham", "1": "spam"},
        "thresholds": {
            "fp32": fp32_thr,
            "int8_retuned": int8_thr,
            "int8_retuned_rule": int8_rule,
            "recommended_for_int8": recommended,
        },
        "models": {
            "fp32": {"file": fp32_path.name, "size_mb": size_mb(fp32_path), "sha256": sha256_file(fp32_path)},
            "int8": {"file": int8_path.name, "size_mb": size_mb(int8_path), "sha256": sha256_file(int8_path)},
        },
        "quantization": {"type": "dynamic", "weight_type": "QInt8", "per_channel": args.per_channel},
        "parity": parity,
        "evaluation": eval_report,
        "warnings": warnings,
        "benchmark_ms": bench,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
    }
    (out_dir / "inference_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\nExport complete.")
    print(f"  FP32 : {fp32_path} ({size_mb(fp32_path):.1f} MB)")
    print(f"  INT8 : {int8_path} ({size_mb(int8_path):.1f} MB)")
    print(f"  Config: {out_dir / 'inference_config.json'}")
    print(f"  Use threshold {recommended:.4f} with the INT8 model; do not pass token_type_ids unless "
          f"listed in input_names {input_names}.")

    if args.predict:
        probs = softmax_spam(run_ort(int8_sess, tokenizer, args.predict, max_length))
        print("\nPredictions (INT8):")
        for text, p in zip(args.predict, probs):
            label = "spam" if p >= recommended else "ham"
            print(f"  {label:<4} p_spam={p:.3f}  {text[:90]}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        raise