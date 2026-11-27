"""
Benchmark & edge-case verification suite for SmishClassifier.

Measures:
1. Single-message and micro-batch latency percentiles (p50, p95, p99)
2. False-positive resistance on legitimate Indian banking, UPI, and OTP messages
3. Recall on regional/Hinglish smishing lures and Unicode homoglyph evasion
"""

from __future__ import annotations

import statistics
import time
from ml.infer import SmishClassifier

LEGIT_INDIAN_SMS = [
    "HDFC Bank: Rs. 1,450.00 debited from A/c **4821 on 27-11-26 to VPA swiggy@axisbank. UPI Ref 632910482910.",
    "482910 is your OTP for SBI NetBanking login. Valid for 5 mins. Do not share with anyone.",
    "Dear Customer, your Jio plan of Rs 299 has been recharged successfully. Enjoy unlimited 5G data.",
    "IRCTC Ticket Booked: PNR 4521890321, Train 12801 PURUSHOTTAM EXP, 28-Nov, B2 42. Have a safe journey.",
    "Your Amazon order #408-9182736-1234561 has been delivered. Thank you for shopping with us.",
]

SMISHING_EDGE_CASES = [
    "URGENT: Your SBI YONO account will be blocked today. Update PAN card immediately: http://bit.ly/sbi-kyc-update",
    "Aapka bijli connection aaj raat 9:30 baje kat diya jayega kyunki pichla bill update nahi hua. Call 9876543210",
    "Congratulations! You won Rs 25,00,000 in KBC Lucky Draw. Click http://tinyurl.com/kbc-claim to claim prize.",
    "Dear user, your electricity bill is overdue. Download official update app: http://192.168.1.45/bill_pay.apk",
    # Homoglyph / Cyrillic 'а' and 'о' evasion test
    "Your bаnk аccоunt is suspended. Verify KYC at http://bit.ly/verify-nоw",
]


def run_benchmark() -> None:
    clf = SmishClassifier(use_int8=True)
    print(f"Loaded {clf.model_version} ({clf.variant}) | Threshold: {clf.threshold:.4f}\n")

    # 1. Edge-case accuracy check
    print("=== 1. Legitimate Indian SMS (Expected: is_smishing=False) ===")
    legit_preds = clf.predict(LEGIT_INDIAN_SMS)
    false_positives = 0
    for msg, res in zip(LEGIT_INDIAN_SMS, legit_preds):
        flag = "FAIL (FP)" if res["is_smishing"] else "PASS"
        if res["is_smishing"]:
            false_positives += 1
        print(f"[{flag}] prob={res['spam_probability']:.4f} | {msg[:65]}...")

    print("\n=== 2. Multilingual & Adversarial Smishing (Expected: is_smishing=True) ===")
    smish_preds = clf.predict(SMISHING_EDGE_CASES)
    false_negatives = 0
    for msg, res in zip(SMISHING_EDGE_CASES, smish_preds):
        flag = "PASS" if res["is_smishing"] else "FAIL (FN)"
        if not res["is_smishing"]:
            false_negatives += 1
        print(f"[{flag}] prob={res['spam_probability']:.4f} | {msg[:65]}...")

    # 2. Single-message latency (500 iterations)
    sample_pool = (LEGIT_INDIAN_SMS + SMISHING_EDGE_CASES) * 50
    latencies_ms: list[float] = []
    for text in sample_pool:
        t0 = time.perf_counter()
        clf.predict_one(text)
        latencies_ms.append((time.perf_counter() - t0) * 1e3)

    latencies_ms.sort()
    p50 = statistics.median(latencies_ms)
    p95 = latencies_ms[int(len(latencies_ms) * 0.95)]
    p99 = latencies_ms[int(len(latencies_ms) * 0.99)]

    # 3. Batch throughput (500 messages in one call)
    t0 = time.perf_counter()
    clf.predict(sample_pool, batch_size=64)
    batch_total_ms = (time.perf_counter() - t0) * 1e3
    throughput = len(sample_pool) / (batch_total_ms / 1e3)

    print("\n=== 3. Performance Summary ===")
    print(f"Legit SMS False Positives : {false_positives}/{len(LEGIT_INDIAN_SMS)}")
    print(f"Smishing Missed (FN)      : {false_negatives}/{len(SMISHING_EDGE_CASES)}")
    print(f"Single-SMS Latency (p50)  : {p50:.2f} ms")
    print(f"Single-SMS Latency (p95)  : {p95:.2f} ms")
    print(f"Single-SMS Latency (p99)  : {p99:.2f} ms")
    print(f"Batch Throughput (bs=64)  : {throughput:.0f} msgs/sec ({batch_total_ms:.1f} ms for {len(sample_pool)} msgs)")


if __name__ == "__main__":
    run_benchmark()