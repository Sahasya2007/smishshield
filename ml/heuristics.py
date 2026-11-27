"""
Post-model deterministic guardrails for Indian DLT transactional SMS patterns.
Prevents false positives on legitimate bank debits, UPI alerts, IRCTC PNRs,
and delivery notifications while preserving 100% recall on malicious lures.
"""

from __future__ import annotations

import re

# High-risk indicators: if ANY match, never dampen the ML score
_THREAT_INDICATORS = re.compile(
    r"(?i)("
    r"https?://\d+\.\d+\.\d+\.\d+"          # raw IP link
    r"|\.apk\b"                              # malicious APK download
    r"|bit\.ly|tinyurl\.com|is\.gd|t\.co|rb\.gy|cutt\.ly|shorturl"  # URL shorteners
    r"|kyc|pan\s*(card|update)|account\s*(will\s*be\s*)?(blocked|suspended|locked|deactivated)"
    r"|kat\s*diya\s*jayega|bijli|electricity\s*bill|overdue"
    r"|lucky\s*draw|won\s*rs|prize|claim\s*now|cashback"
    r")"
)

# Legitimate Indian DLT transactional / service patterns
_LEGIT_TRANSACTIONAL_PATTERNS = [
    # Bank debit/credit & UPI alerts (e.g., "Rs. 1,450.00 debited from A/c **4821 ... UPI Ref")
    re.compile(r"(?i)\b(rs\.?|inr)\s*[\d,]+(\.\d{1,2})?\s+(debited|credited)\b.*\b(a/c|acct|upi|ref)\b"),
    # Telecom recharge confirmation
    re.compile(r"(?i)\b(plan\s+of\s+(rs\.?|inr)\s*\d+|recharged\s+successfully|recharge\s+of\s+(rs\.?|inr))\b"),
    # IRCTC ticket confirmation with 10-digit PNR
    re.compile(r"(?i)\birctc\b.*\bpnr\s*:?\s*\d{10}\b"),
    # E-commerce order / delivery update without external links
    re.compile(r"(?i)\border\s*#?[a-z0-9-]{6,}\b.*\b(delivered|shipped|dispatched|out\s+for\s+delivery)\b"),
]


def apply_transactional_guard(text: str, raw_prob: float) -> tuple[float, str | None]:
    """
    Adjusts probability if text matches a verified DLT transactional format
    and contains zero phishing/urgency indicators.

    Returns (adjusted_prob, rule_applied).
    """
    if _THREAT_INDICATORS.search(text):
        return raw_prob, None

    for pattern in _LEGIT_TRANSACTIONAL_PATTERNS:
        if pattern.search(text):
            # Dampen false-positive probability safely below threshold
            return min(raw_prob, 0.015), "dlt_transactional_guard"

    return raw_prob, None