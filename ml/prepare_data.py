#!/usr/bin/env python3
"""
Multilingual smishing / spam SMS dataset builder (v2).

What changed vs v1
------------------
* Template-grouped splits with a ratio-based allocation, `template_id` on every
  synthetic row, and optional extra templates from `extra_templates.jsonl`.
* Real held-out data: UCI is split train/val/test (70/15/15), and a hand-labelled
  multilingual set (`real_test_labeled.csv`) is picked up if you provide one.
* Anti-shortcut design: spam links include short, look-alike and IP links; ham
  includes legitimate short links, phone numbers and toll-free numbers.
  Spam without links exists in English and Hinglish.
* Label-independent augmentation (sender tags, spacing, casing, suffixes) so
  low-variability templates still produce diverse text without adding a shortcut.
* Round-robin template sampling, explicit shortfall tracking (`--strict` fails on it).
* Near-duplicate aware: UCI is de-duplicated on a normalised form (URLs, numbers and
  bank names masked), label conflicts are dropped, and cross-split overlap is
  removed on both exact and normalised text.
* Template validation, local RNGs per stage, loud failure on missing UCI data,
  a manifest.json with hashes + versions, and a richer quality report.

Usage
-----
    python build_smishing_dataset.py
    python build_smishing_dataset.py --seed 7 --out ./datasets --strict
    python build_smishing_dataset.py --real-test my_labeled.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import random
import re
import string
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from string import Formatter

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

log = logging.getLogger("smishing_dataset")

# =========================================================
# CONFIGURATION
# =========================================================

SEED = 42

TRAIN_SYNTHETIC_PER_LANGUAGE = 300
VAL_SYNTHETIC_PER_LANGUAGE = 100
TEST_SYNTHETIC_PER_LANGUAGE = 100
SYNTH_SPAM_RATIO = 0.5

UCI_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
UCI_URL = (
    "https://archive.ics.uci.edu/static/public/"
    "228/sms+spam+collection.zip"
)
UCI_MEMBER = "SMSSpamCollection"

MIN_REAL_PER_LANGUAGE = 30  # warn if the hand-labelled set is thinner than this

try:
    DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "datasets"
except NameError:  # notebooks / REPL
    DEFAULT_DATASET_DIR = Path.cwd() / "datasets"

# =========================================================
# CONSTANTS
# =========================================================

BANKS = [
    "SBI", "HDFC", "ICICI", "Axis", "PNB", "Canara",
    "Kotak", "IndusInd", "Bank of Baroda", "Union Bank",
]

UTILITIES = [
    "TPCODL", "Bescom", "TNEB", "KSEB",
    "TSSPDCL", "Bijli Vibhag", "MSEDCL", "WBSEDCL",
]

SHORT_DOMAINS = ["bit.ly", "tinyurl.com", "is.gd", "t.co", "cutt.ly"]

BENIGN_DOMAINS = [
    "sbi.co.in", "hdfcbank.com", "icicibank.com",
    "axisbank.com", "pnbindia.in", "canarabank.com",
]

BENIGN_PATHS = [
    "login", "account", "netbanking", "offers",
    "support", "services", "mobile-banking",
]

LOOKALIKE_WORDS = ["kyc", "secure", "verify", "update", "care"]
LOOKALIKE_TLDS = ["xyz", "top", "info", "online", "site", "co.in.net"]

PHONE_PREFIXES = ["98", "94", "97", "89", "70", "88", "91"]

AMOUNTS = [500, 1200, 2500, 5000, 15000, 25000, 50000, 75000, 100000]

SLUG_CHARS = string.ascii_letters + string.digits

# =========================================================
# RANDOM VALUE GENERATORS (all take an explicit RNG)
# =========================================================


def rand_phone(rng):
    return f"{rng.choice(PHONE_PREFIXES)}{rng.randint(10000000, 99999999)}"


def rand_tollfree(rng):
    return f"1800{rng.randint(1000000, 9999999)}"


def rand_amount(rng):
    return f"{rng.choice(AMOUNTS):,}"


def rand_time(rng):
    hour = rng.choice([7, 8, 9, 10, 11, 12])
    minute = rng.choice(["00", "15", "30", "45"])
    return f"{hour}:{minute} {rng.choice(['AM', 'PM'])}"


def rand_otp(rng):
    return str(rng.randint(100000, 999999))


def rand_slug(rng, n=6):
    return "".join(rng.choice(SLUG_CHARS) for _ in range(n))


def rand_spam_link(rng, keyword, bank):
    """Mix of short links, look-alike domains and raw-IP links."""
    kind = rng.choices(["short", "lookalike", "ip"], weights=[0.55, 0.35, 0.10])[0]

    if kind == "short":
        return f"{rng.choice(SHORT_DOMAINS)}/{keyword}-{rng.randint(10, 999)}"

    if kind == "lookalike":
        slug = re.sub(r"[^a-z0-9]", "", bank.lower())
        host = (
            f"{slug}{rng.choice(['-', ''])}"
            f"{rng.choice(LOOKALIKE_WORDS)}.{rng.choice(LOOKALIKE_TLDS)}"
        )
        return f"{rng.choice(['https://', 'http://', ''])}{host}/{keyword}"

    return (
        f"http://{rng.randint(20, 220)}.{rng.randint(0, 255)}."
        f"{rng.randint(0, 255)}.{rng.randint(1, 254)}/{keyword}"
    )


def rand_benign_link(rng):
    """Mostly official domains, but legitimate senders sometimes use shorteners."""
    if rng.random() < 0.20:
        return f"{rng.choice(SHORT_DOMAINS)}/{rand_slug(rng)}"
    return f"https://www.{rng.choice(BENIGN_DOMAINS)}/{rng.choice(BENIGN_PATHS)}"


# =========================================================
# TEMPLATES
#
# (label, text)   1 = smishing / spam, 0 = ham
#
# NOTE: Indic-language templates should be reviewed by native speakers.
# Add more via datasets/extra_templates.jsonl:
#   {"language": "or", "label": 1, "text": "... {bank} ... {link_kyc}"}
# =========================================================

TEMPLATE_SPECS = {
    "en": [
        (1, "URGENT: Your {bank} account is suspended due to pending KYC. Update now at {link_kyc}"),
        (1, "Your {bank} account will be blocked today. Complete KYC immediately: {link_kyc}"),
        (1, "WARNING: Your {bank} banking access has been restricted. Verify your account at {link_kyc}"),
        (1, "Power cut tonight at {time} due to unpaid electricity bill. Pay now at {link_bill}"),
        (1, "Your {utility} electricity bill is overdue. Connection will be disconnected today. Pay {link_bill}"),
        (1, "Final notice: Electricity service may be disconnected unless payment is completed at {link_bill}"),
        (1, "Congratulations! You won Rs {amount} cash reward. Claim through UPI: upi://pay?pa=claim{otp}@upi"),
        (1, "You have received a reward of Rs {amount}. Approve the UPI collect request to receive it."),
        (1, "Your account has been selected for a Rs {amount} cashback. Confirm your UPI request now."),
        (1, "Your {bank} KYC is incomplete. Verify your bank account immediately to avoid suspension."),
        (1, "{bank} ALERT: Unusual activity detected on your account. Call {phone} immediately to avoid account block."),
        (1, "Dear customer, your {bank} debit card is blocked. Call {phone} now to reactivate."),
        (0, "Your one time password is {otp} for txn of Rs {amount}. Do not share with anyone."),
        (0, "Dear Customer, your {bank} account ending in {otp_last4} has been credited with Rs {amount}."),
        (0, "Power cut scheduled at {time} tomorrow for routine transformer maintenance by {utility}."),
        (0, "Your {bank} transaction of Rs {amount} was successful. Ref: {otp}."),
        (0, "For banking services, visit the official website {benign_link}."),
        (0, "Scheduled maintenance will temporarily affect {utility} services at {time}."),
        (0, "Your {bank} account statement is available through the official banking application."),
        (0, "{bank}: For any queries, contact customer care at {tollfree}. Never share your PIN or OTP."),
        (0, "Your {bank} statement is ready. View it at {benign_link}."),
    ],
    "hi_hinglish": [
        (1, "{utility} bijli bill pending hai. Aaj raat connection cut ho jayega. Pay karein {link_bill}"),
        (1, "à¤ªà¥à¤°à¤¿à¤¯ à¤—à¥à¤°à¤¾à¤¹à¤•, à¤†à¤ªà¤•à¤¾ {bank} à¤–à¤¾à¤¤à¤¾ à¤¬à¥à¤²à¥‰à¤• à¤•à¤° à¤¦à¤¿à¤¯à¤¾ à¤—à¤¯à¤¾ à¤¹à¥ˆà¥¤ à¤¤à¥à¤°à¤‚à¤¤ à¤ªà¥ˆà¤¨ à¤²à¤¿à¤‚à¤• à¤•à¤°à¥‡à¤‚ {link_kyc}"),
        (1, "Aapka {bank} account KYC pending hone ke karan band ho jayega. Abhi verify karein {link_kyc}"),
        (1, "Badhai ho! Aapne Rs {amount} reward jeeta hai. UPI se collect karein upi://pay?pa=cash{otp}@upi"),
        (1, "Aapka electricity bill pending hai. Connection aaj disconnect ho sakta hai. Payment karein {link_bill}"),
        (1, "Aapka bank KYC update nahi hua hai. Account block hone se pehle verify karein {link_kyc}"),
        (1, "Aapke {bank} account me suspicious activity mili hai. Turant call karein {phone}."),
        (0, "Aapka OTP {otp} hai. {bank} login ke liye ise kisi ke sath share na karein."),
        (0, "à¤†à¤ªà¤•à¥‡ à¤–à¤¾à¤¤à¥‡ à¤®à¥‡à¤‚ à¤°à¥ {amount} à¤•à¥à¤°à¥‡à¤¡à¤¿à¤Ÿ à¤•à¤¿à¤ à¤—à¤ à¤¹à¥ˆà¤‚à¥¤ {bank} à¤®à¥‡à¤‚ à¤¶à¥‡à¤· à¤°à¤¾à¤¶à¤¿ à¤°à¥ {amount2} à¤¹à¥ˆà¥¤"),
        (0, "Kripya dhyan dein, {utility} line maintenance ke karan bijli {time} tak band rahegi."),
        (0, "Aapka {bank} transaction Rs {amount} ka successful raha. Ref: {otp}."),
        (0, "Official banking services ke liye {benign_link} visit karein."),
        (0, "Kisi bhi sahayata ke liye {bank} customer care ko {tollfree} par call karein."),
        (0, "Aapka {bank} statement taiyar hai. Dekhne ke liye {benign_link} par jayein."),
    ],
    "or": [
        (1, "à¬œà¬°à­à¬°à­€ à¬¸à­‚à¬šà¬¨à¬¾: à¬†à¬ªà¬£à¬™à­à¬• à¬¬à¬¿à¬¦à­à­Ÿà­à¬¤ à¬¬à¬¿à¬²à­ à¬¬à¬¾à¬•à¬¿ à¬…à¬›à¬¿à¥¤ à¬†à¬œà¬¿ à¬°à¬¾à¬¤à¬¿ {time} à¬°à­‡ à¬²à¬¾à¬‡à¬¨ à¬•à¬Ÿà¬¿à¬¯à¬¿à¬¬à¥¤ à¬•à¬²à­ à¬•à¬°à¬¨à­à¬¤à­ {phone} à¬•à¬¿à¬®à­à¬¬à¬¾ {link_bill}"),
        (1, "à¬†à¬ªà¬£à¬™à­à¬• {bank} à¬–à¬¾à¬¤à¬¾ à¬¬à¬¨à­à¬¦ à¬¹à­‹à¬‡à¬¯à¬¾à¬‡à¬›à¬¿à¥¤ à¬¤à­à¬°à¬¨à­à¬¤ KYC à¬…à¬ªà¬¡à­‡à¬Ÿ à¬•à¬°à¬¨à­à¬¤à­ {link_kyc}"),
        (1, "Apanka bijuli bill baki achi. Rati re line kati jiba. Jogajoga karantu {phone}"),
        (1, "à¬†à¬ªà¬£à¬™à­à¬• à¬¬à­à­Ÿà¬¾à¬™à­à¬• à¬–à¬¾à¬¤à¬¾ KYC à¬…à¬ªà¬¡à­‡à¬Ÿà­ à¬¹à­‹à¬‡à¬¨à¬¾à¬¹à¬¿à¬à¥¤ à¬¤à­à¬°à¬¨à­à¬¤ à¬¯à¬¾à¬žà­à¬š à¬•à¬°à¬¨à­à¬¤à­ {link_kyc}"),
        (0, "à¬†à¬ªà¬£à¬™à­à¬•à¬° à¬à¬•à¬•à¬¾à¬³à­€à¬¨ à¬ªà¬¾à¬¸à­±à¬¾à¬°à­à¬¡ (OTP) à¬¹à­‡à¬‰à¬›à¬¿ {otp}à¥¤ à¬à¬¹à¬¾à¬•à­ à¬•à¬¾à¬¹à¬¾ à¬¸à¬¹à¬¿à¬¤ à¬¸à­‡à­Ÿà¬¾à¬° à¬•à¬°à¬¨à­à¬¤à­ à¬¨à¬¾à¬¹à¬¿à¬à¥¤"),
        (0, "Apanka account re Rs {amount} credit heichi. Balance janiba pain check karantu."),
        (0, "Bijuli line maintenance pain {time} re samayika seva band rahiba."),
        (0, "Apanka {bank} transaction Rs {amount} successful heichi. Ref {otp}."),
        (0, "Banking service pain official website {benign_link} use karantu."),
    ],
    "te": [
        (1, "à°®à±à°–à±à°¯ à°—à°®à°¨à°¿à°•: à°®à±€ à°µà°¿à°¦à±à°¯à±à°¤à± à°¬à°¿à°²à±à°²à± à°¬à°•à°¾à°¯à°¿ à°‰à°‚à°¦à°¿. à°ˆ à°°à°¾à°¤à±à°°à°¿ {time} à°•à± à°ªà°µà°°à± à°•à°Ÿà± à°…à°µà±à°¤à±à°‚à°¦à°¿. à°šà±†à°²à±à°²à°¿à°‚à°šà°‚à°¡à°¿ {link_bill}"),
        (1, "à°®à±€ {bank} à°–à°¾à°¤à°¾ à°¤à°¾à°¤à±à°•à°¾à°²à°¿à°•à°‚à°—à°¾ à°¨à°¿à°²à°¿à°ªà°¿à°µà±‡à°¯à°¬à°¡à°¿à°‚à°¦à°¿. à°µà±†à°‚à°Ÿà°¨à±‡ à°†à°§à°¾à°°à± à°²à°¿à°‚à°•à± à°šà±‡à°¯à°‚à°¡à°¿ {link_kyc}"),
        (1, "Mee {utility} bill pay cheyandi ledante current cut avtundi. Call {phone}"),
        (1, "Mee bank KYC pending undi. Account block kakamundey verify cheyandi {link_kyc}"),
        (1, "à°®à±€ à°¬à±à°¯à°¾à°‚à°•à± KYC à°‡à°‚à°•à°¾ à°ªà±‚à°°à±à°¤à°¿à°•à°¾à°²à±‡à°¦à±. à°µà±†à°‚à°Ÿà°¨à±‡ à°§à±ƒà°µà±€à°•à°°à°¿à°‚à°šà°‚à°¡à°¿ {link_kyc}"),
        (0, "à°®à±€ à°²à°¾à°—à°¿à°¨à± OTP {otp}. à°ˆ à°•à±‹à°¡à±â€Œà°¨à± à°Žà°µà°°à°¿à°¤à±‹à°¨à±‚ à°ªà°‚à°šà±à°•à±‹à°µà°¦à±à°¦à±. {bank} à°…à°²à°°à±à°Ÿà±."),
        (0, "Mee bank account lo Rs {amount} deposit cheyabadindi. Ref UPI/{otp}."),
        (0, "Mee {bank} transaction Rs {amount} successful ayindi."),
        (0, "Official banking services kosam {benign_link} ni use cheyandi."),
    ],
    "ta": [
        (1, "à®Žà®šà¯à®šà®°à®¿à®•à¯à®•à¯ˆ: à®‰à®™à¯à®•à®³à¯ à®®à®¿à®©à¯ à®•à®Ÿà¯à®Ÿà®£à®®à¯ à®šà¯†à®²à¯à®¤à¯à®¤à®ªà¯à®ªà®Ÿà®µà®¿à®²à¯à®²à¯ˆ. à®‡à®©à¯à®±à¯ à®‡à®°à®µà¯ à®®à®¿à®©à¯à®šà®¾à®°à®®à¯ à®¤à¯à®£à¯à®Ÿà®¿à®•à¯à®•à®ªà¯à®ªà®Ÿà¯à®®à¯. à®¤à¯Šà®Ÿà®°à¯à®ªà¯ à®•à¯Šà®³à¯à®³à®µà¯à®®à¯ {phone}"),
        (1, "à®‰à®™à¯à®•à®³à¯ {bank} à®µà®™à¯à®•à®¿ à®•à®£à®•à¯à®•à¯ à®®à¯à®Ÿà®•à¯à®•à®ªà¯à®ªà®Ÿà¯à®Ÿà¯à®³à¯à®³à®¤à¯. à®‰à®Ÿà®©à®Ÿà®¿à®¯à®¾à®• à®šà®°à®¿à®ªà®¾à®°à¯à®•à¯à®• {link_kyc}"),
        (1, "Unga current bill katta villai endral inru iravu power cut aagum. Click {link_bill}"),
        (1, "à®‰à®™à¯à®•à®³à¯ à®µà®™à¯à®•à®¿ KYC à®ªà¯à®¤à¯à®ªà¯à®ªà®¿à®•à¯à®•à®ªà¯à®ªà®Ÿà®µà®¿à®²à¯à®²à¯ˆ. à®‰à®Ÿà®©à®Ÿà®¿à®¯à®¾à®• à®šà®°à®¿à®ªà®¾à®°à¯à®•à¯à®•à®µà¯à®®à¯ {link_kyc}"),
        (0, "à®‰à®™à¯à®•à®³à¯ à®’à®°à¯ à®®à¯à®±à¯ˆ à®•à®Ÿà®µà¯à®šà¯à®šà¯Šà®²à¯ (OTP) {otp}. à®‡à®¤à¯ˆ à®¯à®¾à®°à®¿à®Ÿà®®à¯à®®à¯ à®ªà®•à®¿à®° à®µà¯‡à®£à¯à®Ÿà®¾à®®à¯."),
        (0, "Unga accountil Rs {amount} credit seyyappattullathu. Ref: UPI/{otp}."),
        (0, "Unga {bank} transaction Rs {amount} successful aagivittathu."),
        (0, "Official banking service ku {benign_link} website ai payanpaduthavum."),
    ],
    "kn": [
        (1, "à²Žà²šà³à²šà²°à²¿à²•à³†: à²¨à²¿à²®à³à²® à²µà²¿à²¦à³à²¯à³à²¤à³ à²¬à²¿à²²à³ à²ªà²¾à²µà²¤à²¿à²¸à²¿à²²à³à²². à²‡à²‚à²¦à³ à²°à²¾à²¤à³à²°à²¿ à²¸à²‚à²ªà²°à³à²• à²•à²¡à²¿à²¤à²—à³Šà²³à³à²³à²²à²¿à²¦à³†. à²¸à²‚à²ªà²°à³à²•à²¿à²¸à²¿ {phone}"),
        (1, "à²¨à²¿à²®à³à²® {bank} à²–à²¾à²¤à³† à²¬à³à²²à²¾à²•à³ à²†à²—à²¿à²¦à³†. à²¤à²•à³à²·à²£ KYC à²¨à²µà³€à²•à²°à²¿à²¸à²¿ {link_kyc}"),
        (1, "Nimma {utility} bill due ide. Ee rathri power cut agatte. Contact maadi {phone}"),
        (1, "Nimma bank KYC pending ide. Account block aguvudakke munche verify maadi {link_kyc}"),
        (0, "à²¨à²¿à²®à³à²® à²¬à³à²¯à²¾à²‚à²•à²¿à²‚à²—à³ OTP à²¸à²‚à²–à³à²¯à³† {otp} à²†à²—à²¿à²¦à³†. à²‡à²¦à²¨à³à²¨à³ à²¯à²¾à²°à³Šà²‚à²¦à²¿à²—à³‚ à²¹à²‚à²šà²¿à²•à³Šà²³à³à²³à²¬à³‡à²¡à²¿."),
        (0, "Nimma account ge Rs {amount} credit aagide. Balance check madi."),
        (0, "Nimma {bank} transaction Rs {amount} successful aagide."),
        (0, "Official banking services ge {benign_link} balasi."),
    ],
    "ml": [
        (1, "à´®àµà´¨àµà´¨à´±à´¿à´¯à´¿à´ªàµà´ªàµ: à´¨à´¿à´™àµà´™à´³àµà´Ÿàµ† à´µàµˆà´¦àµà´¯àµà´¤à´¿ à´¬à´¿àµ½ à´•àµà´Ÿà´¿à´¶àµà´¶à´¿à´•à´¯à´¾à´£àµ. à´‡à´¨àµà´¨àµ à´°à´¾à´¤àµà´°à´¿ à´•à´£à´•àµà´·àµ» à´µà´¿à´šàµà´›àµ‡à´¦à´¿à´•àµà´•àµà´‚. à´µà´¿à´³à´¿à´•àµà´•àµà´• {phone}"),
        (1, "à´¨à´¿à´™àµà´™à´³àµà´Ÿàµ† {bank} à´…à´•àµà´•àµ—à´£àµà´Ÿàµ à´¤à´¾àµ½à´•àµà´•à´¾à´²à´¿à´•à´®à´¾à´¯à´¿ à´±à´¦àµà´¦à´¾à´•àµà´•à´¿. KYC à´…à´ªàµâ€Œà´¡àµ‡à´±àµà´±àµ à´šàµ†à´¯àµà´¯àµà´• {link_kyc}"),
        (1, "Ningale {utility} bill adachilla engil innu rathri current cut aavum. Pay {link_bill}"),
        (1, "Bank KYC pending aanu. Account block aakunnathinu munpu verify cheyyuka {link_kyc}"),
        (0, "à´¨à´¿à´™àµà´™à´³àµà´Ÿàµ† à´µàµº à´Ÿàµˆà´‚ à´ªà´¾à´¸àµâ€Œà´µàµ‡à´¡àµ (OTP) {otp} à´†à´£àµ. à´‡à´¤àµ à´†à´°àµ‹à´Ÿàµà´‚ à´ªà´™àµà´•à´¿à´Ÿà´°àµà´¤àµ."),
        (0, "Ningalude accountil Rs {amount} credit aayi. Ref UPI/{otp}."),
        (0, "Ningalude {bank} transaction Rs {amount} successful aayi."),
        (0, "Official banking services inu {benign_link} upayogikkuka."),
    ],
}

RENDER_FIELDS = {
    "bank", "utility", "time", "phone", "tollfree", "amount", "amount2",
    "otp", "otp_last4", "link_kyc", "link_bill", "benign_link",
}


@dataclass(frozen=True)
class Template:
    id: str
    language: str
    label: int
    text: str


def make_template(language, label, text):
    fields = {f for _, f, _, _ in Formatter().parse(text) if f}
    unknown = fields - RENDER_FIELDS
    if unknown:
        raise ValueError(f"Unknown placeholder(s) {sorted(unknown)} in template: {text!r}")
    if label not in (0, 1):
        raise ValueError(f"Label must be 0 or 1, got {label!r}")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return Template(f"{language}-{label}-{digest}", language, int(label), text)


def load_templates(dataset_dir):
    by_lang = {}
    for lang, specs in TEMPLATE_SPECS.items():
        by_lang[lang] = [make_template(lang, lab, txt) for lab, txt in specs]

    extra = Path(dataset_dir) / "extra_templates.jsonl"
    if extra.exists():
        n = 0
        for line in extra.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            by_lang.setdefault(row["language"], []).append(
                make_template(row["language"], row["label"], row["text"])
            )
            n += 1
        log.info("Loaded %d extra templates from %s", n, extra.name)

    all_ids = [t.id for ts in by_lang.values() for t in ts]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate templates detected.")
    return by_lang


# =========================================================
# RENDERING + LABEL-INDEPENDENT AUGMENTATION
# =========================================================


def augment(rng, text, bank):
    """Applied identically to both classes, so it adds variety, not a shortcut."""
    if rng.random() < 0.25:
        text = rng.choice(
            [f"[{bank}] ", f"{bank}: ", f"VM-{bank.upper()[:6]}: ", "ALERT: "]
        ) + text
    if rng.random() < 0.15:
        text += rng.choice([" Thank you.", f" Regards, {bank}", " -Team"])
    if rng.random() < 0.10:
        text = text.replace(" ", "  ", 1)
    if rng.random() < 0.10:
        text = text.lower()
    if rng.random() < 0.10:
        text = text.rstrip(".!")
    return text


def render(rng, tpl):
    bank = rng.choice(BANKS)
    values = dict(
        bank=bank,
        utility=rng.choice(UTILITIES),
        time=rand_time(rng),
        phone=rand_phone(rng),
        tollfree=rand_tollfree(rng),
        amount=rand_amount(rng),
        amount2=rand_amount(rng),
        otp=rand_otp(rng),
        otp_last4=str(rng.randint(1000, 9999)),
        link_kyc=rand_spam_link(rng, "kyc-verify", bank),
        link_bill=rand_spam_link(rng, "quick-pay", bank),
        benign_link=rand_benign_link(rng),
    )
    return augment(rng, tpl.text.format(**values), bank)


# =========================================================
# TEMPLATE SPLITTING (group-wise by template)
# =========================================================


def split_templates(rng, by_lang, warnings):
    train, val, test = {}, {}, {}
    for lang, templates in by_lang.items():
        tr, va, te = [], [], []
        for label in (0, 1):
            pool = sorted((t for t in templates if t.label == label), key=lambda t: t.id)
            rng.shuffle(pool)
            n = len(pool)
            n_eval = max(1, round(n * 0.20))
            while n - 2 * n_eval < 2 and n_eval > 0:
                n_eval -= 1
            if n_eval < 1:
                raise ValueError(
                    f"Language '{lang}' label {label} has only {n} templates; need >= 4."
                )
            if n_eval == 1:
                warnings.append(
                    f"[{lang}] label {label}: only {n} templates -> 1 val + 1 test template. "
                    "Add more templates for meaningful generalisation numbers."
                )
            va += pool[:n_eval]
            te += pool[n_eval:2 * n_eval]
            tr += pool[2 * n_eval:]
        train[lang], val[lang], test[lang] = tr, va, te
    return train, val, test


# =========================================================
# SYNTHETIC GENERATION
# =========================================================


def generate_synthetic(rng, templates, language, split_name, target, shortfalls):
    target_spam = round(target * SYNTH_SPAM_RATIO)
    targets = {1: target_spam, 0: target - target_spam}
    seen, records = set(), []

    for label, want in targets.items():
        pool = [t for t in templates if t.label == label]
        if not pool:
            raise ValueError(f"No class-{label} templates for {language}/{split_name}")

        got, attempts, max_attempts = 0, 0, want * 60
        while got < want and attempts < max_attempts:
            tpl = pool[attempts % len(pool)]  # round-robin keeps templates balanced
            attempts += 1
            text = render(rng, tpl)
            if text in seen:
                continue
            seen.add(text)
            records.append({
                "text": text,
                "label": label,
                "language": language,
                "source": "synthetic",
                "split": split_name,
                "template_id": tpl.id,
            })
            got += 1

        if got < want:
            shortfalls.append({
                "language": language, "split": split_name,
                "label": label, "target": want, "generated": got,
            })

    return pd.DataFrame(records)


# =========================================================
# NORMALISATION (near-duplicate detection)
# =========================================================

_ENT_RE = re.compile(
    r"\b(?:" + "|".join(
        re.escape(e) for e in sorted(BANKS + UTILITIES, key=len, reverse=True)
    ) + r")\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"\S*(?:://|/|@|www\.)\S*")
_NUM_RE = re.compile(r"\d[\d,.]*")
_PUNCT = str.maketrans({c: " " for c in string.punctuation + "à¥¤à¥¥"})


def normalize(text):
    t = str(text).lower()
    t = _URL_RE.sub(" urltok ", t)
    t = _ENT_RE.sub(" enttok ", t)
    t = _NUM_RE.sub(" numtok ", t)
    return " ".join(t.translate(_PUNCT).split())


# =========================================================
# REAL DATA
# =========================================================


def load_uci(dataset_dir):
    zip_path = dataset_dir / "sms.zip"
    txt_path = dataset_dir / UCI_MEMBER

    if not txt_path.exists():
        log.info("Downloading UCI SMS Spam Collection...")
        with urllib.request.urlopen(UCI_URL, timeout=60) as resp:
            zip_path.write_bytes(resp.read())
        with zipfile.ZipFile(zip_path) as zf:
            if UCI_MEMBER not in zf.namelist():
                raise RuntimeError(f"{UCI_MEMBER} not found in downloaded archive.")
            zf.extract(UCI_MEMBER, dataset_dir)

    df = pd.read_csv(txt_path, sep="\t", names=["raw_label", "text"], encoding="latin-1")
    df["label"] = df["raw_label"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)
    df["norm"] = df["text"].map(normalize)

    # Drop norms that carry conflicting labels, then near-duplicates.
    conflicting = df.groupby("norm")["label"].nunique()
    conflicting = set(conflicting[conflicting > 1].index)
    n_conflict = int(df["norm"].isin(conflicting).sum())
    df = df[~df["norm"].isin(conflicting)]
    before = len(df)
    df = df.drop_duplicates(subset=["norm"]).reset_index(drop=True)
    log.info(
        "UCI: %d usable rows (dropped %d near-duplicates, %d label-conflict rows)",
        len(df), before - len(df), n_conflict,
    )

    df["source"] = "uci"
    df["language"] = "en"
    df["template_id"] = "uci"
    return df[["text", "label", "source", "language", "template_id", "norm"]]


def split_uci(df, seed):
    f = UCI_FRACTIONS
    train, temp = train_test_split(
        df, test_size=f["val"] + f["test"], random_state=seed, stratify=df["label"]
    )
    val, test = train_test_split(
        temp, test_size=f["test"] / (f["val"] + f["test"]),
        random_state=seed, stratify=temp["label"],
    )
    out = {}
    for name, part, split in (
        ("train", train, "train"), ("val", val, "validation"), ("test", test, "uci_test")
    ):
        part = part.copy()
        part["split"] = split
        out[name] = part.reset_index(drop=True)
    return out


def load_real_test(path, warnings):
    """Hand-labelled real messages. Columns: text,label,language."""
    path = Path(path)
    if not path.exists():
        warnings.append(
            f"No hand-labelled real test set found at {path.name}. Synthetic and UCI "
            "scores will NOT reflect real multilingual performance. Add a CSV with "
            "columns text,label,language (aim for >= "
            f"{MIN_REAL_PER_LANGUAGE} per language and class)."
        )
        return None

    df = pd.read_csv(path, encoding="utf-8")
    missing = {"text", "label", "language"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    df = df.dropna(subset=["text", "label", "language"]).copy()
    df["label"] = df["label"].astype(int)
    if not set(df["label"]).issubset({0, 1}):
        raise ValueError(f"{path.name}: labels must be 0 or 1.")

    df["source"] = "real_labeled"
    df["split"] = "real_test"
    df["template_id"] = "real"
    df["norm"] = df["text"].map(normalize)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    counts = df.groupby("language").size()
    for lang, n in counts.items():
        if n < MIN_REAL_PER_LANGUAGE:
            warnings.append(f"Real test: only {n} rows for language '{lang}'.")
    return df[["text", "label", "source", "language", "split", "template_id", "norm"]]


# =========================================================
# ASSEMBLY HELPERS
# =========================================================

COLUMNS = ["text", "label", "language", "source", "split", "template_id"]


def concat(frames):
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS + ["norm"])


def finalize(df, seed):
    if df.empty:
        return df
    df = df.copy()
    if "norm" not in df.columns:
        df["norm"] = df["text"].map(normalize)
    df = df.drop_duplicates(subset=["text"])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def drop_overlap(df, refs):
    if df is None or df.empty:
        return df, 0
    ref_text, ref_norm = set(), set()
    for r in refs:
        if r is not None and not r.empty:
            ref_text |= set(r["text"])
            ref_norm |= set(r["norm"])
    mask = df["text"].isin(ref_text) | df["norm"].isin(ref_norm)
    return df[~mask].reset_index(drop=True), int(mask.sum())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# =========================================================
# REPORT
# =========================================================


def write_report(path, frames, meta):
    lines = []
    w = lines.append

    w("MULTILINGUAL SMISHING DATASET REPORT")
    w("=" * 60)
    w(f"Generated : {meta['generated']}")
    w(f"Seed      : {meta['seed']}")
    w(f"Python    : {platform.python_version()}  pandas {pd.__version__}  sklearn {sklearn.__version__}")
    w("")

    w("DATASET SIZE")
    w("-" * 40)
    for name, df in frames.items():
        w(f"{name:<22}: {len(df)}")
    w("")

    w("CLASS DISTRIBUTION (ham=0 / spam=1)")
    w("-" * 40)
    for name, df in frames.items():
        c = df["label"].value_counts()
        w(f"{name:<22}: ham={c.get(0, 0)}  spam={c.get(1, 0)}")
    w("")

    w("SOURCE DISTRIBUTION")
    w("-" * 40)
    for name, df in frames.items():
        w(f"\n{name}")
        w(df["source"].value_counts().to_string() if not df.empty else "(empty)")
    w("")

    w("LANGUAGE x LABEL")
    w("-" * 40)
    for name, df in frames.items():
        w(f"\n{name}")
        w(pd.crosstab(df["language"], df["label"]).to_string() if not df.empty else "(empty)")
    w("")

    w("TEMPLATE DIVERSITY (distinct synthetic templates per language)")
    w("-" * 40)
    for name, df in frames.items():
        syn = df[df["source"] == "synthetic"]
        w(f"\n{name}")
        w(syn.groupby("language")["template_id"].nunique().to_string() if not syn.empty else "(no synthetic rows)")
    w("")

    w("OVERLAP REMOVED (exact or normalised match with an earlier split)")
    w("-" * 40)
    for k, v in meta["overlap_removed"].items():
        w(f"{k:<28}: {v}")
    w("")

    w("SYNTHETIC SHORTFALLS (generated < target)")
    w("-" * 40)
    if meta["shortfalls"]:
        for s in meta["shortfalls"]:
            w(f"{s['language']:<12} {s['split']:<22} label={s['label']} "
              f"target={s['target']} generated={s['generated']}")
    else:
        w("None")
    w("")

    w("WARNINGS")
    w("-" * 40)
    if meta["warnings"]:
        for msg in meta["warnings"]:
            w(f"- {msg}")
    else:
        w("None")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================================================
# MAIN PIPELINE
# =========================================================


def build(args):
    seed = args.seed
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    warnings, shortfalls = [], []
    stage_rng = lambda name: random.Random(f"{seed}:{name}")

    # 1. Templates ---------------------------------------------------------
    log.info("[1/8] Loading and validating templates...")
    by_lang = load_templates(out)

    # 2. Real baseline -----------------------------------------------------
    log.info("[2/8] Loading UCI baseline...")
    try:
        uci = load_uci(out)
        uci_parts = split_uci(uci, seed)
    except Exception as exc:
        if not args.allow_missing_uci:
            raise RuntimeError(
                f"UCI load failed ({exc}). Re-run with --allow-missing-uci to continue without it."
            ) from exc
        warnings.append(f"UCI unavailable ({exc}); building synthetic-only data.")
        empty = pd.DataFrame(columns=COLUMNS + ["norm"])
        uci_parts = {"train": empty, "val": empty, "test": empty}

    # 3. Template split ----------------------------------------------------
    log.info("[3/8] Splitting templates by group...")
    tr_t, va_t, te_t = split_templates(stage_rng("split"), by_lang, warnings)

    # 4. Synthetic generation ---------------------------------------------
    log.info("[4/8] Generating synthetic data...")
    plan = [
        ("train", tr_t, TRAIN_SYNTHETIC_PER_LANGUAGE, "train"),
        ("val", va_t, VAL_SYNTHETIC_PER_LANGUAGE, "validation"),
        ("test", te_t, TEST_SYNTHETIC_PER_LANGUAGE, "unseen_template_test"),
    ]
    synth = {}
    for key, tmpl_map, n, split_name in plan:
        rng = stage_rng(f"gen-{key}")
        synth[key] = concat([
            generate_synthetic(rng, tmpl_map[lang], lang, split_name, n, shortfalls)
            for lang in tmpl_map
        ])

    if shortfalls:
        msg = f"{len(shortfalls)} class/split groups produced fewer samples than targeted."
        if args.strict:
            raise RuntimeError(msg + " (--strict)")
        warnings.append(msg + " Add template variety or lower the targets.")

    # 5. Assemble + clean --------------------------------------------------
    log.info("[5/8] Assembling and removing overlap...")
    train = finalize(concat([uci_parts["train"], synth["train"]]), seed)
    val = finalize(concat([uci_parts["val"], synth["val"]]), seed)
    synth_test = finalize(synth["test"], seed)
    uci_test = finalize(uci_parts["test"], seed)
    real_test = load_real_test(args.real_test or out / "real_test_labeled.csv", warnings)
    real_test = finalize(real_test, seed) if real_test is not None else None

    overlap = {}
    val, overlap["validation vs train"] = drop_overlap(val, [train])
    synth_test, overlap["synthetic test vs train/val"] = drop_overlap(synth_test, [train, val])
    uci_test, overlap["uci test vs train/val"] = drop_overlap(uci_test, [train, val])
    if real_test is not None:
        real_test, overlap["real test vs train/val"] = drop_overlap(real_test, [train, val])

    # 6. Save --------------------------------------------------------------
    log.info("[6/8] Saving datasets...")
    files = {
        "train.csv": train,
        "val.csv": val,
        "unseen_template_test.csv": synth_test,
        "uci_test.csv": uci_test,
    }
    if real_test is not None:
        files["real_test.csv"] = real_test

    for fname, df in files.items():
        keep = [c for c in COLUMNS if c in df.columns]
        df[keep].to_csv(out / fname, index=False, encoding="utf-8")

    # 7. Report ------------------------------------------------------------
    log.info("[7/8] Writing quality report...")
    frames = {fname.removesuffix(".csv"): df for fname, df in files.items()}
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "overlap_removed": overlap,
        "shortfalls": shortfalls,
        "warnings": warnings,
    }
    write_report(out / "dataset_report.txt", frames, meta)

    # 8. Manifest ----------------------------------------------------------
    log.info("[8/8] Writing manifest...")
    manifest = {
        "generated": meta["generated"],
        "seed": seed,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "config": {
            "train_synthetic_per_language": TRAIN_SYNTHETIC_PER_LANGUAGE,
            "val_synthetic_per_language": VAL_SYNTHETIC_PER_LANGUAGE,
            "test_synthetic_per_language": TEST_SYNTHETIC_PER_LANGUAGE,
            "synth_spam_ratio": SYNTH_SPAM_RATIO,
            "uci_fractions": UCI_FRACTIONS,
        },
        "files": {
            fname: {"rows": len(df), "sha256": sha256_file(out / fname)}
            for fname, df in files.items()
        },
        "overlap_removed": overlap,
        "shortfalls": shortfalls,
        "warnings": warnings,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Summary --------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    for fname, df in files.items():
        c = df["label"].value_counts()
        print(f"{fname:<26} rows={len(df):<6} ham={c.get(0, 0):<6} spam={c.get(1, 0)}")
    print(f"\nReport   : {out / 'dataset_report.txt'}")
    print(f"Manifest : {out / 'manifest.json'}")
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for msg in warnings:
            print(f"  - {msg}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", default=str(DEFAULT_DATASET_DIR))
    p.add_argument("--real-test", default=None, help="CSV with columns text,label,language")
    p.add_argument("--allow-missing-uci", action="store_true")
    p.add_argument("--strict", action="store_true", help="fail if any synthetic target is not met")
    # parse_known_args keeps this safe inside notebooks
    args, _ = p.parse_known_args(argv)
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        build(parse_args())
    except Exception as e:
        log.error("Pipeline failed: %s", e)
        sys.exit(1)
