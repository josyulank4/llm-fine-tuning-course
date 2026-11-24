"""Shared foundation for the LLM Fine-Tuning in Production companion demos.

One deterministic corpus: 480 synthetic support tickets for a fictional
company, Helios Home Appliances, across 8 intents, split into train / dev /
test by a fixed pattern (the test split is frozen from Lesson 1 onward).
The "models" in this repo are small trainable centroid classifiers over
bag-of-words features: the base model is trained only on a small generic
seed (two phrasings per intent), and later lessons fine-tune it on the full
training split — full, low-rank, and quantized variants are real
computations at toy scale, so every number the course quotes is measured
by running this code, never invented.

Stdlib only. No randomness anywhere: the corpus is data, not generation.
"""
from __future__ import annotations

import math
import re

# ---------------------------------------------------------------------------
# Corpus: 8 intents x 6 templates x 10 fillers = 480 examples.
# Split by position inside each intent block of 60: index % 5 in {0,1,2} ->
# train (36 per intent, 288), 3 -> dev (12 per intent, 96), 4 -> test
# (12 per intent, 96). The test split is FROZEN: no lesson trains on it.
# ---------------------------------------------------------------------------

INTENTS: list[str] = [
    "refund_request",
    "invoice_copy",
    "password_reset",
    "account_locked",
    "delivery_status",
    "address_change",
    "defect_report",
    "plan_upgrade",
]

_TEMPLATES: dict[str, list[str]] = {
    "refund_request": [
        "I want a refund for my {item}, it arrived last week and I changed my mind",
        "Please refund the {item} I bought, the charge is still on my card",
        "How do I get my money back for the {item}? It is within the return window",
        "I would like my payment returned for the {item}, order went through twice",
        "Can you reimburse me for the {item}? I never received a working unit",
        "Refund requested on the {item} — please send the returns label",
    ],
    "invoice_copy": [
        "Could you send me a copy of the invoice for my {item}?",
        "I need the bill for the {item} for my expense report",
        "Where can I download the receipt for the {item} purchase?",
        "Please email me the tax invoice for the {item}",
        "I lost the invoice for my {item}, can you reissue it?",
        "Send the billing document for the {item} to my accounts team",
    ],
    "password_reset": [
        "I forgot my password and cannot sign in to the {item} app",
        "Please help me reset the password on my {item} account",
        "The password reset link for my {item} account has expired, send a new one",
        "I need to change my login password for the {item} app account",
        "My password no longer works after the {item} update, how do I reset it?",
        "Requesting a password reset for the {item} app — I am locked out of the sign-in page",
    ],
    "account_locked": [
        "My account is locked after too many sign-in attempts on the {item} app, please unlock it",
        "I cannot access my {item} account, it says the account has been suspended",
        "Why is my account frozen? I cannot log in to the {item} app since yesterday",
        "My login is blocked and the {item} app says account locked, help",
        "Please restore access to my {item} account, it was locked this morning",
        "Account lockout on my {item} profile — I need it released urgently",
    ],
    "delivery_status": [
        "Where is my {item}? The delivery is three days late",
        "Can you track the shipment for my {item} order?",
        "My {item} has not arrived yet, what is the delivery status?",
        "The courier says delivered but my {item} is not here",
        "When will my {item} arrive? It still shows in transit",
        "Delivery update needed for the {item} I ordered last week",
    ],
    "address_change": [
        "I need to change the delivery address for my {item} order",
        "Please update my account address before the {item} ships, I have moved house",
        "Can I redirect the {item} shipment to my office address?",
        "Wrong address on my {item} order — please correct it before dispatch",
        "Update the shipping address on file for my {item} orders",
        "I moved, so the {item} should go to my new address instead",
    ],
    "defect_report": [
        "My {item} stopped working after two weeks, it will not turn on",
        "The {item} I received is defective, the motor makes a grinding noise",
        "Fault report: the {item} leaks water from the base",
        "My {item} arrived damaged, the casing is cracked",
        "The {item} overheats and shuts down, I think it is faulty",
        "Reporting a defect on the {item} — the display is dead",
    ],
    "plan_upgrade": [
        "I want to upgrade my care plan for the {item} to the premium tier",
        "How much to move my {item} cover to the extended plan?",
        "Please upgrade my service plan for the {item}, the basic cover is not enough",
        "Can I switch my {item} to the annual maintenance plan?",
        "I would like the higher plan tier for my {item} account",
        "Upgrade request: add full cover for the {item} starting this month",
    ],
}

_FILLERS: list[str] = [
    "air fryer", "espresso machine", "robot vacuum", "air purifier",
    "stand mixer", "dehumidifier", "pressure washer", "heat pump",
    "dishwasher", "smart oven",
]


def _build_corpus() -> list[tuple[str, str, str]]:
    """(split, text, intent) for every example, in a fixed order."""
    corpus: list[tuple[str, str, str]] = []
    for intent in INTENTS:
        block: list[tuple[str, str]] = []
        for template in _TEMPLATES[intent]:
            for filler in _FILLERS:
                block.append((template.format(item=filler), intent))
        for idx, (text, label) in enumerate(block):
            split = "train" if idx % 5 in (0, 1, 2) else ("dev" if idx % 5 == 3 else "test")
            corpus.append((split, text, label))
    return corpus


CORPUS: list[tuple[str, str, str]] = _build_corpus()

# Planted defects in the RAW training split (Lesson 11, datacurate, removes
# them and prints the receipts): exact duplicates, mislabeled examples, and
# stale-policy examples written against the retired 14-day return window.
RAW_DEFECTS: dict[str, int] = {"duplicates": 12, "mislabeled": 8, "stale_policy": 6}


def split_examples(split: str) -> list[tuple[str, str]]:
    """(text, intent) pairs for one split, corpus order preserved."""
    return [(text, label) for s, text, label in CORPUS if s == split]


# ---------------------------------------------------------------------------
# Tokenizer. The trainable model arrives with the Lesson 1 stub.
# ---------------------------------------------------------------------------

_STOPWORDS = {"i", "my", "the", "a", "an", "it", "is", "to", "for", "me",
              "please", "can", "you", "of", "on", "in", "and", "my"}


def tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", text.lower()) if t not in _STOPWORDS}
