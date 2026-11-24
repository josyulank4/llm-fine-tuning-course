"""Lesson 11 - Curate the training set, keep the receipts.

The desk's raw training export carries 314 rows: the 288 rows of the
training split plus 26 defects that arrived with the export - exact
duplicates, copies refiled under the wrong desk (contradictions), and
rows written against the retired 14-day return window (stale policy).
This demo cleans the export end to end, prints every removal with the
defect that caused it, and reconciles the result against the frozen
training split row for row.
Run: python -m lesson11.datacurate
"""
from __future__ import annotations

from ftcommon import (INTENTS, CentroidModel, accuracy, split_examples)

# The retired return window. Current policy never quotes a number in
# the corpus templates ("within the return window"); rows that name
# the retired 14 days were written against the old rule.
RETIRED_MARKERS = ("14-day", "14 days")

# Rows written against the retired window, as they arrived in the raw
# export (old tickets, old policy, refund desk labels).
STALE_ROWS: list[tuple[str, str]] = [
    ("I want a refund for my air fryer, it is day 12 so still inside "
     "the 14-day window", "refund_request"),
    ("Please refund the espresso machine, the 14-day return window "
     "closes tomorrow", "refund_request"),
    ("How do I get my money back for the robot vacuum? The 14 days "
     "are nearly up", "refund_request"),
    ("I would like my payment returned for the air purifier under the "
     "14-day policy", "refund_request"),
    ("Can you reimburse me for the stand mixer? I am inside 14 days "
     "of delivery", "refund_request"),
    ("Refund requested on the dehumidifier - the 14-day window was "
     "promised at purchase", "refund_request"),
]


def raw_training_set() -> list[tuple[str, str]]:
    """The raw export as delivered: the clean split plus the defects
    that arrived with it, appended in export batches."""
    clean = split_examples("train")
    rows = list(clean)
    for i in range(0, len(clean), 24):            # 12 exact duplicates
        rows.append(clean[i])
    for i in range(12, 200, 24):                  # 8 refiled copies
        text, _label = clean[i]
        wrong = INTENTS[(INTENTS.index(clean[i][1]) + 1) % len(INTENTS)]
        rows.append((text, wrong))
    rows.extend(STALE_ROWS)                       # 6 stale-policy rows
    return rows


def dedupe(rows):
    """Step 1: exact duplicates - same text, same label, filed twice.
    Keep the first filing; print every removal."""
    seen = set()
    kept, removed = [], []
    for text, label in rows:
        if (text, label) in seen:
            removed.append((text, label))
        else:
            seen.add((text, label))
            kept.append((text, label))
    print("step 1 - exact duplicates (same text, same label):")
    for text, label in removed:
        print(f"  duplicate removed: [{label}] {text[:64]}")
    print(f"  step 1 result: {len(removed)} removed, "
          f"{len(kept)} rows remain")
    return kept


def drop_contradictions(rows):
    """Step 2: contradictions - the same text filed under two desks.
    The corpus of record decides which filing is true."""
    truth = dict(split_examples("train"))
    by_text: dict[str, set] = {}
    for text, label in rows:
        by_text.setdefault(text, set()).add(label)
    kept, removed = [], []
    for text, label in rows:
        if len(by_text[text]) > 1 and label != truth.get(text):
            removed.append((text, label))
        else:
            kept.append((text, label))
    print("step 2 - contradictions (one text, two desks):")
    for text, label in removed:
        print(f"  contradiction removed: filed as {label}, "
              f"record says {truth[text]}")
    print(f"  step 2 result: {len(removed)} removed, "
          f"{len(kept)} rows remain")
    return kept


def drop_stale(rows):
    """Step 3: stale policy - rows that teach the retired 14-day
    return window. Current policy is the only policy that ships."""
    kept, removed = [], []
    for text, label in rows:
        if any(m in text for m in RETIRED_MARKERS):
            removed.append((text, label))
        else:
            kept.append((text, label))
    print("step 3 - stale policy (retired 14-day return window):")
    for text, label in removed:
        print(f"  stale removed: [{label}] {text[:64]}")
    print(f"  step 3 result: {len(removed)} removed, "
          f"{len(kept)} rows remain")
    return kept


def main() -> None:
    raw = raw_training_set()
    print(f"raw training rows: {len(raw)}")
    print(f"the training split of record: "
          f"{len(split_examples('train'))} rows")
    rows = dedupe(raw)
    rows = drop_contradictions(rows)
    rows = drop_stale(rows)
    clean = split_examples("train")
    same = sorted(rows) == sorted(clean)
    print(f"reconciliation: curated == training split of record: "
          f"{same} ({len(rows)} of {len(clean)} rows)")
    test = split_examples("test")
    raw_acc = accuracy(CentroidModel().train(raw), test)
    clean_acc = accuracy(CentroidModel().train(rows), test)
    print(f"frozen test intent, model trained on raw: {raw_acc:.3f}")
    print(f"frozen test intent, model trained on curated: "
          f"{clean_acc:.3f}")
    print("datacurate demo complete")


if __name__ == "__main__":
    main()
