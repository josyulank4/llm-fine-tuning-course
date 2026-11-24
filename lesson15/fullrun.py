"""Lesson 15 - A full fine-tune, run and read.

Start state: the lab can state the run and read the base gap it has
to close - the training split of record, the steering split, the
frozen exam, and the base model's three scores from Lesson 7. The
run itself is not built: no training, no weight ledger, no trained
board. The finish state builds all three.
Run: python -m lesson15.fullrun
"""
from __future__ import annotations

from ftcommon import CORPUS, base_model, split_examples

# Desk rule (Lesson 7): these intents are never auto-answered - a
# defect may be a safety fault and a lockout needs identity checks.
ESCALATE = ("defect_report", "account_locked")

BATCH = 48


def test_rows() -> list[tuple[str, str, int]]:
    """Frozen test rows as (text, intent, phrasing) - phrasing is the
    template index 0-5, recovered from the row's place in its intent
    block (6 templates x 10 fillers, corpus order is fixed)."""
    rows = []
    for pos, (split, text, label) in enumerate(CORPUS):
        if split == "test":
            rows.append((text, label, (pos % 60) // 10))
    return rows


def board_lines(model, rows):
    """The three Lesson 7 scores for one model on the frozen test
    split: (intent correct, intent total, intent acc), the same for
    phrasings the seed never taught, and the policy count."""
    correct = sum(1 for t, l, _p in rows if model.predict(t) == l)
    new = [(t, l) for t, l, p in rows if p >= 2]
    n_ok = sum(1 for t, l in new if model.predict(t) == l)
    esc = [(t, l) for t, l, _p in rows if l in ESCALATE]
    routed = sum(1 for t, l in esc if model.predict(t) == l)
    return (correct, len(rows), n_ok, len(new), routed, len(esc))


def print_base_board(model, rows) -> None:
    c, n, n_ok, n_tot, _r, esc_n = board_lines(model, rows)
    print("the base gap the run has to close:")
    print(f"  intent: {c}/{n} = {c / n:.3f}")
    print(f"  new phrasings: {n_ok}/{n_tot} = {n_ok / n_tot:.3f}")
    # The base model has no escalation path: every ticket, the
    # must-escalate ones included, gets an automated answer, so
    # policy-compliant handling is zero by construction (Lesson 7).
    print(f"  policy: 0/{esc_n} = 0.000")


def main() -> None:
    train = split_examples("train")
    dev = split_examples("dev")
    rows = test_rows()
    base = base_model()
    print("full fine-tune: every weight moves")
    print(f"training rows: {len(train)} - the training split of"
          f" record (Lesson 11 reconciled the curated set to it,"
          f" row for row)")
    print(f"steering split: {len(dev)} dev rows; exam:"
          f" {len(rows)} frozen test rows, untouched by the run")
    print_base_board(base, rows)
    raise NotImplementedError(
        "full fine-tune not built in the l15-start scaffold")


if __name__ == "__main__":
    main()
