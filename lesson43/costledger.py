"""Lesson 43 - The cost ledger: train, serve, retrain.

Start state: the lab can state the ledger's terms - the
training split of record, the checkpoint (Lesson 27's R7,
QLoRA rank 2 on the 4-bit base, held by Lesson 31's gate),
the price card fixed before any line is priced, the three
parts named, and the base model's three scores from
Lesson 7, the gap the money was spent against. The ledger
itself is not built: no line is priced, no prompt is
counted, no break-even is computed. The finish state
builds all three parts.
Run: python -m lesson43.costledger
"""
from __future__ import annotations

from ftcommon import (CORPUS, base_model, split_examples,
                      tokenize)

# Desk rule (Lesson 7): these intents are never auto-answered - a
# defect may be a safety fault and a lockout needs identity checks.
ESCALATE = ("defect_report", "account_locked")

# The price card, fixed before any line is priced. Rates are
# the card's stated assumptions; every quantity they multiply
# is measured by the finish state, running the code.
REVIEW_PER_ROW = 0.25        # human review, per row handled
COMPUTE_PER_ROW_EPOCH = 0.001  # training compute, per row-epoch
SCORE_PER_ROW = 0.001        # evaluation scoring, per row scored
REVIEW_HOURS = 2             # engineer-hours per release review
ENGINEER_PER_HOUR = 30.00
API_IN_PER_MILLION = 8.00    # the API alternative, input words
API_OUT_PER_MILLION = 24.00  # output words price higher, as usual
OUTPUT_WORDS = 40            # the card's allowance per API answer
INSTANCE_PER_MONTH = 70.00   # serving the tuned model: one instance


def money(x: float) -> str:
    return f"${x:,.2f}"


def rate3(x: float) -> str:
    """The card's sub-cent rates, printed at their true precision."""
    return f"${x:.3f}"


def test_rows() -> list[tuple[str, str, int]]:
    """Frozen test rows as (text, intent, phrasing) - phrasing is
    the template index 0-5, from the row's place in its block."""
    rows = []
    for pos, (split, text, label) in enumerate(CORPUS):
        if split == "test":
            rows.append((text, label, (pos % 60) // 10))
    return rows


def board_lines(model, rows):
    """The three Lesson 7 scores for one model on the frozen exam."""
    correct = sum(1 for t, l, _p in rows if model.predict(t) == l)
    new = [(t, l) for t, l, p in rows if p >= 2]
    n_ok = sum(1 for t, l in new if model.predict(t) == l)
    esc = [(t, l) for t, l, _p in rows if l in ESCALATE]
    routed = sum(1 for t, l in esc if model.predict(t) == l)
    return (correct, len(rows), n_ok, len(new), routed, len(esc))


def print_terms(train) -> None:
    print("the cost ledger: train, serve, retrain")
    print(f"training rows: {len(train)} - the training split of"
          f" record (Lesson 11)")
    print("the checkpoint: Lesson 27's R7 - QLoRA rank 2 on the"
          " 4-bit base, Lesson 31's gate verdict HOLD")
    print("the price card, fixed before any line is priced:")
    print(f"  human review of training data:"
          f" {money(REVIEW_PER_ROW)} per row handled")
    print(f"  training compute: {rate3(COMPUTE_PER_ROW_EPOCH)}"
          f" per row-epoch; evaluation scoring:"
          f" {rate3(SCORE_PER_ROW)} per row scored")
    print(f"  release review: {REVIEW_HOURS} engineer-hours at"
          f" {money(ENGINEER_PER_HOUR)} ="
          f" {money(REVIEW_HOURS * ENGINEER_PER_HOUR)} per release")
    print(f"  the API alternative: {money(API_IN_PER_MILLION)}"
          f" per million input words,"
          f" {money(API_OUT_PER_MILLION)} per million output")
    print(f"    words; the prompted call carries the desk prompt"
          f" on every ticket; its answer is priced at"
          f" {OUTPUT_WORDS} words")
    print(f"  serving the tuned model: one instance at"
          f" {money(INSTANCE_PER_MONTH)} per month, no"
          " per-ticket charge")
    print("the ledger's three parts: what training cost, what"
          " serving costs, what a retrain cycle costs")


def print_base_board(base, rows) -> None:
    c, n, n_ok, n_tot, _r, esc_n = board_lines(base, rows)
    print("the gap the money was spent against - the base model"
          " on the frozen exam:")
    print(f"  intent: {c}/{n} = {c / n:.3f}")
    print(f"  new phrasings: {n_ok}/{n_tot} = {n_ok / n_tot:.3f}")
    print(f"  policy: 0/{esc_n} = 0.000")


def main() -> None:
    train = split_examples("train")
    rows = test_rows()
    base = base_model()
    print_terms(train)
    print_base_board(base, rows)
    raise NotImplementedError(
        "the cost ledger is not built in the l43-start"
        " scaffold")


if __name__ == "__main__":
    main()
