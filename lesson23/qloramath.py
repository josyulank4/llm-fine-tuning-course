"""Lesson 23 - The QLoRA ledger: memory saved, accuracy kept.

Start state: the lab can state the ledger and read the base gap
the three methods have to close - the training split of record,
the steering split, the frozen exam, the weight matrix, and the
base model's three scores from Lesson 7. The ledger itself is
not built: no quantization, no adapter training, no table. The
finish state builds all three.
Run: python -m lesson23.qloramath
"""
from __future__ import annotations

from ftcommon import (CORPUS, INTENTS, base_model, split_examples,
                      tokenize)

# Desk rule (Lesson 7): these intents are never auto-answered - a
# defect may be a safety fault and a lockout needs identity checks.
ESCALATE = ("defect_report", "account_locked")

VOCAB = sorted({t for _s, text, _l in CORPUS for t in tokenize(text)})


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


def print_base_board(base, rows) -> None:
    c, n, n_ok, n_tot, _r, esc_n = board_lines(base, rows)
    print("the base gap the methods have to close:")
    print(f"  intent: {c}/{n} = {c / n:.3f}")
    print(f"  new phrasings: {n_ok}/{n_tot} = {n_ok / n_tot:.3f}")
    print(f"  policy: 0/{esc_n} = 0.000")


def main() -> None:
    train = split_examples("train")
    rows = test_rows()
    base = base_model()
    print("the QLoRA ledger: memory saved, accuracy kept")
    print(f"training rows: {len(train)} - the training split of"
          f" record (Lesson 11)")
    print(f"steering split: 96 dev rows; exam: {len(rows)} frozen"
          f" test rows")
    print(f"the weight matrix: {len(INTENTS)} desks x"
          f" {len(VOCAB)} words = {len(INTENTS) * len(VOCAB)}"
          f" numbers")
    print("Lesson 19's smallest sufficient adapter - rank 2, 426"
          " trainable numbers - is the adapter both adapter rows"
          " train")
    print_base_board(base, rows)
    raise NotImplementedError(
        "the QLoRA ledger is not built in the l23-start scaffold")


if __name__ == "__main__":
    main()
