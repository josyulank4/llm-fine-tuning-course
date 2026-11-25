"""Lesson 35 - Probe the tuned model for new failure modes.

Start state: the lab can state the probe suite's terms and read
the base gap the probes were written against - the training
split of record, the frozen exam, the four candidates (the base
and Lesson 27's three eligible runs), the base model's three
scores from Lesson 7, and the five probe families named. The
suite itself is not built: no candidate is probed, no verdict is
read. The finish state builds all three.
Run: python -m lesson35.probecheck
"""
from __future__ import annotations

from ftcommon import (CORPUS, base_model, split_examples,
                      tokenize)

# Desk rule (Lesson 7): these intents are never auto-answered - a
# defect may be a safety fault and a lockout needs identity checks.
ESCALATE = ("defect_report", "account_locked")


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
    print("the base gap the probes were written against:")
    print(f"  intent: {c}/{n} = {c / n:.3f}")
    print(f"  new phrasings: {n_ok}/{n_tot} = {n_ok / n_tot:.3f}")
    print(f"  policy: 0/{esc_n} = 0.000")


def main() -> None:
    train = split_examples("train")
    rows = test_rows()
    base = base_model()
    print("probe the tuned model for new failure modes")
    print(f"training rows: {len(train)} - the training split of"
          f" record (Lesson 11)")
    print("the candidates: the base itself, and Lesson 27's three"
          " eligible runs - R2 full FT, R5 LoRA rank 2, R7 QLoRA"
          " rank 2")
    print_base_board(base, rows)
    raise NotImplementedError(
        "the probe suite is not built in the l35-start scaffold")


if __name__ == "__main__":
    main()
