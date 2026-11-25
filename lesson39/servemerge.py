"""Lesson 39 - Merge and serve: the deployment drill.

Start state: the lab can state the drill's terms - the
training split of record, the checkpoint (Lesson 27's R7,
QLoRA rank 2 on the 4-bit base, held by Lesson 31's gate),
the base model's three scores from Lesson 7, the three acts
named, and the canary bar fixed in advance. The drill itself
is not built: nothing is merged, nothing is served, no
rollback is rehearsed. The finish state builds all three
acts.
Run: python -m lesson39.servemerge
"""
from __future__ import annotations

from ftcommon import (CORPUS, base_model, split_examples,
                      tokenize)

# Desk rule (Lesson 7): these intents are never auto-answered - a
# defect may be a safety fault and a lockout needs identity checks.
ESCALATE = ("defect_report", "account_locked")

# The canary bar, fixed before the drill runs (Lesson 31's
# capability bar, applied to the steering split): a candidate
# that cannot clear it never takes production traffic.
CANARY_BAR = 0.990


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
    print("merge and serve: the deployment drill")
    print(f"training rows: {len(train)} - the training split of"
          f" record (Lesson 11)")
    print("the checkpoint: Lesson 27's R7 - QLoRA rank 2 on the"
          " 4-bit base, Lesson 31's gate verdict HOLD")
    print("the drill, three acts: merge the adapter into its"
          " base; serve the merge behind a routing table;")
    print("  break the deploy and roll back - canary first at"
          " every cutover")
    print(f"the canary: the 96 steering rows, read before"
          f" cutover; the bar is {CANARY_BAR}, fixed now")


def print_base_board(base, rows) -> None:
    c, n, n_ok, n_tot, _r, esc_n = board_lines(base, rows)
    print("the base gap the drill starts from - v1, the version"
          " production runs today:")
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
        "the deployment drill is not built in the l39-start"
        " scaffold")


if __name__ == "__main__":
    main()
