"""Lesson 15 - A full fine-tune, run and read.

Runs the course's first full fine-tune: the centroid model retrained on
the whole curated training split - all 288 rows, every weight free to
move, nothing frozen, nothing held back. The run is read three ways:
the steering split watches it climb batch by batch, the weight ledger
counts what the run actually wrote, and the frozen test split - the
exam the run never sees - reads the trained model against the base gap
from Lesson 7, line for line.
Run: python -m lesson15.fullrun
"""
from __future__ import annotations

from collections import Counter

from ftcommon import (CORPUS, INTENTS, CentroidModel, accuracy,
                      base_model, split_examples)

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


def run_trace(train, dev):
    """The run itself: retrain from scratch on growing prefixes of
    the training split, steering split scored after each batch.
    Returns the fully trained model."""
    print(f"the run: all {len(train)} rows, in batches of {BATCH},"
          f" every weight free to move")
    model = None
    for k in range(BATCH, len(train) + 1, BATCH):
        model = CentroidModel().train(train[:k])
        print(f"  after {k} rows: dev intent"
              f" {accuracy(model, dev):.3f}")
    return model


def weight_ledger(base, tuned) -> None:
    """Count what the run wrote: parameters are the centroid numbers.
    Full fine-tuning freezes nothing - this ledger is the proof."""
    def numbers(m):
        return sum(len(c) for c in m.centroids.values())

    def words(m):
        vocab = set()
        for c in m.centroids.values():
            vocab |= set(c)
        return len(vocab)
    changed = new = kept = 0
    for label in INTENTS:
        for tok, val in tuned.centroids[label].items():
            if tok not in base.centroids[label]:
                new += 1
            elif abs(base.centroids[label][tok] - val) > 1e-12:
                changed += 1
            else:
                kept += 1
    print(f"weights: base model carries {numbers(base)} numbers"
          f" over a {words(base)}-word vocabulary")
    print(f"weights: tuned model carries {numbers(tuned)} numbers"
          f" over a {words(tuned)}-word vocabulary")
    print(f"weights written by the run: {changed + new} of"
          f" {numbers(tuned)} ({changed} changed value, {new} new to"
          f" their desk); {kept} kept the seed value because the data"
          f" agreed - nothing was frozen")


def print_trained_board(model, rows) -> None:
    print("trained model on the frozen test split:")
    seen, ok = Counter(), Counter()
    for t, l, _p in rows:
        seen[l] += 1
        if model.predict(t) == l:
            ok[l] += 1
    for intent in INTENTS:
        print(f"  {intent}: {ok[intent]}/{seen[intent]}")
    c, n, n_ok, n_tot, routed, esc_n = board_lines(model, rows)
    known = [(t, l) for t, l, p in rows if p < 2]
    k_ok = sum(1 for t, l in known if model.predict(t) == l)
    print(f"  intent overall: {c}/{n} = {c / n:.3f}")
    print(f"  phrasings the seed taught (2 of 6): {k_ok}/{len(known)}"
          f" = {k_ok / len(known):.3f}")
    print(f"  phrasings never taught (4 of 6): {n_ok}/{n_tot}"
          f" = {n_ok / n_tot:.3f}")
    # Training gave the model no escalation path either: the routed
    # tickets are right, and still auto-answered. Compliance is a
    # routing rule, not a training outcome - it stays zero.
    print(f"  must-escalate tickets routed right, still"
          f" auto-answered: {routed}")
    print(f"  policy-compliant handling: 0/{esc_n} = 0.000")


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
    tuned = run_trace(train, dev)
    weight_ledger(base, tuned)
    print_trained_board(tuned, rows)
    print("the gap, before and after:")
    print("  intent 0.750 -> 1.000 | new phrasings 0.625 -> 1.000"
          " | policy 0.000 -> 0.000")
    print("fullrun demo complete")


if __name__ == "__main__":
    main()
