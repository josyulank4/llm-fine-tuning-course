"""Lesson 7 - The base gap, measured per task.

Scores the shipped base model (seed-trained, Lesson 1) on the FROZEN test
split three ways: intent (routed to the right desk), format (known
phrasings vs phrasings the seed never taught), and policy (the tickets the
desk rules say must reach a human, never an automated answer).
Run: python -m lesson7.basegap
"""
from __future__ import annotations

from collections import Counter

from ftcommon import CORPUS, INTENTS, base_model

# Desk rule: these intents are never auto-answered. A defect may be a
# safety fault and a lockout needs identity verification, so both go to a
# human being, every time, whatever the model thinks they are.
ESCALATE = ("defect_report", "account_locked")


def test_rows() -> list[tuple[str, str, int]]:
    """Frozen test rows as (text, intent, phrasing) - phrasing is the
    template index 0-5, recovered from the row's place in its intent
    block (6 templates x 10 fillers, corpus order is fixed)."""
    rows = []
    for pos, (split, text, label) in enumerate(CORPUS):
        if split == "test":
            rows.append((text, label, (pos % 60) // 10))
    return rows


def score_intent(model, rows) -> float:
    print("task 1 - intent: routed to the right desk")
    per_intent = Counter()
    seen = Counter()
    confusions = Counter()
    for text, label, _phr in rows:
        pred = model.predict(text)
        seen[label] += 1
        if pred == label:
            per_intent[label] += 1
        else:
            confusions[(label, pred)] += 1
    for intent in INTENTS:
        print(f"  {intent}: {per_intent[intent]}/{seen[intent]}")
    correct = sum(per_intent.values())
    acc = correct / len(rows)
    print(f"  intent overall: {correct}/{len(rows)} = {acc:.3f}")
    worst = sorted(confusions.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    for (true, pred), n in worst:
        print(f"  confusion: {true} read as {pred} x{n}")
    return acc


def score_format(model, rows) -> tuple[float, float]:
    raise NotImplementedError(
        "format scoring is not built yet - intent is all this scaffold "
        "can measure")


def main() -> None:
    rows = test_rows()
    model = base_model()
    print(f"base model on the frozen test split: {len(rows)} tickets")
    score_intent(model, rows)
    score_format(model, rows)


if __name__ == "__main__":
    main()
