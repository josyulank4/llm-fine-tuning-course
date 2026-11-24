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
    print("task 2 - format: phrasings known vs phrasings new")
    known = [(t, l) for t, l, p in rows if p < 2]
    new = [(t, l) for t, l, p in rows if p >= 2]
    k_ok = sum(1 for t, l in known if model.predict(t) == l)
    n_ok = sum(1 for t, l in new if model.predict(t) == l)
    k_acc, n_acc = k_ok / len(known), n_ok / len(new)
    print(f"  phrasings the seed taught (2 of 6): {k_ok}/{len(known)}"
          f" = {k_acc:.3f}")
    print(f"  phrasings never taught (4 of 6): {n_ok}/{len(new)}"
          f" = {n_acc:.3f}")
    return k_acc, n_acc


def score_policy(model, rows) -> float:
    print("task 3 - policy: tickets that must reach a human")
    esc = [(t, l) for t, l, _p in rows if l in ESCALATE]
    routed = sum(1 for t, l in esc if model.predict(t) == l)
    # The base model has no escalation path: every ticket, these
    # included, gets an automated answer. Handling is compliant only
    # when the ticket is escalated, so compliance is zero by design.
    violations = len(esc)
    compliant = len(esc) - violations
    rate = compliant / len(esc) if esc else 0.0
    print(f"  must-escalate tickets in test: {len(esc)}"
          f" (defect_report 12, account_locked 12)")
    print(f"  routed to the right desk, still auto-answered: {routed}")
    print(f"  auto-answered in violation of the desk rule: {violations}")
    print(f"  policy-compliant handling: {compliant}/{len(esc)}"
          f" = {rate:.3f}")
    return rate


def main() -> None:
    rows = test_rows()
    model = base_model()
    print(f"base model on the frozen test split: {len(rows)} tickets")
    intent_acc = score_intent(model, rows)
    _known, new_acc = score_format(model, rows)
    policy_rate = score_policy(model, rows)
    print("the base gap, per task:")
    print(f"  intent {intent_acc:.3f} | new phrasings {new_acc:.3f}"
          f" | policy {policy_rate:.3f}")
    print("basegap demo complete")


if __name__ == "__main__":
    main()
