"""End-to-end validation for the Course 12 code repository (foundation scaffold).
Run from the repo root: python validate.py
"""
from __future__ import annotations
import sys
FAILURES = []
COUNT = [0]
def check(name, condition, detail=""):
    COUNT[0] += 1
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition: FAILURES.append(name)
def main():
    from ftcommon import CORPUS, INTENTS, RAW_DEFECTS, split_examples, tokenize
    check("corpus holds exactly 480 examples", len(CORPUS) == 480)
    check("splits are 288 train / 96 dev / 96 test",
          [len(split_examples(s)) for s in ("train", "dev", "test")] == [288, 96, 96])
    check("eight intents", len(INTENTS) == 8)
    check("corpus texts are unique", len({t for _, t, _ in CORPUS}) == 480)
    check("planted raw defects total 26", sum(RAW_DEFECTS.values()) == 26)
    check("tokenize drops stopwords and case",
          tokenize("The Refund Invoice") == {"refund", "invoice"})
    print()
    print(f"CHECKS: {COUNT[0]}  FAILURES: {len(FAILURES)}")
    if FAILURES:
        print("OVERALL: FAIL")
        return 1
    print("OVERALL: PASS")
    return 0
if __name__ == "__main__":
    sys.exit(main())
