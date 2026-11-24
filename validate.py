"""End-to-end validation for the Course 12 code repository.
Runs every lesson demo as a subprocess (exactly how a student runs them)
and checks expected output markers. Run from the repo root: python validate.py
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent
FAILURES = []
COUNT = [0]
def check(name, condition, detail=""):
    COUNT[0] += 1
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition: FAILURES.append(name)
def expect(module, markers):
    proc = subprocess.run([sys.executable, "-m", module], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    check(f"{module}: runs end-to-end (exit 0)", proc.returncode == 0, proc.stderr.strip()[-300:])
    for m in markers: check(f"{module}: output contains {m!r}", m in proc.stdout)
def main():
    print("== module demos (subprocess) ==")
    expect("lesson1.basegap_preview", ["examples: 480",
                                       "train: 288",
                                       "dev: 96",
                                       "test: 96",
                                       "intents: 8",
                                       "raw training defects planted: 26",
                                       "base model test accuracy: 0.750",
                                       "fully trained test accuracy: 1.000",
                                       "the base gap: 0.250",
                                       "lesson1 demo complete"])
    expect("lesson7.basegap", ["base model on the frozen test split: 96 tickets",
                               "intent overall: 72/96 = 0.750",
                               "defect_report: 4/12",
                               "confusion: defect_report read as invoice_copy x6",
                               "phrasings the seed taught (2 of 6): 32/32 = 1.000",
                               "phrasings never taught (4 of 6): 40/64 = 0.625",
                               "policy-compliant handling: 0/24 = 0.000",
                               "intent 0.750 | new phrasings 0.625 | policy 0.000",
                               "basegap demo complete"])
    expect("lesson11.datacurate", ["raw training rows: 314",
                               "step 1 result: 12 removed, 302 rows remain",
                               "step 2 result: 8 removed, 294 rows remain",
                               "step 3 result: 6 removed, 288 rows remain",
                               "reconciliation: curated == training split of record: True (288 of 288 rows)",
                               "frozen test intent, model trained on raw: 1.000",
                               "frozen test intent, model trained on curated: 1.000",
                               "datacurate demo complete"])
    print()
    print("== direct assertions ==")
    from ftcommon import (CORPUS, INTENTS, RAW_DEFECTS, CentroidModel,
                          accuracy, base_model, split_examples, tokenize)
    check("corpus holds exactly 480 examples", len(CORPUS) == 480)
    check("splits are 288 train / 96 dev / 96 test",
          [len(split_examples(s)) for s in ("train", "dev", "test")] == [288, 96, 96])
    check("eight intents", len(INTENTS) == 8)
    check("test split texts are unique (frozen set, no duplicates)",
          len({t for t, _ in split_examples("test")}) == 96)
    check("planted raw defects total 26",
          sum(RAW_DEFECTS.values()) == 26)
    check("tokenize drops stopwords and case",
          tokenize("The Refund Invoice") == {"refund", "invoice"})
    _test = split_examples("test")
    check("base model leaves a real gap on the frozen test split",
          accuracy(base_model(), _test) < accuracy(
              CentroidModel().train(split_examples("train")), _test))
    print()
    print(f"CHECKS: {COUNT[0]}  FAILURES: {len(FAILURES)}")
    if FAILURES:
        print("OVERALL: FAIL")
        for f in FAILURES: print("  FAILED:", f)
        return 1
    print("OVERALL: PASS")
    return 0
if __name__ == "__main__":
    sys.exit(main())
