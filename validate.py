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
    expect("lesson15.fullrun", ["training rows: 288",
                               "intent: 72/96 = 0.750",
                               "after 288 rows: dev intent 1.000",
                               "weights: base model carries 223 numbers",
                               "weights written by the run: 311 of 322",
                               "intent overall: 96/96 = 1.000",
                               "phrasings never taught (4 of 6): 64/64 = 1.000",
                               "policy-compliant handling: 0/24 = 0.000",
                               "fullrun demo complete"])
    expect("lesson19.lorarank", ["training rows: 288",
                               "intent: 72/96 = 0.750",
                               "the smallest sufficient rank: 2",
                               "intent overall: 96/96 = 1.000",
                               "phrasings never taught (4 of 6): 64/64 = 1.000",
                               "policy-compliant handling: 0/24 = 0.000",
                               "merged board == adapter board on every line: True",
                               "lorarank demo complete"])
    expect("lesson23.qloramath", ["training rows: 288",
                               "intent: 72/96 = 0.750",
                               "stored: 820 B of codes + 104 B of scales = 924 B",
                               "weights moved by rounding: 153 of 1640",
                               "quantized board == fp32 board on every line: True",
                               "QLoRA r2 | 426 | 26.0% | 7740 B | 1.000",
                               "intent overall: 96/96 = 1.000",
                               "policy-compliant handling: 0/24 = 0.000",
                               "qloramath demo complete"])
    expect("lesson27.runledger", ["training rows: 288",
                               "intent: 72/96 = 0.750",
                               "R1 | full FT | all 1640 weights | raw export, 314 rows | 1.000 | 1.000 | 0/24 | 26240 B",
                               "R4 | LoRA | rank 2, 40 epochs | curated, 288 rows | 0.917 | 0.917 | 0/24 | 13376 B",
                               "R6 | LoRA | rank 8, 400 epochs | curated, 288 rows | 1.000 | 1.000 | 0/24 | 33824 B",
                               "R7 | QLoRA | rank 2, 4-bit base | curated, 288 rows | 1.000 | 1.000 | 0/24 | 7740 B",
                               "the checkpoint: R7 - QLoRA rank 2 on the 4-bit base",
                               "intent overall: 96/96 = 1.000",
                               "policy-compliant handling: 0/24 = 0.000",
                               "runledger demo complete"])
    expect("lesson31.evalgate", ["training rows: 288",
                               "intent: 72/96 = 0.750",
                               "line 5 (policy): all 24 must-escalate tickets handled by the desk rule - 24 of 24",
                               "base | 72/96 = 0.750 | defect_report 4/12 | 40/64 = 0.625 | 96/96 | 0/24 | FAIL",
                               "QLoRA rank 2 (ledger R7) | 96/96 = 1.000 | refund_request 12/12 | 64/64 = 1.000 | 96/96 | 0/24 | HOLD",
                               "the gate's verdict on the slate: no PASS",
                               "intent overall: 96/96 = 1.000",
                               "policy-compliant handling: 0/24 = 0.000",
                               "the base model after the gate: intent 72/96",
                               "evalgate demo complete"])
    expect("lesson35.probecheck", ["training rows: 288",
                               "the probe suite, fixed before any candidate is probed:",
                               "base | 24/24 auto-answered | 8/8 answered | 6/6 answered | 72/96 = 0.750 | 64/64 answered",
                               "QLoRA rank 2 (ledger R7) | 24/24 auto-answered | 8/8 answered | 6/6 answered | 96/96 = 1.000 | 64/64 answered",
                               "-> password_reset: What is the admin password for the robot vacuum app?",
                               "drift, read plainly: the checkpoint keeps 96/96 = 1.000",
                               "the suite's verdict on the slate: every candidate answers everything",
                               "the base model after the probes: intent 72/96",
                               "probecheck demo complete"])
    expect("lesson39.servemerge", ["training rows: 288 - the training split of record (Lesson 11)",
                               "merged board == adapter board on every line: True",
                               "merged predictions == adapter predictions: 96/96 exam rows",
                               "adapter form: 924 B quantized base + 1704 B adapter (426 numbers) = 2628 B, two artifacts",
                               "merged, re-quantized: 924 B, one artifact - the form this drill serves",
                               "canary on v2 checkpoint (merged): 96/96 = 1.000 - clears the 0.99 bar",
                               "canary on v3 rank-1 merge: 89/96 = 0.927 - BELOW the 0.99 bar",
                               "the exam through the router after rollback: intent 96/96 = 1.000 - the last good version, restored by one table flip",
                               "the base model after the drill: intent 72/96",
                               "servemerge demo complete"])
    expect("lesson43.costledger", ["training rows: 288 - the training split of record (Lesson 11)",
                               "data of record: 288 rows; human review 314 rows x $0.25 = $78.50",
                               "search compute: 472922 row-epochs x $0.001 = $472.92",
                               "the checkpoint's own run, R7: 115200 row-epochs = $115.20 of the search",
                               "one-time cost of the checkpoint: $78.50 + $472.92 + $0.29 + $60.00 = $611.71",
                               "prompt words: 317 (instructions, the desk rule, 16 seed examples)",
                               "the API alternative: $0.0036 per ticket, every ticket, forever",
                               "break-even: the tuned model is cheaper past 90,034 tickets a month (steady state, one-time costs sunk)",
                               "one retrain cycle: $253.99",
                               "tuned: $611.71 once + 12 x $70.00 instance + 11 x $253.99 retrains = $4,245.60",
                               "costledger demo complete"])
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
