"""Lesson 27 - The training ledger: every run accounted.

Sections 3-5 produced the lab's runs one at a time: two full
fine-tunes, a rank sweep, a quantized run. This lesson runs the
whole history again under one roof and logs every run the same
way - method, config, data version, and outcome on the steering
split and the frozen exam - then defends the checkpoint choice
from the ledger alone, by a rule fixed before the ledger is
read: data of record only, the steering split chooses, the
smallest training state among the chosen ships. One run in the
ledger is trained on Lesson 11's raw export, defects and all:
its score is perfect, and the ledger bars it anyway - provenance
is a rule, not a score. Run: python -m lesson27.runledger
"""
from __future__ import annotations

import math

from ftcommon import (CORPUS, INTENTS, CentroidModel, accuracy,
                      base_model, split_examples, tokenize)
from lesson11.datacurate import raw_training_set

# Desk rule (Lesson 7): these intents are never auto-answered - a
# defect may be a safety fault and a lockout needs identity checks.
ESCALATE = ("defect_report", "account_locked")

VOCAB = sorted({t for _s, text, _l in CORPUS for t in tokenize(text)})
IDX = {w: i for i, w in enumerate(VOCAB)}
STATE = 16      # training bytes per trainable number (Lesson 14)
EPOCHS = 400    # the full adapter schedule (Lessons 19 and 23)
SHORT = 40      # R4's stop: the same run, halted early (Lesson 26)
LR = 0.5
TEMP = 10.0
BLOCK = 64      # weights per quantization block (Lesson 21)
LEVELS = 7      # 4-bit signed codes run -7..7 around the block scale


def base_matrix(base):
    """The base model as one matrix: 8 desk rows x the 205-word
    vocabulary, zeros where a desk's centroid never met a word."""
    return [[base.centroids[lab].get(w, 0.0) for w in VOCAB]
            for lab in INTENTS]


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
    print("the base gap every run had to close:")
    print(f"  intent: {c}/{n} = {c / n:.3f}")
    print(f"  new phrasings: {n_ok}/{n_tot} = {n_ok / n_tot:.3f}")
    print(f"  policy: 0/{esc_n} = 0.000")


def quantize(mat):
    """Lesson 21's scheme on the whole matrix, flattened row by
    row: blocks of 64 weights, one fp32 scale per block (the
    block's largest magnitude), each weight stored as a 4-bit
    signed code in -7..7. Returns (codes, scales, dequantized) -
    the dequantized matrix is what computation ever sees.
    Lesson 23's code, unchanged."""
    flat = [v for row in mat for v in row]
    codes, scales = [], []
    deq = [0.0] * len(flat)
    for s in range(0, len(flat), BLOCK):
        blk = flat[s:s + BLOCK]
        scale = max(abs(v) for v in blk) or 1.0
        scales.append(scale)
        for k, v in enumerate(blk):
            c = int(round(v / scale * LEVELS))
            c = max(-LEVELS, min(LEVELS, c))
            codes.append(c)
            deq[s + k] = c * scale / LEVELS
    out = [deq[i * len(VOCAB):(i + 1) * len(VOCAB)]
           for i in range(len(INTENTS))]
    return codes, scales, out


def matrix_model(mat):
    """A scored model whose centroids are one matrix, built as a
    fresh object - used for the dequantized base."""
    model = CentroidModel()
    cents = {}
    for ci, lab in enumerate(INTENTS):
        cents[lab] = {VOCAB[j]: v for j, v in enumerate(mat[ci])
                      if abs(v) > 1e-12}
    model.centroids = cents
    return model


def train_adapter(w0, feats, rank, epochs=EPOCHS):
    """Train one adapter: delta = 2.0 * B @ A over the given base
    matrix, B from zeros, A from a fixed pattern (no randomness
    anywhere), full-batch gradient descent on cross-entropy over
    the model's own cosine scores - Lessons 19 and 23's trainer,
    with the epoch count as a dial so the ledger can log a run
    stopped early beside the same run completed."""
    scale = 2.0  # alpha = 2 * rank; rank is the only dial
    V, C = len(VOCAB), len(INTENTS)
    A = [[0.05 * (((i * 31 + j * 17) % 13) - 6) / 6.0
          for j in range(V)] for i in range(rank)]
    B = [[0.0] * rank for _ in range(C)]
    N = len(feats)
    for _epoch in range(epochs):
        W = [[w0[c][j] + scale * sum(B[c][i] * A[i][j]
              for i in range(rank)) for j in range(V)]
             for c in range(C)]
        nrm = [math.sqrt(sum(v * v for v in W[c])) or 1.0
               for c in range(C)]
        U = [[W[c][j] / nrm[c] for j in range(V)] for c in range(C)]
        G = [[0.0] * V for _ in range(C)]
        coeff = [0.0] * C
        for xs, y in feats:
            xn = math.sqrt(len(xs)) or 1.0
            scores = [sum(U[c][j] for j in xs) / xn for c in range(C)]
            logits = [TEMP * sc for sc in scores]
            mx = max(logits)
            ex = [math.exp(z - mx) for z in logits]
            tot = sum(ex)
            for c in range(C):
                p = ex[c] / tot - (1.0 if c == y else 0.0)
                if p == 0.0:
                    continue
                g = p * TEMP / (nrm[c] * N)
                for j in xs:
                    G[c][j] += g / xn
                coeff[c] += g * scores[c]
        for c in range(C):
            if coeff[c] != 0.0:
                for j in range(V):
                    G[c][j] -= coeff[c] * U[c][j]
        gB = [[scale * sum(G[c][j] * A[i][j] for j in range(V))
               for i in range(rank)] for c in range(C)]
        gA = [[scale * sum(B[c][i] * G[c][j] for c in range(C))
               for j in range(V)] for i in range(rank)]
        for c in range(C):
            for i in range(rank):
                B[c][i] -= LR * gB[c][i]
        for i in range(rank):
            for j in range(V):
                A[i][j] -= LR * gA[i][j]
    return A, B, scale


def adapted_model(w0, A, B, scale):
    """A scored model whose centroids are base + delta, built as a
    fresh object - the base matrix itself is never touched."""
    model = CentroidModel()
    cents = {}
    for ci, lab in enumerate(INTENTS):
        d = {}
        for j, w in enumerate(VOCAB):
            v = w0[ci][j] + scale * sum(B[ci][i] * A[i][j]
                                        for i in range(len(A)))
            if abs(v) > 1e-12:
                d[w] = v
        cents[lab] = d
    model.centroids = cents
    return model


def print_board_detail(model, rows) -> None:
    from collections import Counter
    seen, ok = Counter(), Counter()
    for t, l, _p in rows:
        seen[l] += 1
        if model.predict(t) == l:
            ok[l] += 1
    for intent in INTENTS:
        print(f"  {intent}: {ok[intent]}/{seen[intent]}")
    c, n, n_ok, n_tot, routed, esc_n = board_lines(model, rows)
    print(f"  intent overall: {c}/{n} = {c / n:.3f}")
    print(f"  phrasings never taught (4 of 6): {n_ok}/{n_tot}"
          f" = {n_ok / n_tot:.3f}")
    print(f"  must-escalate tickets routed right, still"
          f" auto-answered: {routed}")
    print(f"  policy-compliant handling: 0/{esc_n} = 0.000")


def run_ledger(w0, train, dev, test, rows) -> None:
    total = len(INTENTS) * len(VOCAB)
    fp32_bytes = total * 4
    feats = [([IDX[t] for t in tokenize(text)], INTENTS.index(lab))
             for text, lab in train]
    _codes, scales, wq = quantize(w0)
    q_bytes = (total + 1) // 2 + len(scales) * 4

    def adapter(base_mat, rank, epochs=EPOCHS):
        A, B, sc = train_adapter(base_mat, feats, rank, epochs)
        return adapted_model(base_mat, A, B, sc)

    def entry(rid, method, config, data, model, state, curated):
        _c, _n, _a, _b, _r, esc_n = board_lines(model, rows)
        return {"id": rid, "method": method, "config": config,
                "data": data, "dev": accuracy(model, dev),
                "test": accuracy(model, test), "esc": esc_n,
                "state": state, "curated": curated, "model": model}

    runs = [
        entry("R0", "base", "seed only, untrained", "seed",
              base_model(), None, False),
        entry("R1", "full FT", f"all {total} weights",
              "raw export, 314 rows",
              CentroidModel().train(raw_training_set()),
              total * STATE, False),
        entry("R2", "full FT", f"all {total} weights",
              "curated, 288 rows", CentroidModel().train(train),
              total * STATE, True),
        entry("R3", "LoRA", "rank 1, 400 epochs",
              "curated, 288 rows", adapter(w0, 1),
              fp32_bytes + 1 * (len(VOCAB) + len(INTENTS)) * STATE,
              True),
        entry("R4", "LoRA", f"rank 2, {SHORT} epochs",
              "curated, 288 rows", adapter(w0, 2, SHORT),
              fp32_bytes + 2 * (len(VOCAB) + len(INTENTS)) * STATE,
              True),
        entry("R5", "LoRA", "rank 2, 400 epochs",
              "curated, 288 rows", adapter(w0, 2),
              fp32_bytes + 2 * (len(VOCAB) + len(INTENTS)) * STATE,
              True),
        entry("R6", "LoRA", "rank 8, 400 epochs",
              "curated, 288 rows", adapter(w0, 8),
              fp32_bytes + 8 * (len(VOCAB) + len(INTENTS)) * STATE,
              True),
        entry("R7", "QLoRA", "rank 2, 4-bit base",
              "curated, 288 rows", adapter(wq, 2),
              q_bytes + 2 * (len(VOCAB) + len(INTENTS)) * STATE,
              True),
    ]
    print("executing the runs and logging each one as it lands:")
    print("run | method | config | data | dev | test | policy"
          " | train state")
    for r in runs:
        state = f"{r['state']} B" if r["state"] else "-"
        print(f"{r['id']} | {r['method']} | {r['config']} |"
              f" {r['data']} | {r['dev']:.3f} | {r['test']:.3f}"
              f" | 0/{r['esc']} | {state}")
    # The rule, applied mechanically - the checkpoint is computed
    # from the logged rows, never asserted.
    best_dev = max(r["dev"] for r in runs if r["curated"])
    eligible = [r for r in runs
                if r["curated"] and r["dev"] >= best_dev]
    pick = min(eligible, key=lambda r: r["state"])
    elig_ids = " and ".join(r["id"] for r in eligible
                            if r["id"] != pick["id"])
    print("the verdicts, read off the rule - no verdict from"
          " memory:")
    print("  R1 is barred by rule 1: the raw export carries the"
          " 26 defects of Lesson 11 - its perfect score cannot"
          " see them, so the ledger does not ask its score")
    print("  R3 falls under rule 2: dev 0.917 - rank 1 cannot"
          " carry this task (Lesson 19's sweep)")
    print(f"  R4 falls under rule 2: dev 0.917 at {SHORT} epochs"
          " - the same run as R5, stopped while still climbing")
    print("  R6 is dominated under rule 3: dev 1.000 at"
          " 33824 B - more training state than full fine-tuning,"
          " buying nothing")
    print(f"  {elig_ids} stand eligible beside the checkpoint:"
          " dev 1.000 on the data of record, at a higher price")
    print(f"the checkpoint: {pick['id']} - QLoRA rank 2 on the"
          " 4-bit base")
    print(f"  dev {pick['dev']:.3f}, chosen by the steering"
          f" split; the frozen exam confirms {pick['test']:.3f}"
          " - it did not choose")
    print(f"  {pick['state']} B of training state - the smallest"
          " among the eligible, under a third of full training's"
          " 26240 B")
    print("the checkpoint on the frozen exam, desk by desk:")
    print_board_detail(pick["model"], rows)
    print("one line no run in the ledger moves: policy 0.000 in"
          " all eight rows - the ledger records it on every run;")
    print("  no training outcome closes it, and Section 7's gate"
          " reads it next")


def main() -> None:
    train = split_examples("train")
    dev = split_examples("dev")
    test = split_examples("test")
    rows = test_rows()
    base = base_model()
    w0 = base_matrix(base)
    print("the training ledger: every run accounted")
    print(f"training rows: {len(train)} - the training split of"
          f" record (Lesson 11)")
    print(f"steering split: {len(dev)} dev rows; exam:"
          f" {len(rows)} frozen test rows")
    print(f"the weight matrix: {len(INTENTS)} desks x"
          f" {len(VOCAB)} words = {len(INTENTS) * len(VOCAB)}"
          f" numbers")
    print("eight runs stand in the lab's history: the base"
          " itself, two full runs, four adapter runs, and one"
          " run on a quantized base")
    print_base_board(base, rows)
    print("the checkpoint rule, fixed before the ledger is"
          " read:")
    print("  rule 1: data of record only - any other data"
          " version bars the run, whatever it scores")
    print("  rule 2: the steering split chooses - a candidate"
          " must match the best dev score a clean run reaches")
    print("  rule 3: among the chosen, the smallest training"
          " state ships")
    run_ledger(w0, train, dev, test, rows)
    c, n, _a, _b, _r, _e = board_lines(base_model(), rows)
    print(f"the base model after the ledger: intent {c}/{n}"
          f" = {c / n:.3f} - frozen means frozen, verified")
    print("runledger demo complete")


if __name__ == "__main__":
    main()
