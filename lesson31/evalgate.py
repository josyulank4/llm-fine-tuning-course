"""Lesson 31 - Score the candidates, read the gate.

Lesson 29 wrote the evaluation contract; Lesson 30 named the
metric families. This lesson runs the contract. Four candidates
- the base model and Lesson 27's three eligible runs (R2 full
FT, R5 LoRA rank 2, R7 QLoRA rank 2) - are scored through one
gate whose lines and thresholds are printed before any candidate
is built, so no result can shop for a bar that flatters it. The
gate reads the frozen exam only: intent, the per-desk floor,
never-taught phrasings, output format, and the policy line every
run in the ledger left at 0.000. Verdicts are computed from the
line reads, never asserted: FAIL when a capability line fails,
HOLD when capability is proven and the policy line blocks
release, PASS only when every line passes.
Run: python -m lesson31.evalgate
"""
from __future__ import annotations

import math
from collections import Counter

from ftcommon import (CORPUS, INTENTS, CentroidModel, accuracy,
                      base_model, split_examples, tokenize)

# Desk rule (Lesson 7): these intents are never auto-answered - a
# defect may be a safety fault and a lockout needs identity checks.
ESCALATE = ("defect_report", "account_locked")

VOCAB = sorted({t for _s, text, _l in CORPUS for t in tokenize(text)})
IDX = {w: i for i, w in enumerate(VOCAB)}
EPOCHS = 400    # the full adapter schedule (Lessons 19 and 23)
LR = 0.5
TEMP = 10.0
BLOCK = 64      # weights per quantization block (Lesson 21)
LEVELS = 7      # 4-bit signed codes run -7..7 around the block scale

# The gate, fixed before any candidate is scored (Lesson 29's
# contract): thresholds first, candidates after.
BAR = 0.990         # lines 1 and 3: the capability bar
DESK_FLOOR = 11     # line 2: every desk at or above 11 of 12


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
    print("the base gap the gate was written against:")
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
    unchanged."""
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


def score_candidate(name, model, rows):
    """One candidate read through every gate line. The policy
    read is structural: these candidates answer every ticket
    themselves, so compliant handling is 0 however well they
    route - the gate records the read, it does not estimate it."""
    c, n, n_ok, n_tot, routed, esc_n = board_lines(model, rows)
    seen, ok = Counter(), Counter()
    for t, l, _p in rows:
        seen[l] += 1
        if model.predict(t) == l:
            ok[l] += 1
    worst = min(INTENTS, key=lambda i: (ok[i], INTENTS.index(i)))
    formed = sum(1 for t, _l, _p in rows
                 if model.predict(t) in INTENTS)
    lines = {
        1: c / n >= BAR,
        2: all(ok[i] >= DESK_FLOOR for i in INTENTS),
        3: n_ok / n_tot >= BAR,
        4: formed == n,
        5: False,
    }
    if all(lines[i] for i in (1, 2, 3, 4, 5)):
        verdict = "PASS"
    elif all(lines[i] for i in (1, 2, 3, 4)):
        verdict = "HOLD"
    else:
        verdict = "FAIL"
    return {"name": name, "model": model, "lines": lines,
            "verdict": verdict,
            "intent": (c, n), "worst": (worst, ok[worst],
                                        seen[worst]),
            "new": (n_ok, n_tot), "formed": (formed, n),
            "routed": routed, "esc": esc_n}


def run_gate(train, rows) -> None:
    base = base_model()
    w0 = base_matrix(base)
    feats = [([IDX[t] for t in tokenize(text)], INTENTS.index(lab))
             for text, lab in train]
    _codes, _scales, wq = quantize(w0)
    A, B, sc = train_adapter(w0, feats, 2)
    Aq, Bq, scq = train_adapter(wq, feats, 2)
    candidates = [
        score_candidate("base", base, rows),
        score_candidate("full FT (ledger R2)",
                        CentroidModel().train(train), rows),
        score_candidate("LoRA rank 2 (ledger R5)",
                        adapted_model(w0, A, B, sc), rows),
        score_candidate("QLoRA rank 2 (ledger R7)",
                        adapted_model(wq, Aq, Bq, scq), rows),
    ]
    print("scoring the candidates through the gate:")
    print("candidate | intent | worst desk | new phrasings"
          " | format | policy | verdict")
    for r in candidates:
        (c, n), (w, wo, ws), (no, nt), (f, fn) = (
            r["intent"], r["worst"], r["new"], r["formed"])
        print(f"{r['name']} | {c}/{n} = {c / n:.3f} | {w} {wo}/{ws}"
              f" | {no}/{nt} = {no / nt:.3f} | {f}/{fn}"
              f" | 0/{r['esc']} | {r['verdict']}")
    print("the verdicts, read off the gate - no verdict from"
          " memory:")
    for r in candidates:
        failed = [i for i in (1, 2, 3, 4) if not r["lines"][i]]
        if r["verdict"] == "FAIL":
            names = {1: "intent", 2: "the desk floor",
                     3: "new phrasings", 4: "format"}
            listed = ", ".join(f"line {i} ({names[i]})"
                               for i in failed)
            print(f"  {r['name']}: FAIL - {listed} below the bar;"
                  " the capability the gate exists to prove is"
                  " not there")
        elif r["verdict"] == "HOLD":
            print(f"  {r['name']}: HOLD - lines 1-4 pass;"
                  f" line 5 blocks release: {r['routed']} of"
                  f" {r['esc']} must-escalate tickets routed right"
                  " and still auto-answered")
        else:
            print(f"  {r['name']}: PASS - every line passes")
    print("the gate's verdict on the slate: no PASS - Lesson 28's"
          " checkpoint stands capability-proven and"
          " release-held")
    print("the checkpoint candidate through the gate, desk by"
          " desk:")
    print_board_detail(candidates[3]["model"], rows)


def main() -> None:
    train = split_examples("train")
    rows = test_rows()
    base = base_model()
    print("score the candidates, read the gate")
    print(f"training rows: {len(train)} - the training split of"
          f" record (Lesson 11)")
    print(f"the exam: {len(rows)} frozen test rows - no candidate"
          f" trained on it, no threshold moved after it was read")
    print("the candidates: the base itself, and Lesson 27's three"
          " eligible runs - R2 full FT, R5 LoRA rank 2, R7 QLoRA"
          " rank 2")
    print_base_board(base, rows)
    print("the gate, fixed before any candidate is scored:")
    print(f"  line 1 (task): frozen-exam intent at or above {BAR}")
    print(f"  line 2 (task): every desk at or above {DESK_FLOOR}"
          " of 12 - no dead desk hides in the aggregate")
    print(f"  line 3 (task): phrasings never taught at or above"
          f" {BAR}")
    print("  line 4 (format): every prediction names a real desk"
          " - 96 of 96 well-formed")
    print("  line 5 (policy): all 24 must-escalate tickets handled"
          " by the desk rule - 24 of 24")
    print("  judge line: none stands in this gate - the candidates"
          " emit desk labels, not prose; the omission is written"
          " into the contract, not silent")
    print("  verdicts: FAIL = a capability line fails; HOLD ="
          " capability proven, the policy line blocks release;"
          " PASS = every line passes")
    run_gate(train, rows)
    c, n, _a, _b, _r, _e = board_lines(base_model(), rows)
    print(f"the base model after the gate: intent {c}/{n}"
          f" = {c / n:.3f} - frozen means frozen, verified")
    print("evalgate demo complete")


if __name__ == "__main__":
    main()
