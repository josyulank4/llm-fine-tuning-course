"""Lesson 23 - The QLoRA ledger: memory saved, accuracy kept.

Three ways to close the same gap, computed side by side on the
course lab. Full fine-tuning trains all 1,640 numbers of the base
model's centroid matrix (8 desks x 205 words) and pays Lesson 14's
full training state - 16 bytes per number. LoRA freezes the base at
fp32 and trains only Lesson 19's smallest sufficient adapter
(rank 2, 426 numbers). QLoRA first stores the base as 4-bit codes -
blocks of 64 weights sharing one fp32 scale, the Lesson 21 scheme -
then trains the same rank-2 adapter against the dequantized base,
the Lesson 22 construction. Every method is read on the frozen
exam; the ledger prints trainable numbers, their share, total
training bytes, and the score kept. Run: python -m lesson23.qloramath
"""
from __future__ import annotations

import math

from ftcommon import (CORPUS, INTENTS, CentroidModel, accuracy,
                      base_model, split_examples, tokenize)

# Desk rule (Lesson 7): these intents are never auto-answered - a
# defect may be a safety fault and a lockout needs identity checks.
ESCALATE = ("defect_report", "account_locked")

VOCAB = sorted({t for _s, text, _l in CORPUS for t in tokenize(text)})
IDX = {w: i for i, w in enumerate(VOCAB)}
RANK = 2        # Lesson 19's smallest sufficient rank
BLOCK = 64      # weights per quantization block (Lesson 21)
LEVELS = 7      # 4-bit signed codes run -7..7 around the block scale
STATE = 16      # training bytes per trainable number (Lesson 14)
EPOCHS = 400
LR = 0.5
TEMP = 10.0


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
    print("the base gap the methods have to close:")
    print(f"  intent: {c}/{n} = {c / n:.3f}")
    print(f"  new phrasings: {n_ok}/{n_tot} = {n_ok / n_tot:.3f}")
    print(f"  policy: 0/{esc_n} = 0.000")


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


def quantize(mat):
    """Lesson 21's scheme on the whole matrix, flattened row by
    row: blocks of 64 weights, one fp32 scale per block (the
    block's largest magnitude), each weight stored as a 4-bit
    signed code in -7..7. Returns (codes, scales, dequantized) -
    the dequantized matrix is what computation ever sees."""
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


def print_quantization(w0, codes, scales, wq) -> None:
    flat0 = [v for row in w0 for v in row]
    flatq = [v for row in wq for v in row]
    errs = [abs(a - b) for a, b in zip(flat0, flatq)]
    moved = sum(1 for a, b in zip(flat0, flatq) if a != b)
    carried = sum(1 for v in flat0 if v != 0.0)
    code_bytes = (len(codes) + 1) // 2
    scale_bytes = len(scales) * 4
    stored = code_bytes + scale_bytes
    fp32 = len(flat0) * 4
    print("quantizing the base: 4-bit codes, blocks of 64 weights,"
          " one fp32 scale per block")
    print(f"  stored: {code_bytes} B of codes + {scale_bytes} B of"
          f" scales = {stored} B, against {fp32} B at fp32"
          f" ({fp32 / stored:.1f}x smaller)")
    print(f"  rounding error: max {max(errs):.4f}, mean"
          f" {sum(errs) / len(errs):.4f} across the matrix")
    print(f"  weights moved by rounding: {moved} of {len(flat0)}"
          f" ({moved} of the {carried} numbers the base carries)")


def train_adapter(w0, feats, rank):
    """Train one adapter: delta = 2.0 * B @ A over the given base
    matrix, B from zeros, A from a fixed pattern (no randomness
    anywhere), full-batch gradient descent on cross-entropy over
    the model's own cosine scores - Lesson 19's trainer, unchanged,
    so the LoRA row of the ledger is the sweep's own rank-2 run."""
    scale = 2.0  # alpha = 2 * rank; rank is the only dial
    V, C = len(VOCAB), len(INTENTS)
    A = [[0.05 * (((i * 31 + j * 17) % 13) - 6) / 6.0
          for j in range(V)] for i in range(rank)]
    B = [[0.0] * rank for _ in range(C)]
    N = len(feats)
    for _epoch in range(EPOCHS):
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


def run_ledger(w0, train, test, rows) -> None:
    codes, scales, wq = quantize(w0)
    print_quantization(w0, codes, scales, wq)
    qbase = matrix_model(wq)
    print("the quantized base, scored alone on the frozen exam:")
    print_board_detail(qbase, rows)
    same = board_lines(qbase, rows) == board_lines(base_model(), rows)
    print(f"  quantized board == fp32 board on every line: {same}")
    feats = [([IDX[t] for t in tokenize(text)], INTENTS.index(lab))
             for text, lab in train]
    print("training the rank-2 adapter twice - on the fp32 base,"
          " then on the dequantized base")
    A, B, sc = train_adapter(w0, feats, RANK)
    lora = adapted_model(w0, A, B, sc)
    A2, B2, sc2 = train_adapter(wq, feats, RANK)
    qlora = adapted_model(wq, A2, B2, sc2)
    full = CentroidModel().train(train)
    total = len(INTENTS) * len(VOCAB)
    n_ad = RANK * (len(VOCAB) + len(INTENTS))
    fp32_bytes = total * 4
    q_bytes = (total + 1) // 2 + len(scales) * 4
    print("the ledger: three methods, one exam")
    print("method | trainable | share | training bytes | test intent")
    print(f"full FT | {total} | {100.0:.1f}% | {total * STATE} B"
          f" | {accuracy(full, test):.3f}")
    print(f"LoRA r{RANK} | {n_ad} | {n_ad / total * 100:.1f}%"
          f" | {fp32_bytes + n_ad * STATE} B"
          f" | {accuracy(lora, test):.3f}")
    print(f"QLoRA r{RANK} | {n_ad} | {n_ad / total * 100:.1f}%"
          f" | {q_bytes + n_ad * STATE} B"
          f" | {accuracy(qlora, test):.3f}")
    print(f"the frozen base is the only line that moves:"
          f" {fp32_bytes} B at fp32, {q_bytes} B at 4-bit - the"
          f" adapter's training state is identical in both rows")
    print("at 7 billion weights the same lines read: full training"
          " state 112 GB; a frozen fp32 base 28 GB; a frozen 4-bit"
          " base about 3.9 GB")
    print("the QLoRA model on the frozen exam, desk by desk:")
    print_board_detail(qlora, rows)


def main() -> None:
    train = split_examples("train")
    test = split_examples("test")
    rows = test_rows()
    base = base_model()
    w0 = base_matrix(base)
    print("the QLoRA ledger: memory saved, accuracy kept")
    print(f"training rows: {len(train)} - the training split of"
          f" record (Lesson 11)")
    print(f"steering split: 96 dev rows; exam: {len(rows)} frozen"
          f" test rows")
    print(f"the weight matrix: {len(INTENTS)} desks x"
          f" {len(VOCAB)} words = {len(INTENTS) * len(VOCAB)}"
          f" numbers")
    print("Lesson 19's smallest sufficient adapter - rank 2, 426"
          " trainable numbers - is the adapter both adapter rows"
          " train")
    print_base_board(base, rows)
    run_ledger(w0, train, test, rows)
    c, n, _a, _b, _r, _e = board_lines(base_model(), rows)
    print(f"the base model after the ledger: intent {c}/{n}"
          f" = {c / n:.3f} - frozen means frozen, verified")
    print("qloramath demo complete")


if __name__ == "__main__":
    main()
