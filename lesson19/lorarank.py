"""Lesson 19 - Sweep the rank, read the trade.

LoRA on the course lab: the base model's centroid matrix (8 desks x
205 words) stays frozen, and the only trainable thing is a low-rank
delta, delta = scale * B @ A - A is rank x 205, B is 8 x rank, B
starts at zero so the delta starts at zero, and both are trained by
gradient descent on the training split against the model's own
cosine scoring rule. The sweep trains ranks 1, 2, 4 and 8 with alpha
held at twice the rank (scale 2.0 throughout), so rank is the only
dial that moves, and reads each adapter on the steering split and
the frozen exam: accuracy bought per trainable number, on one table.
Run: python -m lesson19.lorarank
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
RANKS = (1, 2, 4, 8)
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
    print("the base gap the adapters have to close:")
    print(f"  intent: {c}/{n} = {c / n:.3f}")
    print(f"  new phrasings: {n_ok}/{n_tot} = {n_ok / n_tot:.3f}")
    print(f"  policy: 0/{esc_n} = 0.000")


def train_adapter(w0, feats, rank):
    """Train one adapter: delta = 2.0 * B @ A over the frozen base,
    B from zeros, A from a fixed pattern (no randomness anywhere),
    full-batch gradient descent on cross-entropy over the model's
    own cosine scores - the training objective is the scoring rule
    the exam will use, so nothing is gained in translation."""
    scale = 2.0  # alpha = 2 * rank at every rank; rank is the dial
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
    fresh object - the base model's own numbers are never touched."""
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


def run_sweep(w0, train, dev, test):
    """The sweep itself: one adapter per rank, trained identically,
    read on the steering split and the frozen exam."""
    feats = [([IDX[t] for t in tokenize(text)], INTENTS.index(lab))
             for text, lab in train]
    print("the sweep: ranks 1, 2, 4, 8 - one adapter per rank,"
          " trained identically")
    print("alpha is twice the rank at every rank, so the scale"
          " alpha/r holds at 2.0 and rank is the only dial"
          " that moves")
    print("rank | trainable | artifact | dev intent | test intent")
    full_acc = accuracy(CentroidModel().train(train), test)
    results = {}
    for rank in RANKS:
        A, B, scale = train_adapter(w0, feats, rank)
        model = adapted_model(w0, A, B, scale)
        n_train = rank * (len(VOCAB) + len(INTENTS))
        results[rank] = (A, B, scale, model)
        print(f"{rank:>4} | {n_train:>9} | {n_train * 4:>5} B"
              f" | {accuracy(model, dev):.3f}"
              f" | {accuracy(model, test):.3f}")
    sufficient = next(r for r in RANKS
                      if accuracy(results[r][3], test) >= full_acc)
    n_train = sufficient * (len(VOCAB) + len(INTENTS))
    print(f"the smallest sufficient rank: {sufficient} -"
          f" {n_train} trainable numbers reach the full run's"
          f" exam score ({full_acc:.3f})")
    return results, sufficient


def print_rank_detail(model, rows) -> None:
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


def main() -> None:
    train = split_examples("train")
    dev = split_examples("dev")
    test = split_examples("test")
    rows = test_rows()
    base = base_model()
    w0 = base_matrix(base)
    print("low-rank adaptation: a frozen base, a small trainable"
          " delta")
    print(f"training rows: {len(train)} - the training split of"
          f" record (Lesson 11)")
    print(f"steering split: {len(dev)} dev rows; exam:"
          f" {len(rows)} frozen test rows")
    print(f"the weight matrix: {len(INTENTS)} desks x"
          f" {len(VOCAB)} words = {len(INTENTS) * len(VOCAB)}"
          f" numbers")
    print("Lesson 15's full run wrote 311 of the tuned model's"
          " 322 carried numbers")
    print_base_board(base, rows)
    results, sufficient = run_sweep(w0, train, dev, test)
    A, B, scale, model = results[sufficient]
    print(f"rank {sufficient} on the frozen exam, desk by desk:")
    print_rank_detail(model, rows)
    # Merging (Lesson 20's subject, checked here): fold the delta
    # into a copy of the base numbers and re-read the board - the
    # merged model must be the adapter model, line for line.
    merged = CentroidModel()
    merged.centroids = {
        lab: {w: base.centroids[lab].get(w, 0.0)
              + scale * sum(B[ci][i] * A[i][IDX[w]]
                            for i in range(len(A)))
              for w in VOCAB
              if abs(base.centroids[lab].get(w, 0.0)
                     + scale * sum(B[ci][i] * A[i][IDX[w]]
                                   for i in range(len(A)))) > 1e-12}
        for ci, lab in enumerate(INTENTS)}
    same = board_lines(merged, rows) == board_lines(model, rows)
    print("merging the adapter back into the base:")
    print(f"  merged board == adapter board on every line: {same}")
    c, n, _a, _b, _r, _e = board_lines(base_model(), rows)
    print(f"the base model after the sweep: intent {c}/{n}"
          f" = {c / n:.3f} - frozen means frozen, verified")
    print("lorarank demo complete")


if __name__ == "__main__":
    main()
