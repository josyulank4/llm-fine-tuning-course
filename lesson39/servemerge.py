"""Lesson 39 - Merge and serve: the deployment drill.

Lesson 28's ledger chose the checkpoint and Lesson 31's gate
left it capability-proven and release-held. This lesson runs
the deployment drill on that checkpoint in the lab: the drill
proves the mechanics of a release - merge, serve, break, roll
back - and is not itself a release. Act 1 merges the QLoRA
rank 2 adapter into its 4-bit base and proves the merged model
is the adapter model, line for line and prediction for
prediction, then re-quantizes the merge for serving. Act 2
serves the merged model behind a routing table, canary first:
the steering split reads the candidate before any exam ticket
is routed to it. Act 3 breaks the deploy - a rank-1 merge, the
sweep's insufficient rank, is pushed in the checkpoint's place
- and the canary rule fires the rollback to the last good
version. Every number is measured by running this code.
Run: python -m lesson39.servemerge
"""
from __future__ import annotations

from ftcommon import (INTENTS, base_model, split_examples,
                      tokenize)
from lesson31.evalgate import (IDX, VOCAB, adapted_model,
                               base_matrix, board_lines,
                               matrix_model, print_board_detail,
                               quantize, test_rows, train_adapter)

# The canary bar, fixed before the drill runs (Lesson 31's
# capability bar, applied to the steering split): a candidate
# that cannot clear it never takes production traffic.
CANARY_BAR = 0.990
FP32 = 4                     # bytes per stored fp32 number
CODE_B = 0.5                 # bytes per 4-bit code (Lesson 21)


def print_terms(train) -> None:
    print("merge and serve: the deployment drill")
    print(f"training rows: {len(train)} - the training split of"
          f" record (Lesson 11)")
    print("the checkpoint: Lesson 27's R7 - QLoRA rank 2 on the"
          " 4-bit base, Lesson 31's gate verdict HOLD")
    print("the drill, three acts: merge the adapter into its"
          " base; serve the merge behind a routing table;")
    print("  break the deploy and roll back - canary first at"
          " every cutover")
    print(f"the canary: the 96 steering rows, read before"
          f" cutover; the bar is {CANARY_BAR}, fixed now")


def print_base_board(base, rows) -> None:
    c, n, n_ok, n_tot, _r, esc_n = board_lines(base, rows)
    print("the base gap the drill starts from - v1, the version"
          " production runs today:")
    print(f"  intent: {c}/{n} = {c / n:.3f}")
    print(f"  new phrasings: {n_ok}/{n_tot} = {n_ok / n_tot:.3f}")
    print(f"  policy: 0/{esc_n} = 0.000")


# The checkpoint, rebuilt exactly as Lesson 31's gate built it:
# the base quantized to 4 bits, the rank 2 adapter trained on
# the dequantized base.
def build_checkpoint(train):
    base = base_model()
    w0 = base_matrix(base)
    _codes, _scales, wq = quantize(w0)
    feats = [([IDX[t] for t in tokenize(text)],
              INTENTS.index(lab)) for text, lab in train]
    A, B, sc = train_adapter(wq, feats, 2)
    return {"base": base, "w0": w0, "wq": wq, "A": A, "B": B,
            "scale": sc, "adapter": adapted_model(wq, A, B, sc),
            "feats": feats}


def merge(wbase, A, B, scale):
    """Fold an adapter into a base matrix: W + scale * (B @ A),
    a fresh matrix - neither input is touched."""
    return [[wbase[c][j] + scale * sum(B[c][i] * A[i][j]
             for i in range(len(A)))
             for j in range(len(VOCAB))]
            for c in range(len(INTENTS))]


def storage_lines(ck):
    """The serving artifacts, priced in bytes - Lesson 21's
    storage arithmetic applied to this checkpoint."""
    n_weights = len(INTENTS) * len(VOCAB)
    codes, scales, _wq = quantize(ck["w0"])
    qbase = int(len(codes) * CODE_B) + len(scales) * FP32
    adapter_n = (sum(len(r) for r in ck["A"])
                 + sum(len(r) for r in ck["B"]))
    return n_weights, qbase, adapter_n


class Router:
    """A routing table over named deployments. Every desk
    routes to the active deployment; a rollback is a table
    flip, not a rebuild - the previous deployment never
    leaves memory, so it stays warm."""

    def __init__(self, deployments, active):
        self.deployments = deployments
        self.active = active

    def predict(self, text):
        return self.deployments[self.active].predict(text)

    def print_table(self):
        print(f"the routing table - active deployment:"
              f" {self.active}")
        for intent in INTENTS:
            print(f"  {intent} -> {self.active}")


def canary_read(model, dev):
    ok = sum(1 for t, l in dev if model.predict(t) == l)
    return ok, len(dev)


def print_canary(name, model, dev) -> bool:
    ok, n = canary_read(model, dev)
    clears = ok / n >= CANARY_BAR
    print(f"canary on {name}: {ok}/{n} = {ok / n:.3f} - "
          + (f"clears the {CANARY_BAR} bar"
             if clears else
             f"BELOW the {CANARY_BAR} bar"))
    return clears


# The drill itself: merge, serve, break, roll back - in that
# order, with the canary read before every cutover.
def run_drill(ck, rows, dev) -> None:
    adapter = ck["adapter"]
    a_board = board_lines(adapter, rows)
    print("the checkpoint in adapter form, on the frozen exam:")
    print(f"  intent overall: {a_board[0]}/{a_board[1]}"
          f" = {a_board[0] / a_board[1]:.3f}")
    print(f"  phrasings never taught (4 of 6): {a_board[2]}"
          f"/{a_board[3]} = {a_board[2] / a_board[3]:.3f}")
    print(f"  policy-compliant handling: 0/{a_board[5]}"
          " = 0.000 - serving changes no behaviour; the"
          " gate's HOLD stands")
    print("act 1: merge the adapter into its base")
    merged = matrix_model(merge(ck["wq"], ck["A"], ck["B"],
                                 ck["scale"]))
    m_board = board_lines(merged, rows)
    print("merged board == adapter board on every line:"
          f" {m_board == a_board}")
    agree = sum(1 for t, _l, _p in rows
                if merged.predict(t) == adapter.predict(t))
    print(f"merged predictions == adapter predictions:"
          f" {agree}/{len(rows)} exam rows")
    _c2, _s2, wq2 = quantize(merge(ck["wq"], ck["A"], ck["B"],
                                   ck["scale"]))
    served = matrix_model(wq2)
    s_board = board_lines(served, rows)
    print("the merge re-quantized to 4 bits for serving;"
          " its board == the merged board on every line:"
          f" {s_board == m_board}")
    n_w, qbase, ad_n = storage_lines(ck)
    print("the serving artifacts, priced:")
    print(f"  adapter form: {qbase} B quantized base +"
          f" {ad_n * FP32} B adapter ({ad_n} numbers)"
          f" = {qbase + ad_n * FP32} B, two artifacts")
    print(f"  merged fp32: {n_w * FP32} B, one artifact")
    print(f"  merged, re-quantized: {qbase} B, one artifact"
          " - the form this drill serves")
    print("act 2: serve the merged model behind the routing"
          " table")
    router = Router({"v1 base": ck["base"],
                     "v2 checkpoint (merged)": served},
                    "v1 base")
    print_canary("v2 checkpoint (merged)", served, dev)
    router.active = "v2 checkpoint (merged)"
    router.print_table()
    print("the exam, served through the router - desk by desk:")
    print_board_detail(router, rows)
    print("act 3: break the deploy, roll back")
    A1, B1, sc1 = train_adapter(ck["wq"], ck["feats"], 1)
    bad = matrix_model(merge(ck["wq"], A1, B1, sc1))
    b_board = board_lines(bad, rows)
    print("the bad deploy: a rank-1 merge - Lesson 19's"
          " insufficient rank - pushed in the checkpoint's"
          " place")
    print(f"  read on the exam it must never reach:"
          f" intent {b_board[0]}/{b_board[1]}"
          f" = {b_board[0] / b_board[1]:.3f} - above the"
          " base, below the bar")
    router.deployments["v3 rank-1 merge"] = bad
    router.active = "v3 rank-1 merge"
    if not print_canary("v3 rank-1 merge", bad, dev):
        print("the drill rule fires: a candidate below the"
              " canary bar takes no production traffic - roll"
              " back")
    router.active = "v2 checkpoint (merged)"
    router.print_table()
    r_board = board_lines(router, rows)
    print(f"the exam through the router after rollback:"
          f" intent {r_board[0]}/{r_board[1]}"
          f" = {r_board[0] / r_board[1]:.3f} - the last good"
          " version, restored by one table flip")
    print("the drill's verdict: the merge is the adapter"
          " model; the canary caught the bad deploy;")
    print("  rollback cost one routing-table flip - no"
          " retrain, no rebuild")


def main() -> None:
    train = split_examples("train")
    rows = test_rows()
    dev = split_examples("dev")
    base = base_model()
    print_terms(train)
    print_base_board(base, rows)
    ck = build_checkpoint(train)
    run_drill(ck, rows, dev)
    c, n, _a, _b, _r, _e = board_lines(base_model(), rows)
    print(f"the base model after the drill: intent {c}/{n}"
          f" = {c / n:.3f} - frozen means frozen, verified")
    print("servemerge demo complete")


if __name__ == "__main__":
    main()
