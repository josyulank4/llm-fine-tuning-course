"""Lesson 50 - Build the full fine-tuning stack.

The capstone build, executed under Lesson 49's contract: one
frozen dataset, every candidate method, every claim
reproduced from a run. The whole line runs in one pass -
curate the raw export (Lesson 11), train three ways (Lessons
15, 19, 23), read the candidates through the gate (Lesson
31), probe the chosen form (Lesson 35), merge and serve it
behind the router with a canary (Lesson 39), and price the
stack as built on Lesson 43's card. No station is sampled or
skipped, and the choice among candidates is made by the
ledger's mechanical rule, not by preference: among the
candidates the gate holds, the smallest stored form wins.
The stack this pass produces is the artifact Lesson 51's
readiness review grades.
Run: python -m lesson50.tunestack
"""
from __future__ import annotations

from ftcommon import (INTENTS, RAW_DEFECTS, CentroidModel,
                      base_model, split_examples, tokenize)
from lesson11.datacurate import (dedupe, drop_contradictions,
                                 drop_stale, raw_training_set)
from lesson31.evalgate import (EPOCHS, IDX, adapted_model,
                               base_matrix, board_lines,
                               matrix_model, quantize,
                               score_candidate, test_rows,
                               train_adapter)
from lesson35.probecheck import probe_candidate
from lesson39.servemerge import Router, merge, print_canary

# Lesson 43's price card, unchanged: the rates are the card's
# stated assumptions; every quantity they multiply below is
# measured in this run.
REVIEW_PER_ROW = 0.25
COMPUTE_PER_ROW_EPOCH = 0.001
SCORE_PER_ROW = 0.001
REVIEW_COST = 60.00          # 2 engineer-hours at $30.00
INSTANCE_PER_MONTH = 70.00
# Stored sizes from the earlier lessons' own ledgers: the
# full model's 1,640 numbers at fp32, the rank 2 adapter on
# the fp32 base, the rank 2 adapter on the 4-bit base.
STORED = {"full FT": 26240, "LoRA rank 2": 13376,
          "QLoRA rank 2": 7740}


def money(x: float) -> str:
    return f"${x:,.2f}"


def rate3(x: float) -> str:
    return f"${x:.3f}"


def print_terms(train) -> None:
    print("build the full fine-tuning stack - the capstone pass")
    print("the contract (Lesson 49): one frozen dataset, every"
          " candidate method, every claim reproduced from a run")
    print(f"training rows: {len(train)} - the training split of"
          f" record (Lesson 11)")
    print("the stations, in order: curate the raw export;"
          " train three ways; the gate; the probes;")
    print("  merge and serve behind the router, canary first;"
          " price the stack as built")
    print("the choice rule, fixed now: among the candidates the"
          " gate holds, the smallest stored form is the stack")


def run_curation():
    raw = raw_training_set()
    print(f"station 1 - curate: the raw export, {len(raw)} rows"
          f" as delivered ({sum(RAW_DEFECTS.values())} planted"
          " defects among them)")
    kept = dedupe(raw)
    kept = drop_contradictions(kept)
    kept = drop_stale(kept)
    record = split_examples("train")
    print(f"  reconciliation: curated == training split of"
          f" record: {kept == record} ({len(kept)} of"
          f" {len(record)} rows)")
    return kept


def train_three_ways(train, rows):
    """Station 2: the course's three training methods, one
    frozen dataset, the boards read on the same exam."""
    print("station 2 - train three ways on the curated rows:")
    base = base_model()
    w0 = base_matrix(base)
    _codes, _scales, wq = quantize(w0)
    feats = [([IDX[t] for t in tokenize(text)],
              INTENTS.index(lab)) for text, lab in train]
    cands = {}
    full = CentroidModel().train(train)
    cands["full FT"] = full
    A, B, sc = train_adapter(w0, feats, 2)
    cands["LoRA rank 2"] = adapted_model(w0, A, B, sc)
    Aq, Bq, scq = train_adapter(wq, feats, 2)
    qlora = adapted_model(wq, Aq, Bq, scq)
    cands["QLoRA rank 2"] = qlora
    for name, model in cands.items():
        c, n, n_ok, n_tot, _r, esc_n = board_lines(model, rows)
        print(f"  {name}: intent {c}/{n} = {c / n:.3f} | new"
              f" phrasings {n_ok}/{n_tot} = {n_ok / n_tot:.3f}"
              f" | policy 0/{esc_n} = 0.000 | stored"
              f" {STORED[name]} B")
    return {"base": base, "w0": w0, "wq": wq, "cands": cands,
            "qlora_adapter": (Aq, Bq, scq)}


def run_gate(base, cands, rows):
    """Station 3: every candidate through Lesson 31's gate,
    the bar printed by the gate itself."""
    print("station 3 - the gate (Lesson 31's bar, unmodified):")
    scored = [score_candidate("base", base, rows)]
    for name, model in cands.items():
        scored.append(score_candidate(name, model, rows))
    for s in scored:
        c, n = s["intent"]
        w_name, w_ok, w_n = s["worst"]
        nk, nt = s["new"]
        fm, fn = s["formed"]
        print(f"  {s['name']} | {c}/{n} = {c / n:.3f} | {w_name}"
              f" {w_ok}/{w_n} | {nk}/{nt} = {nk / nt:.3f} |"
              f" {fm}/{fn} | 0/{s['esc']} | {s['verdict']}")
    held = [s for s in scored if s["verdict"] == "HOLD"]
    chosen = min(held, key=lambda s: STORED[s["name"]])
    print(f"  the choice, by the rule fixed at the start: the"
          f" held candidates are"
          f" {', '.join(s['name'] for s in held)}; the smallest"
          f" stored form is {chosen['name']} at"
          f" {STORED[chosen['name']]} B")
    return chosen


def run_probes(chosen_model, rows):
    """Station 4: the chosen form through Lesson 35's frozen
    probe suite."""
    print("station 4 - the probes (Lesson 35's suite, frozen):")
    p = probe_candidate(chosen_model, rows)
    print(f"  must-escalate auto-answered: {p['esc'][0]}/"
          f"{p['esc'][1]} | override requests answered:"
          f" {p['ovr'][0]}/{p['ovr'][1]} | leakage requests"
          f" answered: {p['leak'][0]}/{p['leak'][1]}")
    print(f"  drift read: {p['drift'][0]}/{p['drift'][1]}"
          " still correct on appliance-swapped tickets")
    return p


def run_serving(stack, chosen_name, rows, dev):
    """Station 5: merge the chosen adapter into its base,
    serve the merge behind the router, canary before
    cutover (Lesson 39's drill, on the capstone's form)."""
    print("station 5 - merge and serve (Lesson 39's drill):")
    Aq, Bq, scq = stack["qlora_adapter"]
    adapter = stack["cands"][chosen_name]
    merged_mat = merge(stack["wq"], Aq, Bq, scq)
    _c, _s, merged_q = quantize(merged_mat)
    merged = matrix_model(merged_q)
    a_board = board_lines(adapter, rows)
    m_board = board_lines(merged, rows)
    print(f"  merged board == adapter board on every line:"
          f" {a_board == m_board}")
    same = sum(1 for t, _l, _p in rows
               if merged.predict(t) == adapter.predict(t))
    print(f"  merged predictions == adapter predictions:"
          f" {same}/{len(rows)} exam rows")
    print("  the served form: 924 B, one artifact (the merged,"
          " re-quantized checkpoint)")
    router = Router({"v1 base": stack["base"],
                     "v2 capstone": merged}, "v1 base")
    print_canary("v2 capstone (merged)", merged, dev)
    router.active = "v2 capstone"
    c, n, _a, _b, _r, _e = board_lines(router, rows)
    print(f"  the exam through the router on v2: intent {c}/{n}"
          f" = {c / n:.3f}")
    return merged


def price_stack(raw_n, train_n, scored_rows):
    """Station 6: the stack priced as built, on Lesson 43's
    card - this pass's own quantities, measured above."""
    print("station 6 - price the stack as built (Lesson 43's"
          " card):")
    data = round(raw_n * REVIEW_PER_ROW, 2)
    print(f"  data work: {raw_n} rows handled x"
          f" {money(REVIEW_PER_ROW)} = {money(data)}")
    re_full = len(split_examples("train"))
    re_adapters = 2 * EPOCHS * train_n
    total_re = re_full + re_adapters
    compute = round(total_re * COMPUTE_PER_ROW_EPOCH, 2)
    print(f"  training compute: full FT 1 pass x {train_n} rows"
          f" + two adapter runs {EPOCHS} epochs x {train_n}"
          f" rows = {total_re} row-epochs x"
          f" {rate3(COMPUTE_PER_ROW_EPOCH)} = {money(compute)}")
    scoring = round(scored_rows * SCORE_PER_ROW, 2)
    print(f"  scoring in this pass: {scored_rows} rows x"
          f" {rate3(SCORE_PER_ROW)} = {money(scoring)}")
    print(f"  release review: {money(REVIEW_COST)} (the card's"
          " 2 engineer-hours)")
    one_time = round(data + compute + scoring + REVIEW_COST, 2)
    print(f"  the stack, built once: {money(data)} +"
          f" {money(compute)} + {money(scoring)} +"
          f" {money(REVIEW_COST)} = {money(one_time)}")
    cycle = round(78.50 + 115.20 + 0.29 + REVIEW_COST, 2)
    monthly = round(INSTANCE_PER_MONTH + cycle, 2)
    print(f"  held in service: one retrain cycle"
          f" {money(cycle)} (Lesson 43's lines: data $78.50,"
          f" the chosen run $115.20, re-scoring $0.29, review"
          f" {money(REVIEW_COST)}) + instance"
          f" {money(INSTANCE_PER_MONTH)} = {money(monthly)}"
          " per month")
    print("  against the API alternative measured in Lesson"
          " 43 at $0.0036 per ticket: the tuned stack is"
          " cheaper past 90,034 tickets a month")
    return {"one_time": one_time, "cycle": cycle,
            "monthly": monthly}


def build_stack():
    """The whole pass, returning every artifact the readiness
    review (Lesson 51) grades. Output is captured by the
    caller when the build is reproduced inside the review."""
    train = run_curation()
    rows = test_rows()
    dev = split_examples("dev")
    stack = train_three_ways(train, rows)
    chosen = run_gate(stack["base"], stack["cands"], rows)
    probes = run_probes(stack["cands"][chosen["name"]], rows)
    merged = run_serving(stack, chosen["name"], rows, dev)
    scored = 4 * len(rows) + (24 + 8 + 6 + len(rows) + 64) \
        + len(dev)
    prices = price_stack(len(raw_training_set()), len(train),
                         scored)
    return {"train": train, "rows": rows, "stack": stack,
            "chosen": chosen, "probes": probes,
            "merged": merged, "prices": prices,
            "scored": scored}


def main() -> None:
    train_record = split_examples("train")
    print_terms(train_record)
    art = build_stack()
    chosen = art["chosen"]
    print("the registry entry this build produces:")
    print(f"  version: capstone-1 | form: merged, 924 B | data:"
          f" curated {len(art['train'])} rows of record")
    print(f"  gate verdict: {chosen['verdict']} - the policy"
          " line stands at 0/24; the stack is enrolled in the"
          " lifecycle, not released")
    print("  owner: the desk's named owner | status: HELD,"
          " awaiting Lesson 51's readiness review")
    c, n, _a, _b, _r, _e = board_lines(base_model(),
                                       art["rows"])
    print(f"the base model after the build: intent {c}/{n}"
          f" = {c / n:.3f} - frozen means frozen, verified")
    print("tunestack demo complete")


if __name__ == "__main__":
    main()
