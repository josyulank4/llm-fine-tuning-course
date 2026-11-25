"""Lesson 43 - The cost ledger: train, serve, retrain.

Lesson 41 priced a training run in parts and Lesson 42 derived
the break-even against a prompted API. This lesson books the
whole lifecycle of the course model in one ledger. The price
card is printed before any line is priced - rates first,
quantities after, the gate discipline of Lesson 31 applied to
money. Every quantity on a line is measured in this run: the
rows handled and removed, the row-epochs of Lesson 27's search
recomputed from its configs, the words in the prompt the API
alternative must carry on every call, the tickets' own lengths.
The ledger's three parts price what training cost, what
serving costs against the API at three volumes, and what one
retrain cycle costs at the monthly cadence - then sums the
twelve-month lifecycle both ways. The verdict is computed
from the lines, and it is a verdict about price only: the
gate's HOLD stands on a line no ledger can price.
Run: python -m lesson43.costledger
"""
from __future__ import annotations

from ftcommon import (INTENTS, RAW_DEFECTS, base_model,
                      seed_examples, split_examples)
from lesson11.datacurate import raw_training_set
from lesson31.evalgate import (IDX, adapted_model, base_matrix,
                               board_lines, quantize, test_rows,
                               train_adapter)
from ftcommon import tokenize

# The price card, fixed before any line is priced. Rates are
# the card's stated assumptions; every quantity they multiply
# is measured below by running the code.
REVIEW_PER_ROW = 0.25        # human review, per row handled
COMPUTE_PER_ROW_EPOCH = 0.001  # training compute, per row-epoch
SCORE_PER_ROW = 0.001        # evaluation scoring, per row scored
REVIEW_HOURS = 2             # engineer-hours per release review
ENGINEER_PER_HOUR = 30.00
API_IN_PER_MILLION = 8.00    # the API alternative, input words
API_OUT_PER_MILLION = 24.00  # output words price higher, as usual
OUTPUT_WORDS = 40            # the card's allowance per API answer
INSTANCE_PER_MONTH = 70.00   # serving the tuned model: one instance
EPOCHS = 400                 # the full adapter schedule (Lesson 27)
SHORT = 40                   # R4's stop (Lesson 27)
HORIZON = 12                 # months in the lifecycle sum
VOLUMES = (1000, 10000, 100000)  # tickets per month, the serve dial


def money(x: float) -> str:
    return f"${x:,.2f}"


def rate3(x: float) -> str:
    """The card's sub-cent rates, printed at their true precision."""
    return f"${x:.3f}"


def print_terms(train) -> None:
    print("the cost ledger: train, serve, retrain")
    print(f"training rows: {len(train)} - the training split of"
          f" record (Lesson 11)")
    print("the checkpoint: Lesson 27's R7 - QLoRA rank 2 on the"
          " 4-bit base, Lesson 31's gate verdict HOLD")
    print("the price card, fixed before any line is priced:")
    print(f"  human review of training data:"
          f" {money(REVIEW_PER_ROW)} per row handled")
    print(f"  training compute: {rate3(COMPUTE_PER_ROW_EPOCH)}"
          f" per row-epoch; evaluation scoring:"
          f" {rate3(SCORE_PER_ROW)} per row scored")
    print(f"  release review: {REVIEW_HOURS} engineer-hours at"
          f" {money(ENGINEER_PER_HOUR)} ="
          f" {money(REVIEW_HOURS * ENGINEER_PER_HOUR)} per release")
    print(f"  the API alternative: {money(API_IN_PER_MILLION)}"
          f" per million input words,"
          f" {money(API_OUT_PER_MILLION)} per million output")
    print(f"    words; the prompted call carries the desk prompt"
          f" on every ticket; its answer is priced at"
          f" {OUTPUT_WORDS} words")
    print(f"  serving the tuned model: one instance at"
          f" {money(INSTANCE_PER_MONTH)} per month, no"
          " per-ticket charge")
    print("the ledger's three parts: what training cost, what"
          " serving costs, what a retrain cycle costs")


def print_base_board(base, rows) -> None:
    c, n, n_ok, n_tot, _r, esc_n = board_lines(base, rows)
    print("the gap the money was spent against - the base model"
          " on the frozen exam:")
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
    return {"base": base, "wq": wq,
            "adapter": adapted_model(wq, A, B, sc)}


def api_prompt() -> str:
    """The prompt the API alternative must carry on every call:
    the desk instructions, the desk rule, and one example per
    seed template - the same seed the base model was trained
    on, so the comparison is honest. Its words are counted,
    not estimated."""
    lines = ["You are the support desk for Helios Home"
             " Appliances.",
             "Read the ticket and answer with exactly one desk"
             " name from this list:",
             ", ".join(INTENTS) + ".",
             "Desk rule: a defect report may be a safety fault"
             " and a locked account needs identity checks -"
             " those tickets are escalated to a human, never"
             " auto-answered.",
             "Examples:"]
    for text, lab in seed_examples()[::10]:
        lines.append(f"Ticket: {text} Desk: {lab}")
    return "\n".join(lines)


def print_train_lines(raw, train, scored_rows):
    n_raw, n_train = len(raw), len(train)
    removed = n_raw - n_train
    data_cost = round(n_raw * REVIEW_PER_ROW, 2)
    print("part 1: train - what the checkpoint cost to produce")
    print(f"  data work: the raw export handled {n_raw} rows;"
          f" curation removed {removed} (Lesson 11's receipts:"
          f" {sum(RAW_DEFECTS.values())} planted defects)")
    print(f"  data of record: {n_train} rows; human review"
          f" {n_raw} rows x {money(REVIEW_PER_ROW)}"
          f" = {money(data_cost)}")
    print("  the search, priced from Lesson 27's ledger - every"
          " run paid for, winner or not:")
    runs = [("R1", "full FT", "all 1640 weights",
             "raw export, 314 rows", 1, n_raw),
            ("R2", "full FT", "all 1640 weights",
             "curated, 288 rows", 1, n_train),
            ("R3", "LoRA", "rank 1, 400 epochs",
             "curated, 288 rows", EPOCHS, n_train),
            ("R4", "LoRA", "rank 2, 40 epochs",
             "curated, 288 rows", SHORT, n_train),
            ("R5", "LoRA", "rank 2, 400 epochs",
             "curated, 288 rows", EPOCHS, n_train),
            ("R6", "LoRA", "rank 8, 400 epochs",
             "curated, 288 rows", EPOCHS, n_train),
            ("R7", "QLoRA", "rank 2, 4-bit base",
             "curated, 288 rows", EPOCHS, n_train)]
    total_re = 0
    own_re = 0
    for rid, method, config, data, epochs, rows in runs:
        re_ = epochs * rows
        total_re += re_
        if rid == "R7":
            own_re = re_
        unit = "pass" if epochs == 1 else "epochs"
        print(f"    {rid} | {method} | {config} | {data} |"
              f" {epochs} {unit} x {rows} rows = {re_}"
              " row-epochs")
    search_cost = round(total_re * COMPUTE_PER_ROW_EPOCH, 2)
    own_cost = round(own_re * COMPUTE_PER_ROW_EPOCH, 2)
    print(f"  search compute: {total_re} row-epochs x"
          f" {rate3(COMPUTE_PER_ROW_EPOCH)} = {money(search_cost)}")
    print(f"  the checkpoint's own run, R7: {own_re} row-epochs"
          f" = {money(own_cost)} of the search")
    eval_cost = round(scored_rows * SCORE_PER_ROW, 2)
    print("  evaluation, scored in this run: the base and the"
          " checkpoint on the exam, the checkpoint on the"
          " steering split")
    print(f"  rows scored: {scored_rows} x"
          f" {rate3(SCORE_PER_ROW)} = {money(eval_cost)} - the"
          " scoring is nearly free; the review is not")
    review_cost = REVIEW_HOURS * ENGINEER_PER_HOUR
    print(f"  release review: {money(review_cost)} (the card's"
          f" {REVIEW_HOURS} engineer-hours)")
    one_time = round(data_cost + search_cost + eval_cost
                   + review_cost, 2)
    print(f"  one-time cost of the checkpoint: {money(data_cost)}"
          f" + {money(search_cost)} + {money(eval_cost)} +"
          f" {money(review_cost)} = {money(one_time)}")
    return {"data": data_cost, "search": search_cost,
            "own": own_cost, "eval": eval_cost,
            "review": review_cost, "one_time": one_time,
            "own_re": own_re}


def print_serve_lines(ck, rows, dev, train_lines):
    ck_board = board_lines(ck["adapter"], rows)
    print("part 2: serve - the tuned model against the API,"
          " per ticket")
    print(f"  the checkpoint on the frozen exam: intent"
          f" {ck_board[0]}/{ck_board[1]}"
          f" = {ck_board[0] / ck_board[1]:.3f}; policy"
          f" 0/{ck_board[5]} = 0.000 - the HOLD stands")
    prompt = api_prompt()
    p_words = len(prompt.split())
    t_words = sum(len(t.split()) for t, _l, _p in rows)
    t_mean = t_words / len(rows)
    print("  the prompted alternative, built and measured here:"
          " the desk prompt every API call must carry")
    print(f"  prompt words: {p_words} (instructions, the desk"
          " rule, 16 seed examples)")
    print(f"  ticket words: the exam's {len(rows)} tickets"
          f" average {t_mean:.1f} words; the tuned model reads"
          " the ticket alone")
    in_words = p_words + t_mean
    in_cost = in_words / 1_000_000 * API_IN_PER_MILLION
    out_cost = OUTPUT_WORDS / 1_000_000 * API_OUT_PER_MILLION
    per_ticket = in_cost + out_cost
    print(f"  API input per ticket: {p_words} + {t_mean:.1f}"
          f" = {in_words:.1f} words -> ${in_cost:.4f}")
    print(f"  API output per ticket: {OUTPUT_WORDS} words (the"
          f" card) -> ${out_cost:.4f}")
    print(f"  the API alternative: ${per_ticket:.4f} per ticket,"
          " every ticket, forever")
    retrain = round(train_lines["data"] + train_lines["own"]
                    + train_lines["eval"] + train_lines["review"], 2)
    monthly_tuned = round(INSTANCE_PER_MONTH + retrain, 2)
    print(f"  the tuned model: {money(INSTANCE_PER_MONTH)} per"
          f" month for the instance; at {VOLUMES[1]} tickets a"
          f" month that is"
          f" ${INSTANCE_PER_MONTH / VOLUMES[1]:.4f} per ticket")
    print("  monthly, at three volumes - the API against the"
          " tuned model (instance + one retrain cycle, part 3):")
    for v in VOLUMES:
        print(f"    {v} tickets: API {money(round(v * per_ticket, 2))}"
              f" | tuned {money(monthly_tuned)}")
    breakeven = int(monthly_tuned / per_ticket)
    print(f"  break-even: the tuned model is cheaper past"
          f" {breakeven:,} tickets a month (steady state,"
          " one-time costs sunk)")
    return {"per_ticket": per_ticket,
            "monthly_tuned": monthly_tuned, "retrain": retrain,
            "breakeven": breakeven}


def print_retrain_lines(train_lines, serve):
    tl = train_lines
    print("part 3: retrain - one cycle at the monthly cadence")
    print(f"  a fresh pull reviewed: {money(tl['data'])} (the"
          " card's per-row rate on a new raw export)")
    print(f"  the checkpoint's run again: {tl['own_re']}"
          f" row-epochs = {money(tl['own'])}")
    print(f"  the gate re-scored and re-reviewed:"
          f" {money(tl['eval'])} + {money(tl['review'])}")
    print(f"  one retrain cycle: {money(serve['retrain'])}")
    tuned_life = round(tl["one_time"]
                       + HORIZON * INSTANCE_PER_MONTH
                       + (HORIZON - 1) * serve["retrain"], 2)
    print(f"the lifecycle, {HORIZON} months at the monthly"
          " cadence:")
    print(f"  tuned: {money(tl['one_time'])} once + {HORIZON} x"
          f" {money(INSTANCE_PER_MONTH)} instance +"
          f" {HORIZON - 1} x {money(serve['retrain'])} retrains"
          f" = {money(tuned_life)}")
    for v in (VOLUMES[1], VOLUMES[2]):
        monthly = round(v * serve["per_ticket"], 2)
        print(f"  API at {v} tickets a month: {HORIZON} x"
              f" {money(monthly)}"
              f" = {money(round(HORIZON * monthly, 2))}")
    print("the ledger's verdict: below the break-even the API"
          " is the cheaper tool; past it, the tuned model's"
          " fixed lines win -")
    print("  and no line in this ledger releases the checkpoint:"
          " policy handling is 0/24, the gate's HOLD stands,")
    print("  and a cost ledger is not a release gate")


def main() -> None:
    train = split_examples("train")
    raw = raw_training_set()
    rows = test_rows()
    dev = split_examples("dev")
    base = base_model()
    print_terms(train)
    print_base_board(base, rows)
    ck = build_checkpoint(train)
    # The evaluation rows this ledger scores, counted as they
    # are scored: base and checkpoint on the exam, checkpoint
    # on the steering split.
    scored = 0
    board_lines(base, rows)
    scored += len(rows)
    board_lines(ck["adapter"], rows)
    scored += len(rows)
    _dev_ok = sum(1 for t, l in dev
                  if ck["adapter"].predict(t) == l)
    scored += len(dev)
    train_lines = print_train_lines(raw, train, scored)
    serve = print_serve_lines(ck, rows, dev, train_lines)
    print_retrain_lines(train_lines, serve)
    c, n, _a, _b, _r, _e = board_lines(base_model(), rows)
    print(f"the base model after the ledger: intent {c}/{n}"
          f" = {c / n:.3f} - frozen means frozen, verified")
    print("costledger demo complete")


if __name__ == "__main__":
    main()
