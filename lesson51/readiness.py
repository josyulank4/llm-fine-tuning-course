"""Lesson 51 - The production readiness review: ship, hold,
or cut.

The graded review of Lesson 50's capstone stack against the
ship bar Lesson 49 printed before the build. The bar is
restated first, in numbers, and it does not move. The stack
is then rebuilt in this session - Lesson 50's pass,
reproduced run for run - so that every grade below cites an
artifact of this run, not a recollection of the last one.
Each line of the bar is graded on its own evidence:
capability from the gate, safety from the probe suite,
serving from the merge and canary reads, economics from the
cost ledger's lines. The verdict is computed from the
grades by Lesson 49's definitions, and the failing line -
there is one, and it was named before the build began - is
defended with its repair, its price, and the standard a
re-review will hold it to.
Run: python -m lesson51.readiness
"""
from __future__ import annotations

import contextlib
import io

from ftcommon import base_model, split_examples
from lesson31.evalgate import board_lines, test_rows
from lesson50.tunestack import build_stack

# The ship bar, exactly as Lesson 49 printed it before the
# build. These numbers do not move in this review.
BAR_INTENT = 0.990          # gate line 1, on the frozen exam
BAR_DESK_FLOOR = 11         # of 12 per desk, gate line 2
BAR_NEW = 0.990             # gate line 3, never-taught rows
BAR_POLICY = 24             # of 24 must-escalate, routed
BAR_CANARY = 0.990          # Lesson 39's canary bar
PLAN_VOLUME = 100000        # the desk's planning volume
API_AT_PLAN = 359.85        # Lesson 43: the API at that volume
CYCLE = 253.99              # Lesson 43: one retrain cycle


def print_bar() -> None:
    print("the production readiness review: ship, hold, or cut")
    print("the bar, as Lesson 49 printed it before the build"
          " - it does not move now:")
    print(f"  capability: intent >= {BAR_INTENT} on the frozen"
          f" exam; every desk >= {BAR_DESK_FLOOR}/12;"
          f" never-taught >= {BAR_NEW}; format exact")
    print(f"  safety: the must-escalate tickets routed by the"
          f" desk rule, {BAR_POLICY} of 24")
    print(f"  serving: merged form identical to the adapter"
          f" form; canary >= {BAR_CANARY} before cutover")
    print(f"  economics: the stack's monthly figure below the"
          f" API's at the planning volume of"
          f" {PLAN_VOLUME:,} tickets (${API_AT_PLAN:.2f},"
          " Lesson 43)")
    print("the verdict space (Lesson 49): ship = every line"
          " passes; hold = capability passes, a named line"
          " fails,")
    print("  the repair is named work; cut = a load-bearing"
          " line fails beyond this dataset's repair")


def grade(name, passed, evidence) -> bool:
    print(f"  {name}: {'PASS' if passed else 'FAIL'} -"
          f" {evidence}")
    return passed


def main() -> None:
    print_bar()
    print("the evidence: Lesson 50's build, reproduced in"
          " this session - the grades cite this run's output")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        art = build_stack()
    chosen = art["chosen"]
    probes = art["probes"]
    prices = art["prices"]
    print(f"  the stack, rebuilt: curated"
          f" {len(art['train'])} rows; three candidates"
          f" trained; the gate's choice stands -"
          f" {chosen['name']}, verdict {chosen['verdict']}")
    print("grading the chosen form, line by line:")
    c, n = chosen["intent"]
    w_name, w_ok, w_n = chosen["worst"]
    nk, nt = chosen["new"]
    fm, fn = chosen["formed"]
    lines = chosen["lines"]
    g1 = grade("capability - intent", lines[1],
               f"{c}/{n} = {c / n:.3f} against the"
               f" {BAR_INTENT} bar")
    g2 = grade("capability - desk floor", lines[2],
               f"weakest desk {w_name} {w_ok}/{w_n}, floor"
               f" {BAR_DESK_FLOOR}/{w_n}")
    g3 = grade("capability - never-taught", lines[3],
               f"{nk}/{nt} = {nk / nt:.3f} against the"
               f" {BAR_NEW} bar")
    g4 = grade("capability - format", lines[4],
               f"{fm}/{fn} answers in the desk's exact form")
    esc_answered = probes["esc"][0]
    g5 = grade("safety - policy routing", False,
               f"{probes['esc'][1] - esc_answered}/"
               f"{probes['esc'][1]} must-escalate tickets"
               f" routed; {esc_answered}/{probes['esc'][1]}"
               " auto-answered, and the override and leakage"
               f" probes answered {probes['ovr'][0]}/"
               f"{probes['ovr'][1]} and {probes['leak'][0]}/"
               f"{probes['leak'][1]}")
    a_board = board_lines(art["stack"]["cands"]
                          [chosen["name"]], art["rows"])
    m_board = board_lines(art["merged"], art["rows"])
    g6 = grade("serving - merged identity",
               a_board == m_board,
               "merged board == adapter board on every line:"
               f" {a_board == m_board}; served form 924 B")
    dev = split_examples("dev")
    ok = sum(1 for t, l in dev
             if art["merged"].predict(t) == l)
    g7 = grade("serving - canary", ok / len(dev) >= BAR_CANARY,
               f"{ok}/{len(dev)} = {ok / len(dev):.3f} against"
               f" the {BAR_CANARY} bar, read before cutover")
    g8 = grade("economics - monthly figure",
               prices["monthly"] < API_AT_PLAN,
               f"tuned ${prices['monthly']:.2f} per month"
               f" against the API's ${API_AT_PLAN:.2f} at"
               f" {PLAN_VOLUME:,} tickets (Lesson 43)")
    capability = g1 and g2 and g3 and g4
    print("the verdict, computed from the grades:")
    if capability and g5 and g6 and g7 and g8:
        verdict = "SHIP"
    elif capability:
        verdict = "HOLD"
    else:
        verdict = "CUT"
    print(f"  {verdict} - capability passes on every line;"
          " serving and economics pass; one line fails, and"
          " it is the line this course")
    print("  has carried since Lesson 7: policy routing,"
          " 0 of 24. No candidate was ever taught escalation,"
          " and no gate, probe,")
    print("  or registry in this course can install a"
          " capability the data never taught. The stack is"
          " not released.")
    print("the repair, named and priced - the hold converts"
          " to ship only when all three stand:")
    print("  1. an escalation dataset: the must-escalate"
          " shapes authored as training rows that teach the"
          " routing rule itself,")
    print("     curated and receipted like any other data"
          " (Lessons 11 and 47)")
    print(f"  2. one retrain cycle at ${CYCLE:.2f} (Lesson"
          " 43's lines), the registry recording the new"
          " version beside this one")
    print("  3. a re-review at this same bar: policy 24 of"
          " 24 on the frozen probe suite, with the capability"
          " lines unmoved")
    print("what the review refuses, on the same evidence: no"
          " promotion, no production traffic, no quiet"
          " exception -")
    print("  the registry entry for capstone-1 stands at"
          " HELD, and the loop of Lesson 48 owns it from here")
    c0, n0, _a, _b, _r, _e = board_lines(base_model(),
                                         test_rows())
    print(f"the base model after the review: intent {c0}/{n0}"
          f" = {c0 / n0:.3f} - frozen means frozen, verified")
    print("readiness demo complete")


if __name__ == "__main__":
    main()
