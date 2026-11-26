"""Lesson 50 - Build the full fine-tuning stack (SCAFFOLD).

The capstone build, executed under Lesson 49's contract: one
frozen dataset, every candidate method, every claim
reproduced from a run. This scaffold states the contract and
reads the base gap the stack must close; the build itself -
curate, train three ways, gate, probe, merge, serve, price -
is the lesson's work, and this file raises until it is done.
Run: python -m lesson50.tunestack
"""
from __future__ import annotations

from ftcommon import base_model, split_examples
from lesson31.evalgate import board_lines, test_rows


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


def main() -> None:
    train = split_examples("train")
    print_terms(train)
    rows = test_rows()
    c, n, n_ok, n_tot, _r, esc_n = board_lines(base_model(),
                                               rows)
    print("the gap the stack must close - the base model on"
          " the frozen exam:")
    print(f"  intent: {c}/{n} = {c / n:.3f}")
    print(f"  new phrasings: {n_ok}/{n_tot} = {n_ok / n_tot:.3f}")
    print(f"  policy: 0/{esc_n} = 0.000")
    raise NotImplementedError(
        "the stack is not built in the l50-start scaffold")


if __name__ == "__main__":
    main()
