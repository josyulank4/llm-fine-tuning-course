"""Lesson 1 - The frozen dataset and the base model's gap, in numbers.

Prints the corpus splits, the planted raw-data defects the curation lesson
will remove, and the base model scored on the frozen test split against a
model trained on the full training split — the gap Lesson 7 (basegap)
later reads per task. Run: python -m lesson1.basegap_preview
"""
from __future__ import annotations

from ftcommon import (CORPUS, INTENTS, RAW_DEFECTS, CentroidModel, accuracy,
                      base_model, split_examples)


def main() -> None:
    print(f"examples: {len(CORPUS)}")
    for split in ("train", "dev", "test"):
        print(f"{split}: {len(split_examples(split))}")
    print(f"intents: {len(INTENTS)}")
    defects = sum(RAW_DEFECTS.values())
    print(f"raw training defects planted: {defects} "
          f"(duplicates {RAW_DEFECTS['duplicates']}, "
          f"mislabeled {RAW_DEFECTS['mislabeled']}, "
          f"stale policy {RAW_DEFECTS['stale_policy']})")
    test = split_examples("test")
    base_acc = accuracy(base_model(), test)
    full = CentroidModel().train(split_examples("train"))
    full_acc = accuracy(full, test)
    print(f"base model test accuracy: {base_acc:.3f}")
    print(f"fully trained test accuracy: {full_acc:.3f}")
    print(f"the base gap: {full_acc - base_acc:.3f}")
    print("lesson1 demo complete")


if __name__ == "__main__":
    main()
