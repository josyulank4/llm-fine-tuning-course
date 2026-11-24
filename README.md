# LLM Fine-Tuning in Production — Companion Code (Udemy Course 12)

Working, stdlib-only Python demos for every hands-on lesson of the course.
No paid services, no third-party packages: every mechanic runs offline so
students see what the production tools actually compute. One shared module,
`ftcommon.py`, carries the deterministic corpus — 480 synthetic support
tickets for a fictional company, Helios Home Appliances, across 8 intents,
split 288 train / 96 dev / 96 test by a fixed pattern (the test split is
frozen from Lesson 1). The repo's models are small trainable centroid
classifiers: full fine-tuning, low-rank (LoRA-style) and quantized
(QLoRA-style) variants are real computations at toy scale, so every number
in the course is measured by running this code.

## Run the full validation
```
python validate.py
```
Expected final line: `OVERALL: PASS`.

## Packages (one per DEMO lesson — added as each lesson is produced)
| Package | Lesson | What it shows |
|---|---|---|
| lesson1 | 1 | The frozen dataset, its planted defects, and the base model's gap (foundation stub) |
| basegap | 7 | The base model scored per task — intent, format, policy — the gap, in numbers |
| datacurate | 11 | The raw training set cleaned end to end, every removal printed with its defect |
| fullrun | 15 | A full fine-tune run and read: the trained model against the base gap |
| lorarank | 19 | LoRA ranks swept: accuracy, trainable parameters, artifact size on one table |
| qloramath | 23 | Full FT vs LoRA vs QLoRA: memory, trainable share, and score kept |
| runledger | 27 | Every training run logged with config, data version, and outcome |
| evalgate | 31 | All candidates scored through one fixed gate; pass / hold verdicts printed |
| probecheck | 35 | The tuned candidates probed for failure modes the task metrics miss |
| servemerge | 39 | The winning adapter merged and served; a bad deploy rolled back in the drill |
| costledger | 43 | The lifecycle ledger: train, serve, retrain vs the prompted alternative |
| tunestack | 50/51 | The capstone stack: build end to end, then the production readiness review |

## Lesson tags
Each DEMO lesson is tagged in this repo: `lXX-start` (code before the lesson's
build) and `lXX-finish` (the exact code the lesson ships). Tags are added as
each DEMO lesson is produced. The foundation carries the Lesson 1 stub tags
`l01-start` (scaffold) and `l01-finish` (corpus + base-gap preview demo).
