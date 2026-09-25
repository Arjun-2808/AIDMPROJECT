# IE4483 Mini Project (Option 2) — Dogs vs. Cats

**Deadline: Friday 13 Nov 2026, 11:59pm.** Submit report + `submission.csv` via
NTULearn. Group of three; the report must state each member's contribution.

Source PDF: `IE4483-Project2.pdf` (not committed — course material).

## Dataset
25,000 train (10k/class), 5,000 val (2.5k/class), 500 unlabelled test.
Using all of it is optional; the brief suggests ≥2000 train / ≥500 val is enough.

## Deliverable
`submission.csv` — two columns, image id and predicted label, **1 = dog, 0 = cat**.

## Report questions and marks

| Q | Marks | What it asks | Owner |
|---|---|---|---|
| a | 10% | Data amounts used, pre-processing, augmentation | TBD |
| b | 20% | Model(s): architecture figure, in/out dims, loss, training strategy, code + run instructions, reproducibility | TBD |
| c | 10% | How parameters (lr etc.) were chosen and why | TBD |
| d | 20% | Validation accuracy + `submission.csv` on the test set | TBD |
| e | 10% | Analyse 1–2 correct and incorrect test predictions; model strengths/weaknesses | TBD |
| f | 10% | How model choice and data processing affect val accuracy | TBD |
| g | 10% | Extend to **CIFAR-10** multi-class; describe changes vs dogs-vs-cats | TBD |
| h | 10% | Handle **class imbalance** in CIFAR-10; justify ≥2 approaches | TBD |

Fill in the Owner column once the split is agreed.

## Notes on the split

Questions a–f come almost free if all three of us train on the same harness and
log to the same leaderboard: (c) and (f) are literally the comparison table.

**(g) and (h) are 20% and nobody gets them for free** — CIFAR-10 is a separate
dataset, 32×32, 10 classes, and (h) needs a deliberately imbalanced training
split plus at least two remedies (class-weighted loss, oversampling, focal loss,
augmentation of rare classes). Worth assigning explicitly rather than leaving to
the end.

`src/optim.py` already ships `focal` loss and class-weight support for (h), and
`src/models/vit.py` has a from-scratch `tiny_vit` sized for 32×32 inputs.
