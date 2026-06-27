# BirdCLEF 2026 Archive

This repository is a historical archive of the BirdCLEF 2026 competition work that grew out of the Julia Crop Caretaker codebase and earlier audio work from Project Aria.

The purpose of this archive is not to continue active development here. The purpose is to preserve:

- what was built
- why it was built that way
- what competition constraints shaped the design
- what technical lessons were learned

This makes the repo useful as an engineering record instead of just a dead code dump.

## What This Project Tried To Do

BirdCLEF 2026 focused on wildlife species detection from short audio windows in Pantanal soundscapes. The working approach here adapted an existing audio classification stack into a multilabel bioacoustics pipeline that could run under Kaggle competition constraints.

Core goals:

- classify species from 5-second windows
- support local training and Kaggle inference
- preserve a fast iteration path for model and threshold experiments
- explore CFAR-inspired adaptive thresholding as a signal-processing idea applied to bioacoustics

## Why The Design Looks Like This

The project architecture reflects several practical constraints:

- Kaggle inference had to run offline and on CPU.
- BirdCLEF is multilabel, so the pipeline had to move away from single-label genre-style assumptions.
- Rare-species ranking mattered, so raw probability preservation and threshold discipline mattered more than human-friendly hard filtering.
- Boundary effects in soundscape windows were a real risk, which pushed the design toward overlap-aware inference.

That is why you see a mix of:

- reusable audio feature extraction
- multiple candidate backbones
- local training scripts
- Kaggle notebooks and submission helpers
- threshold sweep tooling
- backlog documents describing architectural corrections and follow-up work

## Archive Contents

Top-level items worth reading first:

- `birdclef/` — main module source code
- `birdclef/README.md` — original project README from the competition effort
- `birdclef_colab_cfar.ipynb` — Colab-oriented workflow notebook
- `k_sweep_results.json` — threshold sweep artifact kept for historical context
- `k_sweep_figure.png` — visualization artifact kept for historical context
- `birdclef_todo.py` — backlog and corrective notes from the original effort
- `LESSONS_LEARNED.md` — retrospective summary for future reuse

## What Is Intentionally Not Included

This archive excludes heavyweight or generated artifacts that do not belong in normal git history:

- trained model weights such as `.pt` and `.pth`
- generated submission outputs
- caches and `__pycache__`
- local environment clutter

The goal is a repo that explains the work and preserves the implementation surface without turning into object storage.

## Recommended Reading Order

1. `LESSONS_LEARNED.md`
2. `birdclef/README.md`
3. `birdclef/train.py`
4. `birdclef/inference.py`
5. `birdclef/evaluate_thresholds.py`
6. `birdclef_todo.py`

## Reuse Value

Even though the competition is over, this archive is still useful as a reference for:

- audio classification pipeline migration work
- offline Kaggle deployment constraints
- multilabel inference design
- threshold-tuning experiments
- translating signal-processing intuition into ML decision logic

## Status

Archived for recordkeeping and lessons learned. Not the active development home.