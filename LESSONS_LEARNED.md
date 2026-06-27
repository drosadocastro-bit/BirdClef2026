# Lessons Learned

## What We Built

The BirdCLEF 2026 effort assembled a competition-oriented bioacoustics stack with:

- audio feature extraction for spectrogram and classical features
- multiple model backbones
- a training path for local experimentation
- Kaggle-compatible inference tooling
- threshold evaluation and sweep utilities
- Colab workflow support

It was not just a single notebook. It was a small working system built to support repeated iteration under competition constraints.

## Why We Built It This Way

Several design decisions came from practical pressure rather than theory purity.

### 1. Reuse beat starting from zero

The fastest path was adapting proven audio infrastructure from earlier work instead of inventing a new stack from scratch. That reduced startup time and gave immediate leverage in feature extraction and model scaffolding.

### 2. Competition constraints shaped architecture

Kaggle runtime limits, offline execution, CPU inference, and submission formatting all mattered. That pushed the project toward:

- explicit inference scripts
- notebook handoff paths
- lightweight deployment assumptions
- careful probability handling

### 3. Signal-processing instincts were worth carrying over

The CFAR-style thresholding idea was valuable because BirdCLEF is not only a model problem. It is also a detection and ranking problem under shifting background conditions. Treating thresholding as part of the engineering problem was the right instinct.

## What Worked

- Reusing an existing audio stack accelerated progress.
- Keeping both script and notebook paths made iteration easier.
- Preserving threshold sweep artifacts made later reasoning easier.
- Documenting backlog items in code-adjacent files captured design corrections before they were forgotten.

## What Became Clear Midstream

### Window-boundary handling was more important than it first looked

Hard 5-second cuts risked missing calls at segment boundaries. Overlap-aware windowing was not an optimization detail; it was part of the correctness story.

### Raw probability preservation mattered

For ROC-AUC-driven evaluation, aggressive early thresholding can destroy ranking signal. Human-readable filtering and leaderboard scoring are not the same problem.

### Class imbalance could not be treated as an afterthought

Rare species matter disproportionately in macro-style evaluation settings. Loss weighting and probability handling were architectural concerns, not cleanup tasks.

## What We Would Do Differently Now

1. Lock down archive hygiene from day one.
   Keep weights, outputs, caches, and data outside normal git history from the start.

2. Separate active code from experiment residue earlier.
   Too many notebook variants make historical repos harder to read unless they are explicitly labeled.

3. Maintain a short running working note during development.
   Competition projects move fast and memory decays fast. A live design log reduces retrospective guesswork.

4. Make inference correctness checks first-class.
   Probability semantics, overlap behavior, and submission formatting deserve explicit tests early.

## What Is Worth Reusing Elsewhere

- the migration pattern from one domain-specific audio problem to another
- the split between training, inference, and evaluation utilities
- the threshold-sweep workflow
- the discipline of writing down corrective backlog items while still in-flight

## Bottom Line

This project is worth preserving because it shows an honest engineering path:

- adapt quickly
- ship usable structure early
- identify scoring and inference pitfalls
- refine the pipeline as the real constraints become visible

That is exactly the kind of record that becomes valuable later.