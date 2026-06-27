"""
birdclef.perch_only_submission - Offline-friendly Perch submission branch.

Uses a runtime-compatible TensorFlow SavedModel to generate a full
BirdCLEF submission CSV keyed by row_id. This script is intentionally
independent from any secondary model so blending can be debugged later.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from birdclef.config import COMPETITION_DIR, OUTPUT_DIR, SAMPLE_RATE, WINDOW_SECONDS


DEFAULT_PERCH_MODEL_DIR = Path(
    "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/bird-vocalization-classifier/8"
)


def parse_row_id(row_id: str) -> tuple[str, int]:
    stem, end_time = row_id.rsplit("_", 1)
    return stem, int(end_time)


def find_audio_file(soundscape_dir: Path, file_stem: str) -> Path | None:
    for ext in (".ogg", ".wav", ".flac"):
        candidate = soundscape_dir / f"{file_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def build_species_mapping(model_dir: Path, taxonomy_csv: Path) -> tuple[list[str], dict[str, int]]:
    taxonomy_df = pd.read_csv(taxonomy_csv)
    competition_species = taxonomy_df["primary_label"].astype(str).tolist()

    label_file = model_dir / "assets" / "labels.csv"
    if not label_file.exists():
        assets_dir = model_dir / "assets"
        csvs = sorted(assets_dir.glob("*.csv")) if assets_dir.exists() else []
        if not csvs:
            raise FileNotFoundError(f"No label CSV found in {assets_dir}")
        label_file = csvs[0]

    label_df = pd.read_csv(label_file)
    ebird_col = label_df.columns[0]
    for col in ("ebird2021", "ebird_code", "species_code"):
        if col in label_df.columns:
            ebird_col = col
            break

    perch_labels = label_df[ebird_col].astype(str).tolist()
    perch_to_idx = {species: idx for idx, species in enumerate(perch_labels)}

    mapped = sum(1 for species in competition_species if species in perch_to_idx)
    print(
        f"Perch label file: {label_file.name} | total classes: {len(perch_labels)} | "
        f"mapped competition species: {mapped}/{len(competition_species)}"
    )
    return competition_species, perch_to_idx


def build_competition_index(species: list[str], perch_to_idx: dict[str, int]) -> np.ndarray:
    """Map BirdCLEF species order to Perch class indices, using -1 for zero-shot labels."""
    return np.array([perch_to_idx.get(species_name, -1) for species_name in species], dtype=np.int32)


def run_perch_submission(
    model_dir: Path = DEFAULT_PERCH_MODEL_DIR,
    competition_dir: Path = COMPETITION_DIR,
    output_path: Path | None = None,
    batch_size: int = 16,
) -> pd.DataFrame:
    import librosa
    import tensorflow as tf

    sample_submission_path = competition_dir / "sample_submission.csv"
    taxonomy_csv = competition_dir / "taxonomy.csv"
    soundscape_dir = competition_dir / "test_soundscapes"

    if output_path is None:
        output_path = OUTPUT_DIR / "perch_submission.csv"

    sample_sub = pd.read_csv(sample_submission_path)
    species, perch_to_idx = build_species_mapping(model_dir, taxonomy_csv)
    competition_index = build_competition_index(species, perch_to_idx)
    mapped_mask = competition_index >= 0
    zero_shot_prior = 1.0 / max(len(species), 1)

    print(f"Loading Perch SavedModel from: {model_dir}")
    perch = tf.saved_model.load(str(model_dir))
    infer_fn = perch.signatures["serving_default"]

    test_wav = tf.constant(np.zeros((1, int(SAMPLE_RATE * WINDOW_SECONDS)), dtype=np.float32))
    test_out = infer_fn(inputs=test_wav)
    if "label" not in test_out:
        raise RuntimeError(f"Expected 'label' output, got {list(test_out.keys())}")
    print(
        f"Perch smoke test OK | output classes: {int(test_out['label'].shape[-1])} | "
        f"keys: {list(test_out.keys())}"
    )

    file_windows: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row_id in sample_sub["row_id"]:
        stem, end_time = parse_row_id(row_id)
        file_windows[stem].append((row_id, end_time))

    row_frames: list[pd.DataFrame] = []
    window_samples = int(SAMPLE_RATE * WINDOW_SECONDS)
    t_start = time.time()

    for file_idx, file_stem in enumerate(sorted(file_windows.keys()), start=1):
        audio_path = find_audio_file(soundscape_dir, file_stem)
        if audio_path is None:
            print(f"[{file_idx}/{len(file_windows)}] Missing audio for {file_stem}, using prior")
            row_ids = [row_id for row_id, _ in file_windows[file_stem]]
            prior_block = np.full((len(row_ids), len(species)), zero_shot_prior, dtype=np.float32)
            frame = pd.DataFrame(prior_block, columns=species)
            frame.insert(0, "row_id", row_ids)
            row_frames.append(frame)
            continue

        audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        print(f"[{file_idx}/{len(file_windows)}] {audio_path.name}")

        chunks: list[np.ndarray] = []
        row_ids: list[str] = []
        for row_id, end_time in file_windows[file_stem]:
            start_sample = max(0, int((end_time - WINDOW_SECONDS) * SAMPLE_RATE))
            end_sample = int(end_time * SAMPLE_RATE)
            chunk = audio[start_sample:end_sample]
            if len(chunk) < window_samples:
                chunk = np.pad(chunk, (0, window_samples - len(chunk)))
            elif len(chunk) > window_samples:
                chunk = chunk[:window_samples]

            chunks.append(chunk.astype(np.float32, copy=False))
            row_ids.append(row_id)

        projected_batches: list[np.ndarray] = []
        for start_idx in range(0, len(chunks), batch_size):
            batch_waveforms = np.stack(chunks[start_idx:start_idx + batch_size], axis=0)
            output = infer_fn(inputs=tf.constant(batch_waveforms, dtype=tf.float32))
            logits = output["label"].numpy()
            probs = 1.0 / (1.0 + np.exp(-logits, dtype=np.float32))

            projected = np.full((probs.shape[0], len(species)), zero_shot_prior, dtype=np.float32)
            projected[:, mapped_mask] = probs[:, competition_index[mapped_mask]]
            projected_batches.append(projected)

        frame = pd.DataFrame(np.vstack(projected_batches), columns=species)
        frame.insert(0, "row_id", row_ids)
        row_frames.append(frame)

    submission = pd.concat(row_frames, ignore_index=True)
    submission = submission.set_index("row_id").reindex(sample_sub["row_id"]).reset_index()
    submission.columns = sample_sub.columns
    submission = submission.fillna(zero_shot_prior)

    if set(submission["row_id"]) != set(sample_sub["row_id"]):
        raise RuntimeError("Perch submission row_id mismatch with sample_submission.csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    elapsed = time.time() - t_start
    print(f"Perch submission saved: {output_path} | shape={submission.shape} | time={elapsed:.1f}s")
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a BirdCLEF submission from a Perch SavedModel.")
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_PERCH_MODEL_DIR))
    parser.add_argument("--competition-dir", type=str, default=str(COMPETITION_DIR))
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR / "perch_submission.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    run_perch_submission(
        model_dir=Path(args.model_dir),
        competition_dir=Path(args.competition_dir),
        output_path=Path(args.output),
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()