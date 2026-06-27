"""
birdclef.blend_submissions - Strict row_id-based submission blender.

This script refuses to blend if row_id or species columns differ. That avoids
the most common silent failure mode in BirdCLEF notebook blends: positional
misalignment across submissions with matching shapes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from birdclef.config import OUTPUT_DIR, SAMPLE_SUBMISSION


def validate_submission(df: pd.DataFrame, sample_sub: pd.DataFrame, name: str) -> pd.DataFrame:
    if "row_id" not in df.columns:
        raise ValueError(f"{name} is missing row_id column")

    expected_columns = list(sample_sub.columns)
    actual_columns = list(df.columns)
    if set(actual_columns) != set(expected_columns):
        missing = sorted(set(expected_columns) - set(actual_columns))
        extra = sorted(set(actual_columns) - set(expected_columns))
        raise ValueError(f"{name} column mismatch | missing={missing} | extra={extra}")

    df = df[expected_columns].copy()
    duplicate_mask = df["row_id"].duplicated()
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, "row_id"].tolist()[:5]
        raise ValueError(f"{name} has duplicate row_id values, e.g. {duplicates}")

    expected_row_ids = list(sample_sub["row_id"])
    if set(df["row_id"]) != set(expected_row_ids):
        raise ValueError(f"{name} row_id set does not match sample_submission.csv")

    return df.set_index("row_id").loc[expected_row_ids].reset_index()


def blend_submissions(
    primary_path: Path,
    secondary_path: Path,
    output_path: Path,
    primary_weight: float = 0.7,
    secondary_weight: float = 0.3,
) -> pd.DataFrame:
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION)
    primary = validate_submission(pd.read_csv(primary_path), sample_sub, "primary")
    secondary = validate_submission(pd.read_csv(secondary_path), sample_sub, "secondary")

    class_columns = [col for col in sample_sub.columns if col != "row_id"]
    blended = primary.copy()
    blended[class_columns] = (
        primary_weight * primary[class_columns].astype(float)
        + secondary_weight * secondary[class_columns].astype(float)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    blended.to_csv(output_path, index=False)
    print(
        f"Blended submission saved: {output_path} | shape={blended.shape} | "
        f"weights=({primary_weight:.3f}, {secondary_weight:.3f})"
    )
    return blended


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend two BirdCLEF submission CSVs by row_id.")
    parser.add_argument("--primary", type=str, required=True)
    parser.add_argument("--secondary", type=str, required=True)
    parser.add_argument("--primary-weight", type=float, default=0.7)
    parser.add_argument("--secondary-weight", type=float, default=0.3)
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR / "submission.csv"))
    args = parser.parse_args()

    total_weight = args.primary_weight + args.secondary_weight
    if total_weight <= 0:
        raise ValueError("Blend weights must sum to a positive value")

    blend_submissions(
        primary_path=Path(args.primary),
        secondary_path=Path(args.secondary),
        output_path=Path(args.output),
        primary_weight=args.primary_weight / total_weight,
        secondary_weight=args.secondary_weight / total_weight,
    )


if __name__ == "__main__":
    main()