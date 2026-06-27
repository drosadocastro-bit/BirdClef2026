"""
birdclef.second_model_submission - Wrapper around the existing CNN inference path.

Generates a submission-shaped CSV from one of the trained PyTorch backbones.
This is designed to be blended later with the Perch branch using row_id.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from birdclef.config import MODEL_DIR, OUTPUT_DIR
from birdclef.inference import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a BirdCLEF submission from an existing PyTorch backbone.")
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b2",
        choices=["small", "efficientnet_b0", "efficientnet_b2", "mobilenet_v2", "convnext_tiny", "perch"],
    )
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR / "second_model_submission.csv"))
    args = parser.parse_args()

    submission = run_inference(
        backbone=args.backbone,
        model_dir=Path(args.model_dir),
        batch_size=args.batch_size,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Second-model submission saved: {output_path} | shape={submission.shape}")


if __name__ == "__main__":
    main()