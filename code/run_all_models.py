#!/usr/bin/env python3
"""Run every Geographic MIL/SIL model on the bundled synthetic data."""

import argparse
import subprocess
import sys
from pathlib import Path

from geographic_mil_utils import default_data_path


MODEL_SCRIPTS = [
    "MIL_Max.py",
    "MIL_Mean.py",
    "MIL_Attention.py",
    "MIL_DeepSets.py",
    "SIL_MajorityVote.py",
    "SIL_ProbabilityAverage.py",
]


def main():
    parser = argparse.ArgumentParser(description="Run all Geographic MIL models.")
    parser.add_argument("--data-path", default=str(default_data_path()))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--encoder", choices=["mock", "bert"], default="mock")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-buildings", type=int, default=40)
    parser.add_argument("--max-len", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra argument passed through to each model script.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    for script in MODEL_SCRIPTS:
        cmd = [
            sys.executable,
            str(root / script),
            "--data-path",
            args.data_path,
            "--output-dir",
            args.output_dir,
            "--encoder",
            args.encoder,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--max-buildings",
            str(args.max_buildings),
            "--max-len",
            str(args.max_len),
            "--device",
            args.device,
        ] + args.extra_arg
        print("\n" + "=" * 80)
        print("running:", " ".join(cmd))
        subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
