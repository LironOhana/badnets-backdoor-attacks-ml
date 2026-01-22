"""
Experiment 1 – Baseline run for BadNets on MNIST (fixed poisoning_rate).

This script is part of an experimental infrastructure developed for a final project.
It is based on the original BadNets attack described in:

  Gu et al., "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain" (2017)

and on an open-source reference implementation:
  https://github.com/verazuo/badnets-pytorch

Important note:
- The original algorithm/model implementation belongs to the original authors.
- This file is responsible only for orchestrating runs (calling main.py) and logging runtime.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = PROJECT_ROOT / "main.py"

# --- Fixed experiment parameters ---
DATASET = "MNIST"
EPOCHS = 100
TRIGGER_LABEL = 1
TRIGGER_SIZE = 5
BATCH_SIZE = 512
NUM_WORKERS = 4


# --- Experiment-specific parameter (fixed for baseline) ---
POISONING_RATE = 0.10


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for this experiment wrapper."""
    parser = argparse.ArgumentParser(
        description="Experiment 1 (Baseline): run main.py with fixed BadNets parameters."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use: cpu | cuda | cuda:0 | mps (default: cpu). "
             "In Colab, use --device cuda to run on GPU.",
    )
    return parser.parse_args()


def build_command(device: str) -> list[str]:
    """Build the command that invokes main.py with the baseline configuration."""
    return [
        sys.executable,
        str(MAIN_PY),
        "--dataset",
        DATASET,
        "--epochs",
        str(EPOCHS),
        "--poisoning_rate",
        str(POISONING_RATE),
        "--trigger_label",
        str(TRIGGER_LABEL),
        "--trigger_size",
        str(TRIGGER_SIZE),
        "--device",
        device,
        "--batch_size",
        str(BATCH_SIZE),
        "--num_workers",
        str(NUM_WORKERS),

    ]


def run_experiment(device: str) -> tuple[int, float]:
    """
    Run main.py interactively (stdout/stderr streamed directly), and return
    (return_code, runtime_seconds).
    """
    cmd = build_command(device)
    start = time.time()

    # Regular execution: prints/tqdm output go directly to the notebook/terminal.
    return_code = subprocess.call(cmd, cwd=str(PROJECT_ROOT))

    runtime_sec = time.time() - start
    return return_code, runtime_sec


def main() -> None:
    args = parse_args()
    print(
        f"[exp01] running main.py with poisoning_rate={POISONING_RATE}, "
        f"epochs={EPOCHS}, device={args.device}"
    )

    code, runtime = run_experiment(args.device)
    print(f"[exp01] finished with returncode={code}, runtime_sec={runtime:.2f}")


if __name__ == "__main__":
    main()
