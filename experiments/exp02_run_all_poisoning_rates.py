"""
Experiment 2 (runner) – run all poisoning_rate values sequentially and save outputs.

This script sweeps over poisoning_rate values (data poisoning percentage).
For each run it saves:
- a log file to: results/logs/
- a CSV file to: results/csv/  (copied from ./logs created by main.py, based on run_name)

Notes:
- main.py writes training CSV logs into ./logs.
- We look for a CSV that ends with "__{run_name}.csv".
- If multiple files match, we copy the newest one.
"""

import argparse
import os
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# This file is under experiments/, so parents[1] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = PROJECT_ROOT / "main.py"

# Fixed experiment settings (change here if needed)
DATASET = "MNIST"
EPOCHS = 100
TRIGGER_LABEL = 1
TRIGGER_SIZE = 5

# Performance-related settings (kept constant)
BATCH_SIZE = 512
NUM_WORKERS = 4

# Default values for the sweep
DEFAULT_RATES = [0.01, 0.02, 0.025, 0.0275, 0.03, 0.05]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for this experiment wrapper."""
    parser = argparse.ArgumentParser(
        description="Experiment 2 – run all poisoning_rate values sequentially"
    )
    parser.add_argument(
        "--rates",
        type=float,
        nargs="*",
        default=DEFAULT_RATES,
        help="List of poisoning rates to run (e.g. --rates 0.01 0.02 0.025 0.03 0.05)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use: cpu | cuda | cuda:0 | mps (default: cpu). "
             "In Colab, use --device cuda to run on GPU.",
    )
    return parser.parse_args()


def _rate_tag(rate: float) -> str:
    """Convert a float rate to a filename-friendly tag, e.g., 0.0275 -> 0p0275."""
    return str(rate).replace(".", "p")


def build_command(poisoning_rate: float, device: str) -> tuple[list[str], str]:
    """Build the command that invokes main.py for a specific poisoning_rate."""
    pr_tag = _rate_tag(poisoning_rate)
    run_name = f"exp02_pr{pr_tag}"

    cmd = [
        sys.executable,
        "-u",
        str(MAIN_PY),
        "--dataset", DATASET,
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--num_workers", str(NUM_WORKERS),
        "--poisoning_rate", str(poisoning_rate),
        "--trigger_label", str(TRIGGER_LABEL),
        "--trigger_size", str(TRIGGER_SIZE),
        "--device", device,
        "--run_name", run_name,
    ]
    return cmd, run_name


def run_with_pty(cmd: list[str], log_path: Path) -> tuple[int, float]:
    """
    Run a command while streaming stdout/stderr both to the notebook/terminal and to a log file.
    Uses a pseudo-terminal to preserve tqdm-like progress bars.
    """
    import pty  # lazy import

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    start = time.time()

    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        env=env,
        close_fds=True,
    )
    os.close(slave_fd)

    with open(log_path, "w", encoding="utf-8") as f:
        while True:
            r, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd in r:
                data = os.read(master_fd, 4096)
                if not data:
                    break
                text = data.decode(errors="replace")
                sys.stdout.write(text)
                sys.stdout.flush()
                f.write(text)
                f.flush()

            if proc.poll() is not None:
                time.sleep(0.2)
                try:
                    rest = os.read(master_fd, 4096)
                    if rest:
                        text = rest.decode(errors="replace")
                        sys.stdout.write(text)
                        sys.stdout.flush()
                        f.write(text)
                        f.flush()
                except OSError:
                    pass
                break

    os.close(master_fd)

    runtime = time.time() - start
    return proc.returncode, runtime


def find_csv_for_run(run_name: str) -> Optional[Path]:
    """
    main.py writes training CSV logs into ./logs.
    We look for the CSV that ends with '__{run_name}.csv'.
    If multiple files match, we take the newest one.
    """
    logs_dir = PROJECT_ROOT / "logs"
    pattern = f"*__{run_name}.csv"
    matches = list(logs_dir.glob(pattern))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def copy_csv_to_results(csv_path: Path, poisoning_rate: float) -> Path:
    """Copy the CSV produced by main.py into results/csv/ with a consistent filename."""
    out_dir = PROJECT_ROOT / "results" / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    pr_tag = _rate_tag(poisoning_rate)
    out_name = f"exp02_{DATASET}_trigger{TRIGGER_LABEL}_pr{pr_tag}_ep{EPOCHS}_ts{TRIGGER_SIZE}.csv"
    out_path = out_dir / out_name

    shutil.copy2(csv_path, out_path)
    return out_path


def main() -> None:
    args = parse_args()
    rates = args.rates
    device = args.device

    log_dir = PROJECT_ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp02-all] running rates={rates}")
    print(
        f"[exp02-all] params: dataset={DATASET}, ep={EPOCHS}, "
        f"batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS}, "
        f"ts={TRIGGER_SIZE}, trigger_label={TRIGGER_LABEL}, device={device}\n"
    )

    results = []

    for pr in rates:
        print("=" * 80)
        print(f"[exp02] START poisoning_rate={pr}")

        cmd, run_name = build_command(pr, device=device)

        pr_tag = _rate_tag(pr)
        log_path = log_dir / f"exp02_{DATASET}_pr{pr_tag}_ts{TRIGGER_SIZE}_ep{EPOCHS}.log"

        code, runtime = run_with_pty(cmd, log_path)

        saved_csv = None
        csv_from_main = find_csv_for_run(run_name)
        if csv_from_main is not None:
            saved_csv = copy_csv_to_results(csv_from_main, pr)

        print(f"\n[exp02] END poisoning_rate={pr} | code={code} | runtime_sec={runtime:.2f}")
        if saved_csv:
            print(f"[exp02] saved csv -> {saved_csv}")
        else:
            print("[exp02] WARNING: could not find CSV produced by main.py under ./logs")

        results.append((pr, code, runtime, str(log_path), str(saved_csv) if saved_csv else None))

        if code != 0:
            print("[exp02-all] WARNING: one run failed; continuing to next rate.\n")

    print("=" * 80)
    print("[exp02-all] SUMMARY")
    for pr, code, runtime, lp, cp in results:
        print(f"- pr={pr:<7} | code={code} | runtime_sec={runtime:.2f} | log={lp} | csv={cp}")


if __name__ == "__main__":
    main()
