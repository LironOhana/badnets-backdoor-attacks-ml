"""
Experiment 3 (runner) – run all trigger_size values sequentially and save outputs.

This script sweeps over trigger_size values (trigger size in pixels).
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = PROJECT_ROOT / "main.py"

# Fixed experiment settings (adjust here if needed)
DATASET = "MNIST"
EPOCHS = 100
TRIGGER_LABEL = 1

# Keep poisoning_rate above the success threshold so we can isolate the effect of trigger_size
POISONING_RATE = 0.05

# Performance-related settings (kept constant)
BATCH_SIZE = 64          # recommended for stable/consistent behavior vs the "original" setup
NUM_WORKERS = 4          # speeds up data loading in Colab

# Default sizes used in the experiment
DEFAULT_SIZES = [1, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 3 – run all trigger_size values sequentially"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=DEFAULT_SIZES,
        help="List of trigger sizes to run (e.g. --sizes 1 3 4 5)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use: cpu | cuda | cuda:0 | mps (default: cpu). In Colab use --device cuda.",
    )
    return parser.parse_args()


def _pr_tag(rate: float) -> str:
    return str(rate).replace(".", "p")


def build_command(trigger_size: int, device: str) -> tuple[list[str], str]:
    ts_tag = str(trigger_size)
    run_name = f"exp03_ts{ts_tag}"

    cmd = [
        sys.executable,
        "-u",
        str(MAIN_PY),
        "--dataset", DATASET,
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--num_workers", str(NUM_WORKERS),
        "--poisoning_rate", str(POISONING_RATE),
        "--trigger_label", str(TRIGGER_LABEL),
        "--trigger_size", str(trigger_size),
        "--device", device,
        "--run_name", run_name,
    ]
    return cmd, run_name


def run_with_pty(cmd: list[str], log_path: Path) -> tuple[int, float]:
    """
    Run a command while streaming stdout/stderr both to the notebook/terminal and to a log file.
    Uses a pseudo-terminal to preserve tqdm-like progress bars.

    Colab note:
    Sometimes PTY reads can raise OSError(Errno 5) when the child process closes the TTY.
    We treat that as end-of-stream and exit gracefully.
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
                try:
                    data = os.read(master_fd, 4096)
                except OSError:
                    # Colab PTY can throw Errno 5 when the process ends.
                    break

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


def copy_csv_to_results(csv_path: Path, trigger_size: int) -> Path:
    out_dir = PROJECT_ROOT / "results" / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    pr_tag = _pr_tag(POISONING_RATE)
    out_name = f"exp03_{DATASET}_trigger{TRIGGER_LABEL}_pr{pr_tag}_ep{EPOCHS}_ts{trigger_size}.csv"
    out_path = out_dir / out_name

    shutil.copy2(csv_path, out_path)
    return out_path


def main() -> None:
    args = parse_args()
    sizes = args.sizes
    device = args.device

    log_dir = PROJECT_ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp03-all] running sizes={sizes}")
    print(
        f"[exp03-all] params: dataset={DATASET}, pr={POISONING_RATE}, ep={EPOCHS}, "
        f"batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS}, trigger_label={TRIGGER_LABEL}, device={device}\n"
    )

    results = []

    for ts in sizes:
        print("=" * 80)
        print(f"[exp03] START trigger_size={ts}")

        cmd, run_name = build_command(ts, device=device)

        pr_tag = _pr_tag(POISONING_RATE)
        log_path = log_dir / f"exp03_{DATASET}_ts{ts}_pr{pr_tag}_ep{EPOCHS}.log"

        code, runtime = run_with_pty(cmd, log_path)

        saved_csv = None
        csv_from_main = find_csv_for_run(run_name)
        if csv_from_main is not None:
            saved_csv = copy_csv_to_results(csv_from_main, ts)

        print(f"\n[exp03] END trigger_size={ts} | code={code} | runtime_sec={runtime:.2f}")
        if saved_csv:
            print(f"[exp03] saved csv -> {saved_csv}")
        else:
            print("[exp03] WARNING: could not find CSV produced by main.py under ./logs")

        results.append((ts, code, runtime, str(log_path), str(saved_csv) if saved_csv else None))

        if code != 0:
            print("[exp03-all] WARNING: one run failed; continuing to next size.\n")

    print("=" * 80)
    print("[exp03-all] SUMMARY")
    for ts, code, runtime, lp, cp in results:
        print(f"- ts={ts:<3} | code={code} | runtime_sec={runtime:.2f} | log={lp} | csv={cp}")


if __name__ == "__main__":
    main()
