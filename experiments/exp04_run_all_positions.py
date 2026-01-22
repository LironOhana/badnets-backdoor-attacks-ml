"""
Experiment 4 (runner) – run all trigger_pos values sequentially and save outputs.

Runs the following trigger positions:
br, bl, tr, tl, center

For each run it saves:
- a log file to: results/logs/
- a CSV file to: results/csv/  (copied from ./logs created by main.py, based on run_name)

Output is streamed live (tqdm works) using a PTY.
Colab note: PTY reads can sometimes raise OSError(Errno 5) when the child process closes the TTY.
We treat that as end-of-stream and exit gracefully.
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

# Fixed experiment settings
DATASET = "MNIST"
EPOCHS = 100

POISONING_RATE = 0.05
TRIGGER_SIZE = 5
TRIGGER_LABEL = 1

# Performance-related settings (kept constant)
BATCH_SIZE = 64
NUM_WORKERS = 4

POSITIONS = ["br", "bl", "tr", "tl", "center"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 4 – run all trigger_pos values sequentially"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use: cpu | cuda | cuda:0 | mps (default: cpu). In Colab use --device cuda.",
    )
    return parser.parse_args()


def _pr_tag(rate: float) -> str:
    return str(rate).replace(".", "p")


def build_command(trigger_pos: str, device: str) -> tuple[list[str], str]:
    run_name = f"exp04_pos{trigger_pos}"
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
        "--trigger_size", str(TRIGGER_SIZE),
        "--trigger_pos", trigger_pos,
        "--run_name", run_name,
        "--device", device,
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
    main.py saves CSV logs into ./logs.
    We look for the newest CSV file that contains run_name in its filename.
    """
    logs_dir = PROJECT_ROOT / "logs"
    if not logs_dir.exists():
        return None

    candidates = list(logs_dir.glob(f"*{run_name}*.csv"))
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def copy_csv_to_results(csv_path: Path, trigger_pos: str) -> Path:
    out_dir = PROJECT_ROOT / "results" / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    pr_str = _pr_tag(POISONING_RATE)
    out_name = f"exp04_{DATASET}_pos{trigger_pos}_pr{pr_str}_ts{TRIGGER_SIZE}_ep{EPOCHS}.csv"
    out_path = out_dir / out_name

    shutil.copy2(csv_path, out_path)
    return out_path


def main() -> None:
    args = parse_args()
    device = args.device

    log_dir = PROJECT_ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp04-all] running positions={POSITIONS}")
    print(
        f"[exp04-all] params: pr={POISONING_RATE}, ts={TRIGGER_SIZE}, ep={EPOCHS}, "
        f"batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS}, device={device}\n"
    )

    results = []

    for pos in POSITIONS:
        print("=" * 80)
        print(f"[exp04] START trigger_pos={pos}")

        cmd, run_name = build_command(pos, device=device)

        pr_tag = _pr_tag(POISONING_RATE)
        log_path = log_dir / f"exp04_{DATASET}_pos{pos}_pr{pr_tag}_ts{TRIGGER_SIZE}_ep{EPOCHS}.log"

        code, runtime = run_with_pty(cmd, log_path)

        saved_csv = None
        csv_from_main = find_csv_for_run(run_name)
        if csv_from_main is not None:
            saved_csv = copy_csv_to_results(csv_from_main, pos)

        print(f"\n[exp04] END trigger_pos={pos} | returncode={code} | runtime_sec={runtime:.2f}")
        print(f"[exp04] log saved: {log_path}")
        if saved_csv:
            print(f"[exp04] csv saved: {saved_csv}\n")
        else:
            print("[exp04] WARNING: could not find CSV under ./logs (main.py may not have saved it, or run_name not present)\n")

        results.append((pos, code, runtime, str(log_path), str(saved_csv) if saved_csv else None))

        if code != 0:
            print("[exp04-all] WARNING: one run failed; continuing to next position.\n")

    print("=" * 80)
    print("[exp04-all] SUMMARY")
    for pos, code, runtime, lp, cp in results:
        print(f"- pos={pos:6s} | code={code} | runtime_sec={runtime:.2f} | log={lp} | csv={cp}")


if __name__ == "__main__":
    main()
