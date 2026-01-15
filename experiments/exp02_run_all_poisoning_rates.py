"""
Experiment 2 (runner) – run all poisoning_rate values one by one and save outputs.

This script runs a sweep over poisoning_rate values (data poisoning percentage).
For each run it saves:
- a log file to: results/logs/
- a CSV file to: results/csv/   (copied from ./logs created by main.py, based on run_name)

"""


import os
import sys
import time
import shutil
import select
import subprocess
import argparse
from pathlib import Path


# Note: this file is under experiments/, so parents[1] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = PROJECT_ROOT / "main.py"

# Fixed experiment settings (change here if needed)
DATASET = "MNIST"
EPOCHS = 100
TRIGGER_LABEL = 1
TRIGGER_SIZE = 5
DEVICE = "cpu"

# Default values for the sweep
DEFAULT_RATES = [0.01, 0.02, 0.025, 0.0275, 0.03, 0.05]


def build_command(poisoning_rate: float):
    pr_tag = str(poisoning_rate).replace(".", "p")
    run_name = f"exp02_pr{pr_tag}"

    cmd = [
        sys.executable,
        "-u",
        str(MAIN_PY),
        "--dataset", DATASET,
        "--epochs", str(EPOCHS),
        "--poisoning_rate", str(poisoning_rate),
        "--trigger_label", str(TRIGGER_LABEL),
        "--trigger_size", str(TRIGGER_SIZE),
        "--device", DEVICE,
        "--run_name", run_name,
    ]
    return cmd, run_name


def run_with_pty(cmd, log_path: Path):
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


    """
    main.py writes training CSV logs into ./logs.
    We look for the CSV that ends with '__{run_name}.csv'.
    If multiple files match, we take the newest one.
    """
def find_csv_for_run(run_name: str) -> Path | None:
    logs_dir = PROJECT_ROOT / "logs"
    pattern = f"*__{run_name}.csv"
    matches = list(logs_dir.glob(pattern))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def copy_csv_to_results(csv_path: Path, poisoning_rate: float, run_name: str) -> Path:
    out_dir = PROJECT_ROOT / "results" / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    pr_tag = str(poisoning_rate).replace(".", "p")
    out_name = f"exp02_{DATASET}_trigger{TRIGGER_LABEL}_pr{pr_tag}_ep{EPOCHS}_ts{TRIGGER_SIZE}.csv"
    out_path = out_dir / out_name

    shutil.copy2(csv_path, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Experiment 2 – run all poisoning_rate values sequentially")
    parser.add_argument(
        "--rates",
        type=float,
        nargs="*",
        default=DEFAULT_RATES,
        help="List of poisoning rates to run (e.g. --rates 0.01 0.02 0.025 0.03 0.05)",
    )
    args = parser.parse_args()

    rates = args.rates

    log_dir = PROJECT_ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp02-all] running rates={rates}")
    print(f"[exp02-all] params: dataset={DATASET}, ep={EPOCHS}, ts={TRIGGER_SIZE}, trigger_label={TRIGGER_LABEL}, device={DEVICE}\n")

    results = []

    for pr in rates:
        print("=" * 80)
        print(f"[exp02] START poisoning_rate={pr}")

        cmd, run_name = build_command(pr)
        log_path = log_dir / f"exp02_{DATASET}_pr{pr}_ts{TRIGGER_SIZE}_ep{EPOCHS}.log"

        code, runtime = run_with_pty(cmd, log_path)

        saved_csv = None
        csv_from_main = find_csv_for_run(run_name)
        if csv_from_main is not None:
            saved_csv = copy_csv_to_results(csv_from_main, pr, run_name)

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
