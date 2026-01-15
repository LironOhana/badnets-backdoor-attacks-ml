"""
Experiment 4 (runner) – run all trigger_pos values one by one and save outputs.

Runs the following trigger positions:
br, bl, tr, tl, center

For each run it saves:
- a log file to: results/logs/
- a CSV file to: results/csv/   (copied from ./logs created by main.py, based on run_name)

Output is shown live in the terminal (tqdm works) using a PTY (macOS/Linux).
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import shutil

import pty
import select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = PROJECT_ROOT / "main.py"

DATASET = "MNIST"
EPOCHS = 100
DEVICE = "cpu"

POISONING_RATE = 0.05
TRIGGER_SIZE = 5
TRIGGER_LABEL = 1

POSITIONS = ["br", "bl", "tr", "tl", "center"]


def build_command(trigger_pos: str):
    run_name = f"exp04_pos{trigger_pos}"
    cmd = [
        sys.executable,
        "-u",
        str(MAIN_PY),
        "--dataset", DATASET,
        "--epochs", str(EPOCHS),
        "--poisoning_rate", str(POISONING_RATE),
        "--trigger_label", str(TRIGGER_LABEL),
        "--trigger_size", str(TRIGGER_SIZE),
        "--trigger_pos", trigger_pos,
        "--run_name", run_name,
        "--device", DEVICE,
    ]
    return cmd, run_name


def run_with_pty(cmd, log_path: Path):
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
    main.py saves CSV logs into ./logs.
    We look for the newest CSV file that contains run_name in its filename.
    """

def find_csv_for_run(run_name: str) -> Path | None:
    logs_dir = PROJECT_ROOT / "logs"
    if not logs_dir.exists():
        return None

    candidates = list(logs_dir.glob(f"*{run_name}*.csv"))
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def copy_csv_to_results(csv_path: Path, trigger_pos: str, run_name: str) -> Path:
    out_dir = PROJECT_ROOT / "results" / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    # דוגמה: exp04_MNIST_posbr_pr0p05_ts5_ep100.csv
    pr_str = str(POISONING_RATE).replace(".", "p")
    out_name = f"exp04_{DATASET}_pos{trigger_pos}_pr{pr_str}_ts{TRIGGER_SIZE}_ep{EPOCHS}.csv"
    out_path = out_dir / out_name

    shutil.copy2(csv_path, out_path)
    return out_path


def main():
    log_dir = PROJECT_ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp04-all] running positions={POSITIONS}")
    print(f"[exp04-all] params: pr={POISONING_RATE}, ts={TRIGGER_SIZE}, ep={EPOCHS}, device={DEVICE}\n")

    results = []

    for pos in POSITIONS:
        print("=" * 80)
        print(f"[exp04] START trigger_pos={pos}")

        cmd, run_name = build_command(pos)
        log_path = log_dir / f"exp04_{DATASET}_pos{pos}_pr{POISONING_RATE}_ts{TRIGGER_SIZE}_ep{EPOCHS}.log"

        code, runtime = run_with_pty(cmd, log_path)

        saved_csv = None
        csv_from_main = find_csv_for_run(run_name)
        if csv_from_main is not None:
            saved_csv = copy_csv_to_results(csv_from_main, pos, run_name)

        print(f"\n[exp04] END trigger_pos={pos} | returncode={code} | runtime_sec={runtime:.2f}")
        print(f"[exp04] log saved: {log_path}")
        if saved_csv:
            print(f"[exp04] csv saved: {saved_csv}\n")
        else:
            print("[exp04] WARNING: לא נמצא CSV בתיקיית ./logs (אולי main.py לא שומר, או שאין run_name בשם הקובץ)\n")

        results.append((pos, code, runtime, str(log_path), str(saved_csv) if saved_csv else None))

        if code != 0:
            print("[exp04-all] WARNING: one run failed; continuing to next position.\n")

    print("=" * 80)
    print("[exp04-all] SUMMARY")
    for pos, code, runtime, lp, cp in results:
        print(f"- pos={pos:6s} | code={code} | runtime_sec={runtime:.2f} | log={lp} | csv={cp}")


if __name__ == "__main__":
    main()
