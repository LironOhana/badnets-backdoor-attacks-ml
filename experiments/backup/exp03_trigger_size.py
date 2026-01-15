"""
ניסוי 3 – השפעת גודל ה-trigger (trigger_size) על ביצועי מודל BadNets.

קוד זה מהווה תשתית ניסויית שנכתבה במסגרת פרויקט גמר.
הקוד מתבסס על מימוש פתוח של BadNets:
"BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain"
(Gu et al., 2017).

המימוש המקורי של המודל, הדאטה והאימון נמצא בריפו:
https://github.com/verazuo/badnets-pytorch

האחריות על מימוש האלגוריתם והמודל היא של מחברי הקוד המקורי.
קובץ זה אחראי אך ורק על הרצת ניסויים, תיעוד תוצאות וניתוחן.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = PROJECT_ROOT / "main.py"

DATASET = "MNIST"
EPOCHS = 100
TRIGGER_LABEL = 1
DEVICE = "cpu"

# נשארים רחוק מה-threshold כדי לבודד את השפעת trigger_size
POISONING_RATE = 0.05


parser = argparse.ArgumentParser(description="Experiment 3: Trigger size sensitivity")
parser.add_argument(
    "--trigger_size",
    type=int,
    required=True,
    help="Trigger size in pixels (e.g. 1, 3, 5, 7, 9)"
)
args = parser.parse_args()
TRIGGER_SIZE = args.trigger_size


def build_command():
    ts_tag = str(TRIGGER_SIZE)
    run_name = f"exp03_ts{ts_tag}"

    cmd = [
        sys.executable,
        str(MAIN_PY),
        "--dataset", DATASET,
        "--epochs", str(EPOCHS),
        "--poisoning_rate", str(POISONING_RATE),
        "--trigger_label", str(TRIGGER_LABEL),
        "--trigger_size", str(TRIGGER_SIZE),
        "--device", DEVICE,
        "--run_name", run_name,
    ]
    return cmd, run_name


def copy_latest_csv_to_results(run_name: str):
    logs_dir = PROJECT_ROOT / "logs"
    results_dir = PROJECT_ROOT / "results" / "csv"
    results_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"*__{run_name}.csv"
    matches = list(logs_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find log csv in {logs_dir} matching: {pattern}")

    src = max(matches, key=lambda p: p.stat().st_mtime)

    pr_tag = str(POISONING_RATE).replace(".", "p")
    dst = results_dir / f"exp03_{DATASET}_trigger{TRIGGER_LABEL}_pr{pr_tag}_ep{EPOCHS}_ts{TRIGGER_SIZE}.csv"
    shutil.copy2(src, dst)
    return src, dst


def run_experiment():
    cmd, run_name = build_command()
    start = time.time()
    returncode = subprocess.call(cmd, cwd=str(PROJECT_ROOT))
    runtime = time.time() - start
    return returncode, runtime, run_name


def main():
    print(
        f"[exp03] running | trigger_size={TRIGGER_SIZE}, "
        f"poisoning_rate={POISONING_RATE}, epochs={EPOCHS}"
    )

    code, runtime, run_name = run_experiment()

    print(f"[exp03] finished | returncode={code}, runtime_sec={runtime:.2f}")

    if code == 0:
        src, dst = copy_latest_csv_to_results(run_name)
        print(f"[exp03] copied log csv:\n  from: {src}\n  to:   {dst}")


if __name__ == "__main__":
    main()

