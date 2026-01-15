"""
ניסוי 2 – השפעת אחוז ההרעלה (poisoning_rate) על ביצועי מודל BadNets.

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
TRIGGER_SIZE = 5
DEVICE = "cpu"

parser = argparse.ArgumentParser(description="Experiment 2: Poisoning rate threshold analysis")
parser.add_argument(
    "--poisoning_rate",
    type=float,
    required=True,
    help="Portion of poisoned samples (e.g. 0.01, 0.02, 0.03, 0.05)"
)
args = parser.parse_args()
POISONING_RATE = args.poisoning_rate


def build_command():
    pr_tag = str(POISONING_RATE).replace(".", "p")
    run_name = f"exp02_pr{pr_tag}"

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
    """
    main.py שומר CSV ל-logs בשם:
    {params}__{run_name}.csv
    כאן אנחנו מעתיקים אותו ל-results/csv בשם שמתחיל ב-exp02_ כדי שיהיה קל לניתוח.
    """
    logs_dir = PROJECT_ROOT / "logs"
    results_dir = PROJECT_ROOT / "results" / "csv"
    results_dir.mkdir(parents=True, exist_ok=True)

    pr_tag = str(POISONING_RATE).replace(".", "p")

    # מחפשים את ה-CSV של הריצה הספציפית לפי run_name
    pattern = f"*__{run_name}.csv"
    matches = list(logs_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find log csv in {logs_dir} matching: {pattern}")

    # אם יש כמה (לא אמור), ניקח את האחרון לפי זמן שינוי
    src = max(matches, key=lambda p: p.stat().st_mtime)

    dst = results_dir / f"exp02_{DATASET}_trigger{TRIGGER_LABEL}_pr{pr_tag}_ep{EPOCHS}_ts{TRIGGER_SIZE}.csv"
    shutil.copy2(src, dst)
    return src, dst


def run_experiment():
    cmd, run_name = build_command()
    start = time.time()

    returncode = subprocess.call(cmd, cwd=str(PROJECT_ROOT))
    runtime = time.time() - start

    return returncode, runtime, run_name


def main():
    print(f"[exp02] running | poisoning_rate={POISONING_RATE}, epochs={EPOCHS}, trigger_size={TRIGGER_SIZE}")
    code, runtime, run_name = run_experiment()
    print(f"[exp02] finished | returncode={code}, runtime_sec={runtime:.2f}")

    if code == 0:
        src, dst = copy_latest_csv_to_results(run_name)
        print(f"[exp02] copied log csv:\n  from: {src}\n  to:   {dst}")


if __name__ == "__main__":
    main()
