"""
ניסוי 1 – השפעת אחוז ההרעלה (poisoning_rate) על ביצועי מודל BadNets.

קוד זה מהווה תשתית ניסויית שנכתבה במסגרת פרויקט גמר.
הקוד מתבסס על מימוש פתוח של BadNets:
"BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain"
(Gu et al., 2017).

המימוש המקורי של המודל, הדאטה והאימון נמצא בריפו:
https://github.com/verazuo/badnets-pytorch

האחריות על מימוש האלגוריתם והמודל היא של מחברי הקוד המקורי.
קובץ זה אחראי אך ורק על הרצת ניסויים, תיעוד תוצאות וניתוחן.
"""

# imports
import subprocess
import sys
import time
from pathlib import Path

# --- וידוא שהקוד ירוץ ללא שגיאות ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = PROJECT_ROOT / "main.py"

# --- פרמטרים קבועים בניסוי ---
DATASET = "MNIST"
EPOCHS = 100
TRIGGER_LABEL = 1
TRIGGER_SIZE = 5
DEVICE = "cpu"

# --- הפרמטר המשתנה בניסוי ---
POISONING_RATE = 0.10

# בניית פקודת ההרצה של main.py עם הפרמטרים של הניסוי
def build_command():
    cmd = [
        sys.executable, str(MAIN_PY),
        "--dataset", DATASET,
        "--epochs", str(EPOCHS),
        "--poisoning_rate", str(POISONING_RATE),
        "--trigger_label", str(TRIGGER_LABEL),
        "--trigger_size", str(TRIGGER_SIZE),
        "--device", DEVICE,
    ]
    return cmd

# הרצת main.py בצורה אינטראקטיבית (זהה להרצה ידנית בטרמינל)
def run_experiment():
    cmd = build_command()
    start = time.time()

    # הרצה רגילה – כל הפלט מודפס ישירות לטרמינל (כולל tqdm וכל print)
    returncode = subprocess.call(cmd, cwd=str(PROJECT_ROOT))

    runtime = time.time() - start
    return returncode, runtime

def main():
    print(f"[exp01] running main.py with poisoning_rate={POISONING_RATE}, epochs={EPOCHS}")
    code, runtime = run_experiment()
    print(f"[exp01] finished with returncode={code}, runtime_sec={runtime:.2f}")

if __name__ == "__main__":
    main()
