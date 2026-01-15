from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = PROJECT_ROOT / "results" / "csv"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def parse_poisoning_rate_from_filename(name: str) -> float | None:
    # תומך בשמות כמו: exp02_MNIST_trigger1_pr0p03_ep100_ts5.csv
    m = re.search(r"_pr(\d+p\d+)_", name)
    if not m:
        return None
    return float(m.group(1).replace("p", "."))


def load_last_epoch_metrics(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)

    # ננסה לקחת poisoning_rate מתוך העמודה (אם קיימת), אחרת מהשם של הקובץ
    pr = None
    if "poisoning_rate" in df.columns:
        try:
            pr = float(df["poisoning_rate"].iloc[-1])
        except Exception:
            pr = None
    if pr is None:
        pr = parse_poisoning_rate_from_filename(csv_path.name)

    if pr is None:
        raise ValueError(f"Could not determine poisoning_rate for file: {csv_path.name}")

    # שורה אחרונה (epoch מקסימלי)
    if "epoch" in df.columns:
        last = df.loc[df["epoch"].idxmax()]
    else:
        last = df.iloc[-1]

    # השמות כפי שנוצרים אצלנו ב-main.py
    tca_col = "test_clean_acc"
    asr_col = "test_asr"

    if tca_col not in df.columns or asr_col not in df.columns:
        raise ValueError(
            f"Missing required columns in {csv_path.name}. "
            f"Expected columns: {tca_col}, {asr_col}. "
            f"Found: {list(df.columns)}"
        )

    return {
        "poisoning_rate": pr,
        "tca": float(last[tca_col]),
        "asr": float(last[asr_col]),
        "file": csv_path.name,
    }


def main():
    csv_files = sorted(CSV_DIR.glob("exp02_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No exp02 CSV files found in: {CSV_DIR}")

    rows = [load_last_epoch_metrics(p) for p in csv_files]
    res = pd.DataFrame(rows).sort_values("poisoning_rate")

    # --- גרף 1: ASR מול poisoning_rate ---
    plt.figure()
    plt.plot(res["poisoning_rate"], res["asr"], marker="o")
    plt.xlabel("poisoning_rate")
    plt.ylabel("ASR (Attack Success Rate)")
    plt.title("Experiment 2: ASR vs poisoning_rate")
    plt.ylim(0, 1.05)
    asr_path = FIG_DIR / "exp02_asr_vs_poisoning_rate.png"
    plt.savefig(asr_path, dpi=200, bbox_inches="tight")
    plt.close()

    # --- גרף 2: TCA מול poisoning_rate ---
    plt.figure()
    plt.plot(res["poisoning_rate"], res["tca"], marker="o")
    plt.xlabel("poisoning_rate")
    plt.ylabel("TCA (Test Clean Accuracy)")
    plt.title("Experiment 2: TCA vs poisoning_rate")
    plt.ylim(0, 1.05)
    tca_path = FIG_DIR / "exp02_tca_vs_poisoning_rate.png"
    plt.savefig(tca_path, dpi=200, bbox_inches="tight")
    plt.close()

    # שמירת טבלת סיכום (נוח למצגת/README)
    summary_path = FIG_DIR / "exp02_summary.csv"
    res.to_csv(summary_path, index=False, encoding="utf-8")

    print("Saved figures:")
    print(f"- {asr_path}")
    print(f"- {tca_path}")
    print("Saved summary table:")
    print(f"- {summary_path}")
    print("\nSummary:")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()

