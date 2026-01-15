"""
ניסוי 2 – התקדמות Attack Success Rate (ASR) לאורך האימון.

גרף מצגת:
- כולל את ערך הביניים 0.0275 (boundary case)
- מציג 100 epochs מלאים
- כולל קו סף הצלחה (ASR=0.9)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# --- נתיבי פרויקט ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = PROJECT_ROOT / "results" / "csv"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- poisoning_rate מייצגים (כולל boundary case) ---
SELECTED_RATES = [0.025, 0.0275, 0.03]

# --- הגדרות תצוגה ---
X_MAX = 100         # מספר epochs להצגה
ASR_SUCCESS = 0.90  # סף "התקפה הצליחה"


def main():
    csv_files = sorted(CSV_DIR.glob("exp02_*.csv"))
    if not csv_files:
        raise RuntimeError("לא נמצאו קובצי CSV של ניסוי 2 תחת results/csv")

    plt.figure(figsize=(9, 6))

    plotted_any = False

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        rate = float(df["poisoning_rate"].iloc[0])

        if rate not in SELECTED_RATES:
            continue

        df = df.sort_values("epoch")

        plt.plot(
            df["epoch"],
            df["test_asr"],
            marker="o",
            markersize=3,
            linewidth=2,
            label=f"poisoning_rate={rate}"
        )
        plotted_any = True

    if not plotted_any:
        raise RuntimeError("לא נמצאו קבצים עבור הערכים שנבחרו")

    # קו סף הצלחה
    plt.axhline(ASR_SUCCESS, linestyle="--", linewidth=2)
    plt.text(1, ASR_SUCCESS + 0.02, "ASR = 0.9 (attack success threshold)")

    # עיצוב
    plt.xlabel("Epoch")
    plt.ylabel("ASR (Attack Success Rate)")
    plt.title("Experiment 2: ASR vs Epoch (selected poisoning rates)")
    plt.grid(True)
    plt.xlim(0, X_MAX)
    plt.ylim(0, 1.05)

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=True)

    output_path = FIG_DIR / "exp02_asr_vs_epoch_presentable.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
