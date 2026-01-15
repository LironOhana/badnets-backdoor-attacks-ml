"""
Experiment 2 – Visualizations:
- Final ASR vs poisoning_rate
- Final TCA vs poisoning_rate
- ASR vs epoch for selected poisoning rates (around the threshold)

Reads CSV files from:
results/csv/
and expects files that start with:
exp02_

Saves figures to:
results/figures/
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = PROJECT_ROOT / "results" / "csv"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ASR_THRESHOLD = 0.9

# Choose which poisoning_rate curves to show in the ASR vs epoch plot

SELECTED_RATES_FOR_EPOCH = [0.01,0.02,0.025, 0.0275, 0.03,0.05]


def _find_exp02_csvs():
    files = sorted(CSV_DIR.glob("exp02_*.csv"))
    if not files:
        raise RuntimeError(f"No exp02_*.csv files found under: {CSV_DIR}")
    return files


def _load_all(files):
    dfs = []
    for p in files:
        df = pd.read_csv(p)

        required = {"epoch", "test_asr", "test_clean_acc", "poisoning_rate"}
        if not required.issubset(df.columns):
            raise RuntimeError(f"Missing required columns in {p.name}. Need at least: {required}")

        df = df.copy()
        df["__file__"] = p.name

     # Convert to numeric and validate
        df["poisoning_rate"] = pd.to_numeric(df["poisoning_rate"], errors="coerce")
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        if df["poisoning_rate"].isna().any() or df["epoch"].isna().any():
            raise RuntimeError(f"Non-numeric poisoning_rate/epoch found in {p.name}")

    # Cleanup: epoch as int, and round poisoning_rate to avoid float artifacts (e.g., 0.0300000004)

        df["epoch"] = df["epoch"].round().astype(int)
        df["poisoning_rate"] = df["poisoning_rate"].round(4)

        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    rates = sorted(all_df["poisoning_rate"].unique().tolist())
    print(f"[exp02] poisoning_rate values found: {rates}")
    return all_df


"""
Take the last epoch row for each (file, poisoning_rate) pair (final result per run).
"""

def _final_per_run(all_df):
    last = (
        all_df.sort_values("epoch")
        .groupby(["__file__", "poisoning_rate"], as_index=False)
        .tail(1)
    )
    return last


def plot_final_metrics(all_df):
    last = _final_per_run(all_df)

    summary = (
        last.groupby("poisoning_rate", as_index=False)
        .agg(
            asr=("test_asr", "mean"),
            tca=("test_clean_acc", "mean"),
            n_runs=("__file__", "count"),
        )
        .sort_values("poisoning_rate")
    )

    out_csv = FIG_DIR / "exp02_summary_poisoning_rate.csv"
    summary.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved summary table: {out_csv}")
    print(summary.to_string(index=False))

    # ASR vs poisoning_rate
    plt.figure(figsize=(9, 5))
    plt.plot(summary["poisoning_rate"], summary["asr"], marker="o", linewidth=2)
    plt.ylim(0, 1.02)
    plt.title("Experiment 2: ASR vs poisoning_rate")
    plt.xlabel("poisoning_rate")
    plt.ylabel("ASR (Attack Success Rate)")
    plt.grid(True, alpha=0.3)
    out1 = FIG_DIR / "exp02_asr_vs_poisoning_rate.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    plt.close()
    print(f"Saved: {out1}")

    # TCA vs poisoning_rate
    plt.figure(figsize=(9, 5))
    plt.plot(summary["poisoning_rate"], summary["tca"], marker="o", linewidth=2)
    plt.ylim(0, 1.02)
    plt.title("Experiment 2: TCA vs poisoning_rate")
    plt.xlabel("poisoning_rate")
    plt.ylabel("TCA (Test Clean Accuracy)")
    plt.grid(True, alpha=0.3)
    out2 = FIG_DIR / "exp02_tca_vs_poisoning_rate.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    plt.close()
    print(f"Saved: {out2}")


def plot_asr_vs_epoch_selected_rates(all_df):
    selected = [round(x, 4) for x in SELECTED_RATES_FOR_EPOCH]
    df = all_df[all_df["poisoning_rate"].isin(selected)].copy()

    if df.empty:
        raise RuntimeError(
            f"No data found for SELECTED_RATES_FOR_EPOCH={SELECTED_RATES_FOR_EPOCH}. "
            f"Make sure the CSV files contain these poisoning_rate values."
        )

 # Average ASR per (poisoning_rate, epoch) in case there are multiple runs per rate
    curve = (
        df.groupby(["poisoning_rate", "epoch"], as_index=False)
        .agg(asr=("test_asr", "mean"))
        .sort_values(["poisoning_rate", "epoch"])
    )

    plt.figure(figsize=(10, 6))
    for pr in sorted(curve["poisoning_rate"].unique()):
        sub = curve[curve["poisoning_rate"] == pr]
        plt.plot(sub["epoch"], sub["asr"], marker="o", markersize=3, linewidth=2,
                 label=f"poisoning_rate={pr}")

    plt.axhline(ASR_THRESHOLD, linestyle="--", linewidth=2,
                label=f"ASR={ASR_THRESHOLD} (attack success threshold)")

    plt.ylim(0, 1.02)
    plt.title("Experiment 2: ASR vs Epoch (selected poisoning rates)")
    plt.xlabel("Epoch")
    plt.ylabel("ASR (Attack Success Rate)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out = FIG_DIR / "exp02_asr_vs_epoch_selected.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")


def main():
    files = _find_exp02_csvs()
    all_df = _load_all(files)

    plot_final_metrics(all_df)
    plot_asr_vs_epoch_selected_rates(all_df)


if __name__ == "__main__":
    main()
