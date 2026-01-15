"""
Experiment 3 – Visualizations:
- Final ASR vs trigger_size
- Final TCA vs trigger_size
- ASR vs epoch for trigger_size=4 and trigger_size=5

Reads CSV files from:
results/csv/
and expects files that start with:
exp03_

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

TS_FOR_EPOCH_PLOT = [4, 5]


def _find_exp03_csvs():
    files = sorted(CSV_DIR.glob("exp03_*.csv"))
    if not files:
        raise RuntimeError(f"No exp03_*.csv files found under: {CSV_DIR}")
    return files


def _load_all(files):
    dfs = []
    for p in files:
        df = pd.read_csv(p)

        required = {"epoch", "test_asr", "test_clean_acc", "trigger_size"}
        if not required.issubset(df.columns):
            raise RuntimeError(f"Missing required columns in {p.name}. Need at least:{required}")

        df = df.copy()
        df["__file__"] = p.name

        # Force numeric + validate
        df["trigger_size"] = pd.to_numeric(df["trigger_size"], errors="coerce")
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        if df["trigger_size"].isna().any() or df["epoch"].isna().any():
            raise RuntimeError(f"Non-numeric trigger_size/epoch found in  {p.name}")

        df["trigger_size"] = df["trigger_size"].round().astype(int)
        df["epoch"] = df["epoch"].round().astype(int)

        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # Extra safety: enforce clean dtypes across concatenated dataframe
    all_df["trigger_size"] = pd.to_numeric(all_df["trigger_size"], errors="coerce").round().astype(int)
    all_df["epoch"] = pd.to_numeric(all_df["epoch"], errors="coerce").round().astype(int)

    sizes = sorted(all_df["trigger_size"].unique().tolist())
    print(f"[exp03] trigger_size values found: {sizes}")
    return all_df


def plot_final_metrics(all_df):
    last = (
        all_df.sort_values("epoch")
        .groupby(["__file__", "trigger_size"], as_index=False)
        .tail(1)
    )

    summary = (
        last.groupby("trigger_size", as_index=False)
        .agg(
            asr=("test_asr", "mean"),
            tca=("test_clean_acc", "mean"),
            n_runs=("__file__", "count"),
        )
        .sort_values("trigger_size")
    )

    out_csv = FIG_DIR / "exp03_summary_trigger_size.csv"
    summary.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved summary table: {out_csv}")
    print(summary.to_string(index=False))

    # ASR vs trigger_size
    plt.figure(figsize=(9, 5))
    plt.plot(summary["trigger_size"], summary["asr"], marker="o", linewidth=2)
    plt.ylim(0, 1.02)
    plt.title("Experiment 3: Final ASR vs trigger_size")
    plt.xlabel("trigger_size (pixels)")
    plt.ylabel("ASR (Attack Success Rate)")
    plt.grid(True, alpha=0.3)
    out1 = FIG_DIR / "exp03_asr_vs_trigger_size.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    plt.close()
    print(f"Saved: {out1}")

    # TCA vs trigger_size
    plt.figure(figsize=(9, 5))
    plt.plot(summary["trigger_size"], summary["tca"], marker="o", linewidth=2)
    plt.ylim(0, 1.02)
    plt.title("Experiment 3: Final TCA vs trigger_size")
    plt.xlabel("trigger_size (pixels)")
    plt.ylabel("TCA (Test Clean Accuracy)")
    plt.grid(True, alpha=0.3)
    out2 = FIG_DIR / "exp03_tca_vs_trigger_size.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    plt.close()
    print(f"Saved: {out2}")


def plot_asr_vs_epoch_selected_sizes(all_df):
    """
    Plot ASR vs epoch for each trigger_size in TS_FOR_EPOCH_PLOT on the same figure.
    Uses mean ASR across runs (and across files) per epoch.
    """
    plt.figure(figsize=(10, 6))

    plotted_any = False

    for ts in TS_FOR_EPOCH_PLOT:
        sub = all_df[all_df["trigger_size"] == ts].sort_values("epoch")

        # Helpful debug print (so you can confirm both exist)
        print(f"[exp03] rows for trigger_size={ts}: {len(sub)}")

        if sub.empty:
            continue

        curve = (
            sub.groupby("epoch", as_index=False)
            .agg(asr=("test_asr", "mean"))
            .sort_values("epoch")
        )

        plt.plot(
            curve["epoch"],
            curve["asr"],
            marker="o",
            markersize=3,
            linewidth=2,
            label=f"trigger_size={ts}",
        )
        plotted_any = True

    if not plotted_any:
        raise RuntimeError(f"No data found for trigger_size in {TS_FOR_EPOCH_PLOT}")

    plt.axhline(ASR_THRESHOLD, linestyle="--", linewidth=2, label=f"ASR={ASR_THRESHOLD} threshold")

    plt.ylim(0, 1.02)
    plt.title(f"Experiment 3: ASR vs Epoch (trigger_size={TS_FOR_EPOCH_PLOT})")
    plt.xlabel("Epoch")
    plt.ylabel("ASR (Attack Success Rate)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out = FIG_DIR / "exp03_asr_vs_epoch.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")


def main():
    files = _find_exp03_csvs()
    all_df = _load_all(files)

    plot_final_metrics(all_df)
    plot_asr_vs_epoch_selected_sizes(all_df)


if __name__ == "__main__":
    main()
