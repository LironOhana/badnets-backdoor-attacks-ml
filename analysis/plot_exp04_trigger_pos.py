"""
Experiment 4 – Visualizations (trigger_pos)

Reads CSV files from:
results/csv/
and expects files that start with:
exp04_

Creates:
1) ASR vs epoch for each trigger_pos (all curves in one figure)
2) Final ASR vs trigger_pos
3) Final TCA vs trigger_pos

Also saves a summary table:
results/figures/exp04_summary.csv
"""

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = PROJECT_ROOT / "results" / "csv"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

POS_RE = re.compile(r"_pos(?P<pos>br|bl|tr|tl|center)_")


POS_ORDER = ["tl", "tr", "bl", "br", "center"]


def _find_exp04_csvs():
    files = sorted(CSV_DIR.glob("exp04_*.csv"))
    if not files:
        raise RuntimeError("No exp04_*.csv files found under results/csv")
    return files


def _load_all(files):
    dfs = []
    for p in files:
        df = pd.read_csv(p)

        required = {"epoch", "test_asr", "test_clean_acc"}
        if not required.issubset(df.columns):
            raise RuntimeError(
                f"Missing required columns in {p.name}. Need at least:{required}"
            )

        m = POS_RE.search(p.name)
        pos = m.group("pos") if m else "unknown"

        df = df.copy()
        df["trigger_pos"] = pos
        df["__file__"] = p.name
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    return all_df


def plot_asr_vs_epoch(all_df):
    curve = (
        all_df.groupby(["trigger_pos", "epoch"], as_index=False)
        .agg(asr=("test_asr", "mean"))
        .sort_values(["trigger_pos", "epoch"])
    )

    plt.figure(figsize=(10, 6))

    positions = [p for p in POS_ORDER if p in curve["trigger_pos"].unique()]
    for p in sorted(set(curve["trigger_pos"].unique()) - set(positions)):
        positions.append(p)

    for pos in positions:
        part = curve[curve["trigger_pos"] == pos]
        plt.plot(part["epoch"], part["asr"], marker="o", markersize=2, label=f"pos={pos}")

    plt.ylim(0, 1.02)
    plt.title("Experiment 4: ASR vs Epoch (different trigger positions)")
    plt.xlabel("Epoch")
    plt.ylabel("ASR (Attack Success Rate)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="trigger_pos", ncol=2)
    out = FIG_DIR / "exp04_asr_vs_epoch_all_positions.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")


def plot_final_bars(all_df):
    last = (
        all_df.sort_values("epoch")
        .groupby(["__file__", "trigger_pos"], as_index=False)
        .tail(1)
    )

    summary = (
        last.groupby("trigger_pos", as_index=False)
        .agg(
            final_asr=("test_asr", "mean"),
            final_tca=("test_clean_acc", "mean"),
            n_runs=("__file__", "count"),
        )
    )

    summary["__order__"] = summary["trigger_pos"].apply(lambda x: POS_ORDER.index(x) if x in POS_ORDER else 999)
    summary = summary.sort_values("__order__").drop(columns="__order__")

    out_csv = FIG_DIR / "exp04_summary.csv"
    summary.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved summary table: {out_csv}")
    print(summary)

    # Final ASR
    plt.figure(figsize=(9, 5))
    plt.bar(summary["trigger_pos"], summary["final_asr"])
    plt.ylim(0, 1.02)
    plt.title("Experiment 4: Final ASR by trigger position")
    plt.xlabel("trigger_pos")
    plt.ylabel("Final ASR")
    plt.grid(True, axis="y", alpha=0.3)
    out1 = FIG_DIR / "exp04_final_asr_by_position.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    plt.close()
    print(f"Saved: {out1}")

    # Final TCA
    plt.figure(figsize=(9, 5))
    plt.bar(summary["trigger_pos"], summary["final_tca"])
    plt.ylim(0, 1.02)
    plt.title("Experiment 4: Final TCA by trigger position")
    plt.xlabel("trigger_pos")
    plt.ylabel("Final TCA")
    plt.grid(True, axis="y", alpha=0.3)
    out2 = FIG_DIR / "exp04_final_tca_by_position.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    plt.close()
    print(f"Saved: {out2}")


def main():
    files = _find_exp04_csvs()
    all_df = _load_all(files)

    plot_asr_vs_epoch(all_df)
    plot_final_bars(all_df)


if __name__ == "__main__":
    main()
