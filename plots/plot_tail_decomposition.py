"""
Figure 3 — Tail risk decomposition

Reads Scenario_Tail_Decomp from one or more result .xlsx files and plots
average factor contributions in the worst 5% of simulated outcomes.

Usage:
  python plots/plot_tail_decomposition.py --files results/rolling_markowitz_results.xlsx results/cvar_max_return_results.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_SHEET = "Scenario_Tail_Decomp"
DEFAULT_OUTDIR = "results/figures"

NAVY = "#0B1F3B"
BURGUNDY = "#7A1E2B"
BENCH_GRAY = "#444444"

# Consistent factor naming for presentation
FACTOR_LABELS: Dict[str, str] = {
    "RISK_ON": "Equity Market Risk",
    "USD": "US Dollar Factor",
    "RATES": "Rate Duration Risk",
    "INFLATION": "Inflation Surprise",
    "CREDIT": "Credit Spread Risk",
}


def _label_from_filename(p: Path) -> str:
    name = p.stem.lower()
    if "markowitz" in name or "sharpe" in name:
        return "Max Sharpe (excess) – Rolling"
    if "cvar" in name:
        return "Max Return s.t. CVaR cap – Rolling"
    if "target_vol" in name:
        return "Target Vol – Rolling"
    if "utility" in name:
        return "Mean–Variance Utility – Rolling"
    return p.stem


def _color_from_label(label: str) -> str:
    l = label.lower()
    if "sharpe" in l or "markowitz" in l:
        return NAVY
    if "cvar" in l:
        return BURGUNDY
    if "60/40" in l or "sp500" in l or "s&p" in l:
        return BENCH_GRAY
    return "black"


def read_tail_decomp(path: Path, sheet: str) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(str(path))

    df = pd.read_excel(path, sheet_name=sheet, index_col=0)

    if "mean_contrib_in_tail" not in df.columns:
        raise ValueError(
            f"{path.name}: sheet '{sheet}' missing column 'mean_contrib_in_tail'"
        )

    s = df["mean_contrib_in_tail"].astype(float)
    s.name = _label_from_filename(path)
    return s


def plot_tail_decomposition(
    data: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
) -> None:
    # Sort factors by average contribution (most negative first)
    order = data.mean(axis=1).sort_values().index
    data = data.loc[order]

    labels = [FACTOR_LABELS.get(k, k) for k in data.index]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))

    n_port = data.shape[1]
    height = 0.7
    bar_h = height / max(n_port, 1)
    offsets = (np.arange(n_port) - (n_port - 1) / 2) * bar_h

    for j, col in enumerate(data.columns):
        ax.barh(
            y + offsets[j],
            data[col].values,
            height=bar_h,
            label=col,
            color=_color_from_label(col),
        )

    ax.axvline(0.0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Contribution to monthly portfolio return")

    # Reverse x-axis so losses visually extend to the right
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmax, xmin)

    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Paths to .xlsx result files",
    )
    ap.add_argument(
        "--sheet",
        default=DEFAULT_SHEET,
        help=f"Sheet name (default: {DEFAULT_SHEET})",
    )
    ap.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help=f"Output directory (default: {DEFAULT_OUTDIR})",
    )
    args = ap.parse_args()

    paths = [Path(f) for f in args.files]
    outdir = Path(args.outdir)

    print("\nReading tail decomposition:")
    series: List[pd.Series] = []
    for p in paths:
        s = read_tail_decomp(p, args.sheet)
        series.append(s)
        print(f"  - {p.name}")

    df = pd.concat(series, axis=1)

    out_png = outdir / "figure3_tail_decomposition.png"
    out_pdf = outdir / "figure3_tail_decomposition.pdf"

    plot_tail_decomposition(df, out_png, out_pdf)

    print(f"\nSaved:\n  {out_png}\n  {out_pdf}\n")


if __name__ == "__main__":
    main()
#python plots/plot_tail_decomposition.py --files results/rolling_markowitz_results1.xlsx results/cvar_max_return_results1.xlsx --outdir results/figures
