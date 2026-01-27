
"""
Figure — Rolling factor betas over time

Reads "Rolling_Betas" from one or more result .xlsx files (produced by the runners)
and plots the rolling betas.

Typical usage:
  python plots/plot_rolling_betas.py --files results/rolling_markowitz_results1.xlsx results/cvar_max_return_results1.xlsx
  python plots/plot_rolling_betas.py --files rolling_markowitz_results1.xlsx --outdir results/figures
  python plots/plot_rolling_betas.py --files results/rolling_markowitz_results1.xlsx --factors RISK_ON RATES CREDIT

Notes
-----
- Expects sheet name: "Rolling_Betas"
- Columns typically include: const (or Intercept) + factor columns
- This script drops the intercept by default and plots only factor betas.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:
    raise SystemExit(
        "matplotlib is not installed. Install it with:\n"
        "  pip install matplotlib\n"
        "or (recommended) install your full requirements:\n"
        "  pip install -r requirements.txt\n"
    ) from e


ROLLING_BETAS_SHEET = "Rolling_Betas"
DEFAULT_OUTDIR = Path("results/figures")

NAVY = "#0B1F3B"
BURGUNDY = "#7A1E2B"
BENCH_GRAY = "#444444"


def _label_from_filename(p: Path) -> str:
    name = p.stem.lower()
    if "markowitz" in name or "sharpe" in name:
        return "Max Sharpe (excess)"
    if "cvar" in name:
        return "Max Return s.t. CVaR cap"
    if "target_vol" in name:
        return "Target Vol"
    if "utility" in name:
        return "Mean–Variance Utility"
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


def _read_rolling_betas(xlsx_path: Path) -> pd.DataFrame:
    if not xlsx_path.exists():
        raise FileNotFoundError(str(xlsx_path))

    df = pd.read_excel(xlsx_path, sheet_name=ROLLING_BETAS_SHEET)

    # Try to interpret first column as date index if it looks like dates
    if df.shape[1] >= 2:
        first = df.columns[0]
        maybe_dt = pd.to_datetime(df[first], errors="coerce")
        if maybe_dt.notna().mean() > 0.8:  # mostly parseable
            df = df.set_index(first)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

    # Coerce numeric columns
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(how="all")
    return df


def _pick_factor_columns(df: pd.DataFrame, factors: Optional[List[str]]) -> List[str]:
    # Common intercept column names to drop
    drop = {"const", "intercept", "Intercept", "alpha", "r2", "R2", "adj_r2", "adjR2"}

    cols = [c for c in df.columns if c not in drop]

    if factors:
        # keep in user order, only those present
        kept = [f for f in factors if f in df.columns]
        missing = [f for f in factors if f not in df.columns]
        if missing:
            raise ValueError(f"Requested factors not found in Rolling_Betas: {missing}. Available: {list(df.columns)}")
        return kept

    # otherwise: all non-intercept-ish columns
    return cols


def plot_rolling_betas(
    betas_by_model: List[pd.DataFrame],
    labels: List[str],
    factor_cols: List[str],
    out_png: Path,
    out_pdf: Path,
    ylims: Optional[List[float]] = None,
) -> None:
    # One figure per factor (cleanest, report-friendly)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    for factor in factor_cols:
        fig, ax = plt.subplots(figsize=(10, 5.5))

        for df, lab in zip(betas_by_model, labels):
            if factor not in df.columns:
                continue
            ax.plot(df.index, df[factor], label=lab, linewidth=2.0, color=_color_from_label(lab))

        ax.set_xlabel("Date")
        ax.set_ylabel("Beta")
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

        if ylims is not None and len(ylims) == 2:
            ax.set_ylim(float(ylims[0]), float(ylims[1]))

        fig.tight_layout()

        # save per-factor
        safe = factor.lower().replace(" ", "_")
        fig.savefig(out_png.with_name(f"{out_png.stem}_{safe}.png"), dpi=200)
        fig.savefig(out_pdf.with_name(f"{out_pdf.stem}_{safe}.pdf"))
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True, help="Paths to .xlsx result files")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output directory for figures")
    ap.add_argument(
        "--factors",
        nargs="*",
        default=None,
        help="Optional subset of factor columns to plot (e.g. RISK_ON RATES CREDIT INFLATION USD)",
    )
    ap.add_argument(
        "--ylims",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Optional y-axis limits (e.g. --ylims -1.5 1.5)",
    )
    args = ap.parse_args()

    paths = [Path(p) for p in args.files]
    outdir = Path(args.outdir)

    betas_by_model: List[pd.DataFrame] = []
    labels: List[str] = []

    # Read all
    for p in paths:
        df = _read_rolling_betas(p)
        betas_by_model.append(df)
        labels.append(_label_from_filename(p))
        print(f"Read {p.name}: rows={len(df):,}, cols={len(df.columns)}")

    # Determine which factors to plot:
    # Use intersection across models (unless user explicitly specified)
    if args.factors:
        factor_cols = args.factors
    else:
        common = set(_pick_factor_columns(betas_by_model[0], None))
        for df in betas_by_model[1:]:
            common &= set(_pick_factor_columns(df, None))
        # Keep a nice canonical order if present
        canonical = ["RISK_ON", "RATES", "CREDIT", "INFLATION", "USD"]
        factor_cols = [c for c in canonical if c in common] + sorted([c for c in common if c not in canonical])

        if not factor_cols:
            raise ValueError("No common factor columns found across files. Pass --factors explicitly.")

    out_png = outdir / "figure_rolling_betas"
    out_pdf = outdir / "figure_rolling_betas"

    plot_rolling_betas(
        betas_by_model=betas_by_model,
        labels=labels,
        factor_cols=factor_cols,
        out_png=out_png,
        out_pdf=out_pdf,
        ylims=args.ylims,
    )

    print(f"\nSaved rolling beta figures to: {outdir.resolve()}")
    for f in factor_cols:
        safe = f.lower().replace(' ', '_')
        print(f"  - {out_png.name}_{safe}.png / .pdf")


if __name__ == "__main__":
    main()

#python plots/plot_rolling_betas.py --files results/rolling_markowitz_results1.xlsx results/cvar_max_return_results1.xlsx --outdir results/figures