# plots/plot_rolling_weights.py
"""
Figure — Rolling Portfolio Weights (stacked area)

Reads rolling weights from one or more result .xlsx files (produced by the runners),
and saves a stacked area chart per strategy.

Usage examples:
  python plots/plot_rolling_weights.py --files results/rolling_markowitz_results1.xlsx results/cvar_max_return_results1.xlsx
  python plots/plot_rolling_weights.py --files rolling_markowitz_results1.xlsx --start 2012-01-01 --end 2024-12-31

Notes:
- Expects the main weights sheet name to be "Weights". If your export uses a different name,
  the script will try a few fallbacks.
- Weights must be wide format: index = dates, columns = assets, values = weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


WEIGHTS_SHEET_CANDIDATES = [
    "Weights",
    "weights",
    "Rolling_Weights",
    "Portfolio_Weights",
]

# Consistent colors for asset classes
ASSET_COLORS = {
    "SP500": "#0B1F3B",      # NAVY (equity)
    "UST_7_10Y": "#7A1E2B",  # BURGUNDY (bonds)
    "T_BILLS": "#FFD700",    # GOLD (short-term bonds)
    "CHINA": "#004225",      # RACING GREEN (equity)
    "REITS": "#FF6347",      # TOMATO (real estate)
    "GOLD": "#DAA520",       # GOLDENROD (commodity)
    # Add more if needed
}


def _repo_root() -> Path:
    # plots/plot_rolling_weights.py -> repo root is parents[1]
    # repo/
    #   plots/
    #     plot_rolling_weights.py
    return Path(__file__).resolve().parents[1]


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


def _read_weights(xlsx_path: Path, sheet: Optional[str] = None) -> tuple[pd.DataFrame, str]:
    if not xlsx_path.exists():
        raise FileNotFoundError(str(xlsx_path))

    xl = pd.ExcelFile(xlsx_path)

    used_sheet = None
    if sheet is not None:
        if sheet not in xl.sheet_names:
            raise ValueError(
                f"Sheet '{sheet}' not found in {xlsx_path.name}. Available: {xl.sheet_names}"
            )
        used_sheet = sheet
    else:
        for cand in WEIGHTS_SHEET_CANDIDATES:
            if cand in xl.sheet_names:
                used_sheet = cand
                break
        if used_sheet is None:
            raise ValueError(
                f"Could not find a weights sheet in {xlsx_path.name}. "
                f"Tried: {WEIGHTS_SHEET_CANDIDATES}. Available: {xl.sheet_names}"
            )

    # Try reading with first column as index (dates). Fallback if needed.
    df = pd.read_excel(xlsx_path, sheet_name=used_sheet, index_col=0)

    # Ensure datetime index
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        # Sometimes Excel exports a column named "date"
        df = pd.read_excel(xlsx_path, sheet_name=used_sheet)
        if "date" not in df.columns:
            raise ValueError(f"Could not infer date index in {xlsx_path.name}/{used_sheet}.")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    # Keep only numeric columns
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(how="all")
    df = df.fillna(0.0)

    if df.empty or df.shape[1] == 0:
        raise ValueError(f"Weights table empty in {xlsx_path.name}/{used_sheet}.")

    # Sort by date
    df = df.sort_index()

    return df, used_sheet


def _limit_assets(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if top_n <= 0 or df.shape[1] <= top_n:
        return df

    avg_abs = df.abs().mean(axis=0).sort_values(ascending=False)
    keep = list(avg_abs.index[:top_n])
    out = df[keep].copy()
    dropped = [c for c in df.columns if c not in keep]
    if dropped:
        out["Other"] = df[dropped].sum(axis=1)
    return out


def plot_stacked_weights(
    W: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> None:
    X = W.copy()

    if start is not None:
        X = X.loc[pd.to_datetime(start):]
    if end is not None:
        X = X.loc[:pd.to_datetime(end)]

    if X.empty:
        raise ValueError("No data after applying start/end filters.")

    # Clip tiny numerical noise
    X = X.where(np.isfinite(X), 0.0).fillna(0.0)
    X = X.clip(lower=-1.0, upper=1.0)

    # If weights are long-only they should sum to ~1. If not, we still plot as-is.
    # But for stacked area, negative weights break the visual. Handle gracefully:
    if (X < -1e-10).any().any():
        # Plot line chart instead if there are negatives
        fig, ax = plt.subplots(figsize=(11, 6))
        for c in X.columns:
            ax.plot(X.index, X[c], linewidth=1.2, label=c, color=ASSET_COLORS.get(c, 'black'))
        ax.set_xlabel("Date")
        ax.set_ylabel("Weight")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.stackplot(X.index, [X[c].to_numpy() for c in X.columns], labels=list(X.columns), alpha=0.9, colors=[ASSET_COLORS.get(c, 'black') for c in X.columns])
        ax.set_xlabel("Date")
        ax.set_ylabel("Weight")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True, help="Paths to .xlsx result files")
    ap.add_argument("--sheet", default=None, help="Optional explicit weights sheet name")
    ap.add_argument("--outdir", default=None, help="Output directory (default: repo_root/results/figures)")
    ap.add_argument("--start", default=None, help="Optional start date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="Optional end date (YYYY-MM-DD)")
    ap.add_argument("--top-n", type=int, default=8, help="Keep top N assets by avg |weight|; rest -> 'Other'")
    args = ap.parse_args()

    repo = _repo_root()
    outdir = Path(args.outdir) if args.outdir else (repo / "results" / "figures")

    paths: List[Path] = [Path(f) for f in args.files]

    print("\nReading weights and saving figures:")
    for p in paths:
        W, used_sheet = _read_weights(p, sheet=args.sheet)
        Wp = _limit_assets(W, top_n=int(args.top_n))

        label = _label_from_filename(p)
        safe = p.stem.replace(" ", "_")

        out_png = outdir / f"figure_weights_{safe}.png"
        out_pdf = outdir / f"figure_weights_{safe}.pdf"

        plot_stacked_weights(
            W=Wp,
            out_png=out_png,
            out_pdf=out_pdf,
            start=args.start,
            end=args.end,
        )

        print(f"  - {p.name}: sheet='{used_sheet}', rows={len(W):,}, cols={W.shape[1]} -> saved:")
        print(f"      {out_png}")
        print(f"      {out_pdf}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()

#python plots/plot_rolling_weights.py --files results/rolling_markowitz_results1.xlsx results/cvar_max_return_results1.xlsx --outdir results/figures

