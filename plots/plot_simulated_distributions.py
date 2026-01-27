# plot_simulated_distributions.py
"""
Figure 2 — Simulated return distributions (Student-t)

Reads simulation output from one or more result .xlsx files (produced by the runners),
extracts the simulated return samples, and plots overlayed distributions.

Usage (examples):
  python plot_simulated_distributions.py --files results/rolling_markowitz_results.xlsx results/cvar_max_return_results.xlsx
  python plot_simulated_distributions.py --files rolling_markowitz_results1.xlsx cvar_max_return_results1.xlsx --outdir results/figures

Notes
-----
- Expects a sheet containing simulated returns with a column named "sim_return".
  By default your runners export: "Scenario_Sim_Returns_Sample".
- If the sheet name differs, the script will try a few common fallbacks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde
except Exception:
    gaussian_kde = None


DEFAULT_SHEETS = [
    "Scenario_Sim_Returns_Sample",
    "Scenario_Sim_Returns",
    "Scenario_Sim_Returns_50k",
    "Scenario_Sim_Returns_Cap",
]

SUMMARY_SHEET = "Scenario_Summary"

NAVY = "#0B1F3B"
BURGUNDY = "#7A1E2B"
BENCH_GRAY = "#444444"


def _read_sim_returns(xlsx_path: Path, sheet_candidates: List[str]) -> Tuple[pd.Series, str]:
    """Return (sim_returns, used_sheet)."""
    if not xlsx_path.exists():
        raise FileNotFoundError(str(xlsx_path))

    xl = pd.ExcelFile(xlsx_path)

    used_sheet = None
    for sh in sheet_candidates:
        if sh in xl.sheet_names:
            used_sheet = sh
            break

    if used_sheet is None:
        raise ValueError(
            f"Could not find a simulation sheet in {xlsx_path.name}. "
            f"Tried: {sheet_candidates}. Available: {xl.sheet_names}"
        )

    df = pd.read_excel(xlsx_path, sheet_name=used_sheet)
    if "sim_return" not in df.columns:
        raise ValueError(
            f"Sheet '{used_sheet}' in {xlsx_path.name} does not have column 'sim_return'. "
            f"Columns: {list(df.columns)}"
        )

    s = pd.to_numeric(df["sim_return"], errors="coerce").dropna()
    if s.empty:
        raise ValueError(f"No valid sim_return values found in {xlsx_path.name} / {used_sheet}")

    return s, used_sheet


def _read_diag_summary(xlsx_path: Path) -> Optional[pd.Series]:
    """Try to read Scenario_Summary and return diag_* rows as a Series for logging."""
    try:
        df = pd.read_excel(xlsx_path, sheet_name=SUMMARY_SHEET, index_col=0)
        if "value" not in df.columns:
            return None
        diag = df.loc[df.index.astype(str).str.startswith("diag_"), "value"]
        return diag if len(diag) else None
    except Exception:
        return None


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


def plot_distributions(
    series_list: List[pd.Series],
    labels: List[str],
    outpath_png: Path,
    outpath_pdf: Path,
    bins: int = 80,
    kde: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    # Convert to numpy arrays
    arrs = [s.to_numpy(dtype=float) for s in series_list]

    # Decide x-limits from pooled percentiles unless user overrides
    if xlim is None:
        pooled = np.concatenate(arrs)
        lo = float(np.quantile(pooled, 0.005))
        hi = float(np.quantile(pooled, 0.995))
        pad = 0.05 * (hi - lo) if hi > lo else 0.01
        xlim = (lo - pad, hi + pad)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram overlays (density)
    for a, lab in zip(arrs, labels):
        color = _color_from_label(lab)
        ax.hist(a, bins=bins, density=True, alpha=0.35, label=lab, color=color)

    # Optional KDE overlays
    if kde and gaussian_kde is not None:
        grid = np.linspace(xlim[0], xlim[1], 600)
        for a, lab in zip(arrs, labels):
            if len(a) >= 20:
                color = _color_from_label(lab)
                k = gaussian_kde(a)
                ax.plot(grid, k(grid), linewidth=2.0, color=color)

    ax.set_xlabel("Simulated portfolio return (monthly)")
    ax.set_ylabel("Density")
    ax.set_xlim(xlim[0], xlim[1])
    ax.axvline(0.0, linewidth=1.0, linestyle="--")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)

    outpath_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath_png, dpi=200)
    fig.savefig(outpath_pdf)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Paths to .xlsx result files (e.g., rolling_markowitz_results.xlsx cvar_max_return_results.xlsx)",
    )
    ap.add_argument("--outdir", default="results/figures", help="Output directory for figures")
    ap.add_argument("--bins", type=int, default=80, help="Histogram bins")
    ap.add_argument("--no-kde", action="store_true", help="Disable KDE overlay (SciPy required for KDE)")
    ap.add_argument(
        "--sheet",
        default=None,
        help="Optional explicit sheet name for sim returns (otherwise tries common defaults)",
    )
    ap.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Optional x-axis limits (e.g. --xlim -0.15 0.12)",
    )
    args = ap.parse_args()

    paths = [Path(f) for f in args.files]
    outdir = Path(args.outdir)

    sheet_candidates = [args.sheet] if args.sheet else DEFAULT_SHEETS

    sims: List[pd.Series] = []
    labels: List[str] = []

    print("\nReading simulation samples:")
    for p in paths:
        s, used_sheet = _read_sim_returns(p, sheet_candidates)
        sims.append(s)
        labels.append(_label_from_filename(p))
        print(f"  - {p.name}: n={len(s):,}  sheet='{used_sheet}'")

        diag = _read_diag_summary(p)
        if diag is not None:
            # Keep the print short: only the most relevant diagnostics
            keys = [k for k in diag.index if k in {"diag_sim_method", "diag_sim_df", "diag_regime_used", "diag_sim_used_obs"}]
            if keys:
                sub = diag.loc[keys]
                print("    diagnostics:", ", ".join([f"{k.replace('diag_','')}={sub[k]}" for k in sub.index]))

    out_png = outdir / "figure2_simulated_distributions.png"
    out_pdf = outdir / "figure2_simulated_distributions.pdf"

    plot_distributions(
        series_list=sims,
        labels=labels,
        outpath_png=out_png,
        outpath_pdf=out_pdf,
        bins=int(args.bins),
        kde=not args.no_kde,
        xlim=tuple(args.xlim) if args.xlim else None,
    )

    print(f"\nSaved:\n  {out_png}\n  {out_pdf}\n")


if __name__ == "__main__":
    main()

#ython plots/plot_simulated_distributions.py --files results/rolling_markowitz_results1.xlsx results/cvar_max_return_results1.xlsx --outdir results/figures