# plots/plot_oos_performance.py
"""
Figure 1 — Rolling out-of-sample performance (realized) + benchmarks

Adds:
- S&P 500 benchmark (SP500 or SPY)
- 60/40 benchmark (SP500/SPY + UST_7_10Y/IEF)

It searches ALL provided .xlsx files for a usable Monthly_Returns sheet,
so benchmarks will show even if the first file doesn’t contain them.

Usage:
  python plots/plot_oos_performance.py --files results/rolling_markowitz_results1.xlsx results/cvar_max_return_results1.xlsx --outdir results/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import pandas as pd
import matplotlib.pyplot as plt


RETURN_SHEET_CANDIDATES = [
    "OOS_Returns_NET",
    "Net_Returns",
    "Transaction_Costs",
    "OOS_Returns",
    "OOS_Returns_GROSS",
    "Returns",
]

RETURN_COL_CANDIDATES = [
    "oos_return",
    "net_return",
    "gross_return",
    "portfolio_return_net",
    "portfolio_return_gross",
    "return",
    "returns",
]

MONTHLY_RETURNS_SHEET_CANDIDATES = [
    "Monthly_Returns",
    "MONTHLY_RETURNS",
    "monthly_returns",
    "Monthly returns",
]

# Accept both your internal names and raw ticker-style names
SP500_COL_CANDIDATES = ["SP500", "SPY", "SP500_TR", "SP500_return"]
BOND_COL_CANDIDATES = ["UST_7_10Y", "IEF", "UST", "BONDS"]


NAVY = "#0B1F3B"
BURGUNDY = "#7A1E2B"
BENCH_GRAY = "#444444"


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


def _linestyle_from_label(label: str) -> str:
    l = label.lower()
    if "60/40" in l or "sp500" in l or "s&p" in l:
        return "--"
    return "-"


def _set_datetime_index_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if len(df.columns) >= 2:
        first = df.columns[0]
        parsed = pd.to_datetime(df[first], errors="coerce")
        if parsed.notna().mean() > 0.8:
            out = df.copy()
            out[first] = parsed
            out = out.set_index(first)
            return out
    return df


def _read_returns_from_sheet(xlsx_path: Path, sheet_name: str) -> Optional[pd.Series]:
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    except Exception:
        return None

    df = _set_datetime_index_if_possible(df)

    col = None
    for c in RETURN_COL_CANDIDATES:
        if c in df.columns:
            col = c
            break

    if col is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) == 1:
            col = num_cols[0]

    if col is None:
        return None

    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None

    if not isinstance(s.index, pd.DatetimeIndex):
        idx = pd.to_datetime(s.index, errors="coerce")
        if idx.notna().mean() > 0.8:
            s.index = idx
        else:
            return None

    s = s.sort_index()
    s.name = "oos_return"
    return s


def read_oos_returns(xlsx_path: Path, sheet_candidates: List[str]) -> Tuple[pd.Series, str]:
    xl = pd.ExcelFile(xlsx_path)

    for sh in sheet_candidates:
        if sh in xl.sheet_names:
            s = _read_returns_from_sheet(xlsx_path, sh)
            if s is not None:
                return s, sh

    for sh in xl.sheet_names:
        s = _read_returns_from_sheet(xlsx_path, sh)
        if s is not None:
            return s, sh

    raise ValueError(f"No usable return series in {xlsx_path.name}")


def _find_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _try_read_monthly_returns_from_file(xlsx_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """
    Returns (df, used_sheet, reason_if_failed)
    """
    try:
        xl = pd.ExcelFile(xlsx_path)
    except Exception as e:
        return None, None, f"cannot open excel: {e}"

    used_sheet = None
    for sh in MONTHLY_RETURNS_SHEET_CANDIDATES:
        if sh in xl.sheet_names:
            used_sheet = sh
            break
    if used_sheet is None:
        return None, None, "no Monthly_Returns sheet"

    try:
        df = pd.read_excel(xlsx_path, sheet_name=used_sheet)
        df = _set_datetime_index_if_possible(df)
        if not isinstance(df.index, pd.DatetimeIndex):
            return None, None, "Monthly_Returns has no datetime index"
        # keep numeric cols
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        df = df[num_cols].sort_index()
        if df.empty:
            return None, None, "Monthly_Returns has no numeric columns"
        return df, used_sheet, None
    except Exception as e:
        return None, None, f"failed reading Monthly_Returns: {e}"


def _build_benchmarks(monthly_assets: pd.DataFrame) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}

    sp = _find_first_existing_col(monthly_assets, SP500_COL_CANDIDATES)
    bd = _find_first_existing_col(monthly_assets, BOND_COL_CANDIDATES)

    if sp is not None:
        out["S&P 500 (proxy)"] = monthly_assets[sp].astype(float).rename("oos_return")

    if sp is not None and bd is not None:
        r_6040 = (0.60 * monthly_assets[sp] + 0.40 * monthly_assets[bd]).astype(float)
        out["60/40 (proxy)"] = r_6040.rename("oos_return")

    return out


def plot_cumulative_performance(series_list: List[pd.Series], labels: List[str], out_png: Path, out_pdf: Path) -> None:
    df = pd.concat(series_list, axis=1, join="inner")
    df.columns = labels
    wealth = (1.0 + df).cumprod()

    fig, ax = plt.subplots(figsize=(11, 6))
    for col in wealth.columns:
        ax.plot(
            wealth.index,
            wealth[col],
            label=col,
            linewidth=2.5,
            color=_color_from_label(col),
            linestyle=_linestyle_from_label(col),
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative wealth (start = 1.0)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--outdir", default="results/figures")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()

    paths = [Path(f) for f in args.files]
    outdir = Path(args.outdir)
    sheet_candidates = [args.sheet] if args.sheet else RETURN_SHEET_CANDIDATES

    series_list: List[pd.Series] = []
    labels: List[str] = []

    print("\nReading OOS return series:")
    for p in paths:
        s, used_sheet = read_oos_returns(p, sheet_candidates)
        series_list.append(s)
        labels.append(_label_from_filename(p))
        print(f"  - {p.name}: sheet='{used_sheet}'  n={len(s):,}  {s.index.min().date()}..{s.index.max().date()}")

    if not args.no_bench:
        monthly_assets = None
        used_file = None
        used_sheet = None
        failures = []

        for p in paths:
            df, sh, reason = _try_read_monthly_returns_from_file(p)
            if df is not None:
                monthly_assets = df
                used_file = p
                used_sheet = sh
                break
            failures.append(f"{p.name}: {reason}")

        if monthly_assets is not None:
            benches = _build_benchmarks(monthly_assets)
            if benches:
                for lab, s in benches.items():
                    series_list.append(s)
                    labels.append(lab)
                print(f"\nBenchmarks added from: {used_file.name} / sheet='{used_sheet}' -> {list(benches.keys())}")
            else:
                print("\nBenchmarks NOT added: Monthly_Returns found, but no suitable columns.")
                print("  Need one of:", SP500_COL_CANDIDATES, "and for 60/40 also one of:", BOND_COL_CANDIDATES)
                print("  Available cols:", list(monthly_assets.columns))
        else:
            print("\nBenchmarks NOT added: couldn't locate usable Monthly_Returns in any file.")
            for msg in failures:
                print("  -", msg)

    out_png = outdir / "figure1_oos_performance.png"
    out_pdf = outdir / "figure1_oos_performance.pdf"
    plot_cumulative_performance(series_list, labels, out_png, out_pdf)

    print(f"\nSaved:\n  {out_png}\n  {out_pdf}\n")


if __name__ == "__main__":
    main()
#python plots/plot_oos_performance.py --files results/rolling_markowitz_results1.xlsx results/cvar_max_return_results1.xlsx --outdir results/figures --no-bench

#python plots/plot_oos_performance.py --files results/rolling_markowitz_results1.xlsx results/cvar_max_return_results1.xlsx --outdir results/figures