"""
Backport runner: QTE Micro Diversified Q2 → pysystemtrade comparison.

Runs three system variants and prints performance metrics:
  A) EWMAC-only   (matching Q2 config — no carry)
  B) EWMAC+Carry  (matching original Q config style)
  C) Chapter 15   (standard PST reference)

Usage:
    cd pysystemtrade
    python -m systems.provided.qte_micro_diversified.run_backport [--years 3]
"""

import sys
import argparse
from datetime import datetime, timedelta

from sysdata.sim.csv_futures_sim_data import csvFuturesSimData
from sysdata.config.configdata import Config
from systems.forecasting import Rules
from systems.basesystem import System
from systems.forecast_combine import ForecastCombine
from systems.forecast_scale_cap import ForecastScaleCap
from systems.rawdata import RawData
from systems.positionsizing import PositionSizing
from systems.portfolio import Portfolios
from systems.accounts.accounts_stage import Account


def build_system(config_path: str) -> System:
    data = csvFuturesSimData()
    config = Config(config_path)
    return System(
        [
            Account(),
            Portfolios(),
            PositionSizing(),
            RawData(),
            ForecastCombine(),
            ForecastScaleCap(),
            Rules(),
        ],
        data,
        config,
    )


def print_stats(label: str, system: System, years: int):
    """Print key account-level stats for the portfolio."""
    try:
        account_curve = system.accounts.portfolio()
        pct = account_curve.percent
    except Exception as e:
        print(f"  [{label}] ERROR computing portfolio account: {e}")
        import traceback
        traceback.print_exc()
        return

    # pct is an accountCurveGroup (subclass of pd.Series) — daily % returns
    # Convert to plain Series to avoid custom __getitem__ issues
    import pandas as pd
    pct_series = pd.Series(pct.values, index=pct.index)

    if pct_series.empty:
        print(f"  [{label}] No data available")
        return

    # Trim to the last N years of available data
    if years > 0:
        end_date = pct_series.index[-1]
        start_date = end_date - timedelta(days=years * 365)
        trimmed = pct_series[pct_series.index >= start_date]
    else:
        trimmed = pct_series

    if trimmed.empty:
        print(f"  [{label}] No data in the requested period")
        return

    ann_mean = float(trimmed.mean()) * 256
    ann_std = float(trimmed.std()) * (256 ** 0.5)
    sharpe = ann_mean / ann_std if ann_std > 0 else 0.0

    # PST percent returns are in percentage points (0.5 = 0.5%), convert to fractions
    trimmed_frac = trimmed / 100.0

    # Cumulative return for CAGR
    cum = (1 + trimmed_frac).cumprod()
    total_return = float(cum.iloc[-1]) - 1.0
    n_years = len(trimmed) / 256.0
    cagr = (1 + total_return) ** (1.0 / n_years) - 1.0 if n_years > 0 and total_return > -1 else 0.0

    # Max drawdown
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = float(drawdown.min())

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Period:          {trimmed.index[0].date()} → {trimmed.index[-1].date()}  ({len(trimmed)} days)")
    print(f"  CAGR:            {cagr * 100:+.2f}%")
    print(f"  Annualised Vol:  {ann_std:.2f}%")
    print(f"  Sharpe:          {sharpe:.3f}")
    print(f"  Max Drawdown:    {max_dd * 100:.2f}%")
    print(f"  Calmar:          {abs(cagr / max_dd):.3f}" if max_dd < 0 else "  Calmar:          N/A")

    # Also print PST's built-in stats for full curve
    try:
        print(f"\n  PST built-in stats (full curve):")
        print(f"    Sharpe:        {pct.sharpe():.3f}")
        print(f"    Ann Mean:      {pct.ann_mean():.2f}%")
        print(f"    Ann Std:       {pct.ann_std():.2f}%")
        print(f"    Worst DD:      {pct.worst_drawdown():.2f}%")
    except Exception:
        pass
    print()


def main():
    parser = argparse.ArgumentParser(description="QTE Micro Diversified Q2 backport comparison")
    parser.add_argument("--years", type=int, default=3, help="Number of trailing years to report (0 = full history)")
    args = parser.parse_args()

    configs = [
        ("A: EWMAC-only (Q2 backport)", "systems.provided.qte_micro_diversified.qte_micro_q2.yaml"),
        ("B: EWMAC+Carry (Q+carry backport)", "systems.provided.qte_micro_diversified.qte_micro_q2_with_carry.yaml"),
    ]

    for label, config_path in configs:
        print(f"\nBuilding system: {label} ...")
        try:
            system = build_system(config_path)
            print_stats(label, system, args.years)
        except Exception as e:
            print(f"  [{label}] FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Also run with full history for reference
    if args.years > 0:
        print(f"\n{'─' * 60}")
        print(f"  (Re-running with full history for reference)")
        print(f"{'─' * 60}")
        for label, config_path in configs:
            try:
                system = build_system(config_path)
                print_stats(f"{label} [FULL]", system, 0)
            except Exception as e:
                print(f"  [{label} FULL] FAILED: {e}")


if __name__ == "__main__":
    main()
