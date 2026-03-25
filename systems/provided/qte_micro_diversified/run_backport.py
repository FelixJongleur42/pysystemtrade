"""
Backport runner: QTE Micro Diversified → pysystemtrade comparison.

Runs stratified system variants and prints performance metrics:
  - Full set (7 instruments, from 2017): EWMAC-only and EWMAC+Carry
  - Tier 1 (5 instruments with 20+ yr data): EWMAC-only and EWMAC+Carry

Usage:
    cd pysystemtrade
    python -m systems.provided.qte_micro_diversified.run_backport [--years 3]
    python -m systems.provided.qte_micro_diversified.run_backport --years 0  # full history
    python -m systems.provided.qte_micro_diversified.run_backport --tier 1   # tier 1 only
"""

import sys
import argparse
from datetime import datetime, timedelta

import pandas as pd
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
    parser = argparse.ArgumentParser(description="QTE Micro Diversified backport comparison")
    parser.add_argument("--years", type=int, default=3,
                        help="Number of trailing years to report (0 = full history)")
    parser.add_argument("--tier", type=int, default=0,
                        help="Run specific tier only: 1=long-history, 2=full-set, 0=all")
    args = parser.parse_args()

    # ── Config registry: (label, config_path, tier) ──────────────────────────
    all_configs = [
        # Tier 2: Full instrument set (7 instruments, constrained by BITCOIN from 2017)
        ("Full (7 instr) EWMAC-only",
         "systems.provided.qte_micro_diversified.qte_micro_q2.yaml", 2),
        ("Full (7 instr) EWMAC+Carry",
         "systems.provided.qte_micro_diversified.qte_micro_q2_with_carry.yaml", 2),

        # Tier 1: Long-history instruments (5 instruments, 20+ years from 1999)
        ("Tier 1 (5 instr, 20+ yr) EWMAC-only",
         "systems.provided.qte_micro_diversified.qte_tier1_ewmac.yaml", 1),
        ("Tier 1 (5 instr, 20+ yr) EWMAC+Carry",
         "systems.provided.qte_micro_diversified.qte_tier1_ewmac_carry.yaml", 1),
    ]

    configs = [(lbl, path) for lbl, path, tier in all_configs
               if args.tier == 0 or tier == args.tier]

    if not configs:
        print(f"No configs found for tier {args.tier}")
        return

    # ── Run requested period ─────────────────────────────────────────────────
    for label, config_path in configs:
        print(f"\nBuilding system: {label} ...")
        try:
            system = build_system(config_path)
            print_stats(label, system, args.years)
        except Exception as e:
            print(f"  [{label}] FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ── Also run with full history for reference ─────────────────────────────
    if args.years > 0:
        print(f"\n{'─' * 60}")
        print(f"  (Full history for reference)")
        print(f"{'─' * 60}")
        for label, config_path in configs:
            try:
                system = build_system(config_path)
                print_stats(f"{label} [FULL]", system, 0)
            except Exception as e:
                print(f"  [{label} FULL] FAILED: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
