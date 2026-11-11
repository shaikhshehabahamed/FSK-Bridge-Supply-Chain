"""
Step 1: FRED Data Collection
FSK Bridge Collapse — Automotive Supply Chain Study
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_RAW  = ROOT / "data" / "raw"

import pandas as pd
import requests
import io
import warnings
warnings.filterwarnings("ignore")

BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
START, END = "2015-01-01", "2024-12-01"

SERIES = {
    # ── Primary price outcomes ──────────────────────────────────────────────
    "ppi_mv_parts_mfg"  : ("PCU33633363",    "PPI: Motor Vehicle Parts Manufacturing"),
    "ppi_mv_parts_comm" : ("WPU1412",        "PPI Commodity: Motor Vehicle Parts"),
    # ── Primary production outcome ──────────────────────────────────────────
    "auto_production"   : ("DAUPSA",         "Domestic Auto Production (Thousands of Units, Monthly, SA)"),
    # ── Demand / new orders ─────────────────────────────────────────────────
    "new_orders_mv"     : ("AMTMNO",         "Mfrs New Orders: Motor Vehicles & Parts"),
    "vehicle_sales"     : ("TOTALSA",        "Total Vehicle Sales (SAAR)"),
    # ── Consumer-side price ─────────────────────────────────────────────────
    "cpi_mv_parts"      : ("CUUR0000SETC",   "CPI: Motor Vehicle Parts & Equipment"),
    # ── PLACEBO (no Baltimore port exposure) ────────────────────────────────
    "ppi_petrochem"     : ("PCU325110325110","PPI: Petrochemical Manufacturing [PLACEBO]"),
}

records, meta = {}, []

print("=" * 65)
print("FRED DATA PULL")
print("=" * 65)

for label, (ticker, title) in SERIES.items():
    resp = requests.get(BASE + ticker, timeout=30)
    assert resp.status_code == 200, f"HTTP {resp.status_code} for {ticker}"

    s = pd.read_csv(io.StringIO(resp.text), parse_dates=["observation_date"],
                    index_col="observation_date").squeeze()
    s.index.name = "date"
    s = s.loc[START:END].resample("MS").last()
    s.name = label
    records[label] = s

    meta.append(dict(label=label, ticker=ticker, title=title,
                     start=str(s.index.min().date()), end=str(s.index.max().date()),
                     n_obs=len(s), n_miss=int(s.isna().sum()),
                     min=round(float(s.min()),3), max=round(float(s.max()),3)))

    print(f"\n  [{ticker}]  {label}")
    print(f"    {title}")
    print(f"    {s.index.min().date()} → {s.index.max().date()}  |  "
          f"Obs={len(s)}  Missing={int(s.isna().sum())}")
    print(f"    Min={s.min():.3f}  Max={s.max():.3f}")

# ── Master dataframe ─────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df.index.name = "date"
df = df.ffill(limit=1)

print("\n" + "=" * 65)
print("MASTER DATAFRAME SUMMARY")
print("=" * 65)
print(f"Shape : {df.shape}  |  {df.index.min().date()} → {df.index.max().date()}")
print(f"Total missing cells: {df.isna().sum().sum()}")

print("\n── Values around FSK collapse (Feb – Jul 2024) ──")
window = df.loc["2024-02-01":"2024-07-01"].round(3)
print(window.to_string())

print("\n── Year-over-year comparison: auto_production ──")
y23 = df.loc["2023-01-01":"2023-12-01", "auto_production"]
y24 = df.loc["2024-01-01":"2024-12-01", "auto_production"]
yoy = pd.DataFrame({"2023": y23.values, "2024": y24.values,
                    "diff": y24.values - y23.values,
                    "pct_chg": (y24.values/y23.values - 1)*100},
                   index=range(1,13))
yoy.index.name = "month"
print(yoy.round(2).to_string())

# ── Save ─────────────────────────────────────────────────────────────────────
# Refuses to overwrite the existing frozen file if a fresh pull disagrees with
# it beyond a wide tolerance -- same guard 02b_portwatch_container_metrics.py
# already uses, extended here across every FRED series (FRED periodically
# revises seasonal-adjustment factors, which can retroactively shift "old"
# months even without any real change; the 5% threshold mirrors 02b's
# existing tolerance rather than a series-specific figure re-derived here).
# fred_raw.csv is the frozen snapshot every number in the manuscript was
# verified against (see scripts/verify_claims.py); `python run_all.py --step 4`
# reproduces the full analysis from it without needing to re-collect.
target = DATA_RAW / "fred_raw.csv"
NUMERIC_COLS = list(SERIES.keys())

if target.exists():
    existing = pd.read_csv(target, parse_dates=["date"], index_col="date")
    common_idx = df.index.intersection(existing.index)
    rel_diffs = [
        ((df.loc[common_idx, c] - existing.loc[common_idx, c]).abs()
         / existing.loc[common_idx, c].abs().clip(lower=1)).median()
        for c in NUMERIC_COLS if c in existing.columns
    ]
    worst_rel_diff = max(rel_diffs) if rel_diffs else 0.0
    if worst_rel_diff > 0.05 or abs(len(df) - len(existing)) > 5:
        print(f"\n  WARNING: fresh pull disagrees with the existing frozen "
              f"{target.name} (worst median relative diff={worst_rel_diff:.1%}, "
              f"row counts {len(df)} vs {len(existing)}).")
        print("  NOT overwriting the frozen file -- it is the snapshot every")
        print("  number in the manuscript was verified against. Fresh pull")
        print("  saved separately for inspection at "
              f"{target.with_name('fred_raw_FRESH_PULL.csv').name}.")
        df.to_csv(target.with_name("fred_raw_FRESH_PULL.csv"))
        print("Step 1 complete (frozen file preserved) ✓")
        raise SystemExit(0)
    else:
        print(f"\n  Fresh pull matches existing frozen file within tolerance "
              f"(worst median relative diff={worst_rel_diff:.2%}) -- safe to overwrite.")

df.to_csv(target)
pd.DataFrame(meta).to_csv(str(DATA_RAW / "fred_meta.csv"), index=False)
print("\nSaved → data/raw/fred_raw.csv  &  data/raw/fred_meta.csv")
print("Step 1 complete ✓")
