"""
Step 3: UN Comtrade Data Collection
US imports of HS 8708 (Motor Vehicle Parts & Accessories) — monthly, 2019–2024
Reporter: USA (842), Partner: World (0), Flow: Import (M)
Public API — no key required.

Variables pulled:
  cifvalue    — CIF import value (USD)     ← primary trade outcome
  fobvalue    — FOB import value (USD)     ← robustness
  netWgt      — Net weight (kg)            ← volume proxy
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_RAW  = ROOT / "data" / "raw"

import requests
import pandas as pd
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

BASE = "https://comtradeapi.un.org/public/v1/preview/C/M/HS"

records = []
missing = []

print("=" * 65)
print("UN COMTRADE DATA PULL — HS 8708, USA Imports, 2019–2024")
print("=" * 65)

for year in range(2019, 2025):
    for month in range(1, 13):
        period = f"{year}{month:02d}"
        params = {
            "reporterCode": "842",
            "period"       : period,
            "cmdCode"      : "8708",
            "flowCode"     : "M",
            "partnerCode"  : "0",          # World total
        }
        try:
            resp = requests.get(BASE, params=params, timeout=30)
            data = resp.json().get("data", [])
            if data:
                r = data[0]
                records.append({
                    "date"     : pd.Timestamp(f"{year}-{month:02d}-01"),
                    "period"   : period,
                    "cif_usd"  : r.get("cifvalue", np.nan),
                    "fob_usd"  : r.get("fobvalue",  np.nan),
                    "net_wgt_kg": r.get("netWgt",   np.nan),
                })
                status = f"${r['cifvalue']/1e9:.2f}B"
            else:
                missing.append(period)
                status = "MISSING"
        except Exception as e:
            missing.append(period)
            status = f"ERROR: {e}"

        print(f"  {period}: {status}")
        time.sleep(0.15)   # stay well under rate limit

# ── Build dataframe ──────────────────────────────────────────────────────────
df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
df = df.set_index("date")

print(f"\n{'='*65}")
print(f"Records fetched : {len(df)}")
print(f"Missing periods : {missing}")

# ── Interpolate sparse gaps (≤2 consecutive months) ──────────────────────────
full_idx = pd.date_range("2019-01-01", "2024-12-01", freq="MS")
df = df.reindex(full_idx)
# Record which months were originally missing from the API response BEFORE
# interpolation overwrites them -- saved as a column (not just printed) so
# 03b_comtrade_validation_plot.py can mark them on the chart from the actual
# data rather than a hardcoded date list that would silently go stale if
# this script is ever re-run against a schema-drifted or backfilled API.
df["interpolated"] = df["cif_usd"].isna()
df[["cif_usd","fob_usd","net_wgt_kg"]] = (
    df[["cif_usd","fob_usd","net_wgt_kg"]]
    .interpolate(method="linear", limit=2, limit_direction="both")
)

n_interp = df["cif_usd"].isna().sum()
print(f"Still missing after interpolation: {n_interp}")
print(f"\nFinal dataframe: {len(df)} monthly rows, {df.isna().sum().sum()} NaN cells")

# ── Key descriptive stats ─────────────────────────────────────────────────────
print("\n── CIF Import Value (USD billions) by year ──")
by_year = df["cif_usd"].resample("YE").sum() / 1e9
print(by_year.round(2).to_string())

print("\n── Monthly values around FSK collapse (Jan–Aug 2024) ──")
window = df.loc["2024-01-01":"2024-08-01", ["cif_usd","fob_usd","net_wgt_kg"]].copy()
window["cif_B"] = window["cif_usd"] / 1e9
print(window[["cif_B","net_wgt_kg"]].round(3).to_string())

# ── Pre-period mean for context ───────────────────────────────────────────────
pre = df.loc["2019-01-01":"2023-12-01", "cif_usd"]
print(f"\nPre-period mean CIF (2019–2023): ${pre.mean()/1e9:.3f}B/month")

# ── Save ──────────────────────────────────────────────────────────────────────
# Refuses to overwrite the existing frozen file if a fresh pull disagrees with
# it beyond a wide tolerance -- same guard 02b_portwatch_container_metrics.py
# already uses, extended to every numeric column here (cif_usd/fob_usd/
# net_wgt_kg) rather than just one. comtrade_hs8708_raw.csv is the frozen
# snapshot every number in the manuscript was verified against (see
# scripts/verify_claims.py); `python run_all.py --step 4` reproduces the full
# analysis from it without needing to re-collect.
df.index.name = "date"
target = DATA_RAW / "comtrade_hs8708_raw.csv"
NUMERIC_COLS = ["cif_usd", "fob_usd", "net_wgt_kg"]

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
              f"{target.with_name('comtrade_hs8708_raw_FRESH_PULL.csv').name}.")
        df.to_csv(target.with_name("comtrade_hs8708_raw_FRESH_PULL.csv"))
        print("Step 3 complete (frozen file preserved) ✓")
        raise SystemExit(0)
    else:
        print(f"\n  Fresh pull matches existing frozen file within tolerance "
              f"(worst median relative diff={worst_rel_diff:.2%}) -- safe to overwrite.")

df.to_csv(target)
print("\nSaved: data/raw/comtrade_hs8708_raw.csv")
print("Step 3 complete ✓")
