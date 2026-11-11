"""
Step 4a: Master Dataset Assembly
Merges FRED + PortWatch + Comtrade into one clean monthly panel.
Adds all derived variables needed for the ITS model.
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"



import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Key dates ────────────────────────────────────────────────────────────────
COLLAPSE   = pd.Timestamp("2024-03-26")  # FSK bridge collapses
PARTIAL_OP = pd.Timestamp("2024-04-29")  # 280-ft channel opens
FULL_OP    = pd.Timestamp("2024-06-10")  # full channel cleared

# For monthly assignment: collapse falls in March 2024
COLLAPSE_M  = pd.Timestamp("2024-03-01")
DISRUPTION_END_M = pd.Timestamp("2024-05-01")  # May = last full disruption month
RECOVERY_START_M = pd.Timestamp("2024-06-01")  # partial; full July onwards

# ── Load datasets ─────────────────────────────────────────────────────────────
print("Loading raw datasets...")

fred = pd.read_csv(str(DATA_RAW / "fred_raw.csv"),
                   parse_dates=["date"], index_col="date")

pw = pd.read_csv(str(DATA_RAW / "portwatch_baltimore_monthly.csv"),
                 parse_dates=["date"]).set_index("date")

ct = pd.read_csv(str(DATA_RAW / "comtrade_hs8708_raw.csv"),
                 parse_dates=["date"], index_col="date")

# ── Align to common index: 2019-01 to 2024-12 ────────────────────────────────
IDX = pd.date_range("2019-01-01", "2024-12-01", freq="MS")

# FRED: slice to 2019-2024
fred = fred.reindex(IDX)

# PortWatch: already has full coverage
pw = pw.reindex(IDX)

# Comtrade: already on IDX
ct = ct.reindex(IDX)

# ── Build master ──────────────────────────────────────────────────────────────
master = pd.DataFrame(index=IDX)
master.index.name = "date"

# ── FRED variables ────────────────────────────────────────────────────────────
# NOTE: WPU1412 (PPI Commodity: Motor Vehicle Parts) is collected in
# 01_fred_data.py and shown in the Step 1 validation figure, but a derived
# "ppi_mv_parts_wpu" column was removed from here on 2026-07-19: it was
# labeled "(robustness)" but never actually used in any ITS/event-study/
# robustness specification anywhere in the pipeline -- dead code, not a real
# robustness check. If a WPU1412-based robustness check is added later, add
# it back deliberately rather than reviving unused, unlabeled scaffolding.
master["auto_prod"]        = fred["auto_production"]        # DAUPSA  [primary outcome]
master["ppi_mv_parts"]     = fred["ppi_mv_parts_mfg"]       # PCU33633363
master["new_orders"]       = fred["new_orders_mv"]          # AMTMNO
master["vehicle_sales"]    = fred["vehicle_sales"]          # TOTALSA
master["cpi_mv_parts"]     = fred["cpi_mv_parts"]           # CUUR0000SETC
master["ppi_petrochem"]    = fred["ppi_petrochem"]          # PLACEBO

# ── PortWatch variable ────────────────────────────────────────────────────────
master["balt_import_vol"]  = pw["baltimore_import_vol"]     # shock variable

# ── Comtrade variable ─────────────────────────────────────────────────────────
master["hs8708_cif_B"]     = ct["cif_usd"] / 1e9           # USD billions
master["hs8708_wgt_M"]     = ct["net_wgt_kg"] / 1e6        # million kg

# ── Time variables ────────────────────────────────────────────────────────────
master["t"]       = np.arange(len(master))      # time index (0, 1, 2, ...)
master["year"]    = master.index.year
master["month"]   = master.index.month

# ── Phase indicators ──────────────────────────────────────────────────────────
# post = 1 from March 2024 onwards (month of collapse)
master["post"]       = (master.index >= COLLAPSE_M).astype(int)

# disruption = 1 for March, April, May 2024 (port substantially closed)
master["disruption"] = (
    (master.index >= COLLAPSE_M) & (master.index <= DISRUPTION_END_M)
).astype(int)

# recovery = 1 for June+ 2024 (port reopening, phased)
master["recovery"]   = (master.index >= RECOVERY_START_M).astype(int)

# time since collapse (0 in pre-period, 1 in March 2024, etc.)
collapse_t = master.loc[COLLAPSE_M, "t"]
master["t_post"]     = (master["t"] - collapse_t) * master["post"]

# ── COVID controls ────────────────────────────────────────────────────────────
# COVID shock: April–June 2020 (plants shut down)
master["covid_shock"] = (
    (master.index >= "2020-04-01") & (master.index <= "2020-06-01")
).astype(int)

# UAW strike: Oct–Nov 2023 (Ford, GM, Stellantis)
master["uaw_strike"]  = (
    (master.index >= "2023-10-01") & (master.index <= "2023-11-01")
).astype(int)

# ── Log transforms for ITS model ─────────────────────────────────────────────
for col in ["auto_prod","ppi_mv_parts",
            "new_orders","vehicle_sales","hs8708_cif_B","balt_import_vol"]:
    master[f"ln_{col}"] = np.log(master[col].clip(lower=0.001))

# ── Month dummies (seasonality) ───────────────────────────────────────────────
for m in range(2, 13):   # Jan is the reference
    master[f"m{m:02d}"] = (master["month"] == m).astype(int)

# ── Validation ───────────────────────────────────────────────────────────────
print(f"\nMaster dataset: {master.shape}")
print(f"Date range: {master.index.min().date()} → {master.index.max().date()}")
print(f"\nPhase counts:")
print(f"  Pre-disruption (pre=0):        {(master.post==0).sum()} months")
print(f"  Disruption (Mar-May 2024):     {master.disruption.sum()} months")
print(f"  Recovery (Jun-Dec 2024):       {(master.recovery & ~master.disruption).sum()} months")
print(f"\nMissing values per series:")
key_cols = ["auto_prod","ppi_mv_parts","balt_import_vol","hs8708_cif_B"]
print(master[key_cols].isna().sum().to_string())

print("\nFirst 3 / last 3 rows (key columns only):")
print(master[["auto_prod","ppi_mv_parts","balt_import_vol","hs8708_cif_B",
              "post","disruption","t"]].head(3).to_string())
print("...")
print(master[["auto_prod","ppi_mv_parts","balt_import_vol","hs8708_cif_B",
              "post","disruption","t"]].tail(6).to_string())

# ── Save ──────────────────────────────────────────────────────────────────────
master.to_csv(str(DATA_PROC / "master.csv"))
print(f"\nSaved: data/processed/master.csv  ({master.shape[0]} rows × {master.shape[1]} cols)")
print("Step 4a complete ✓")
