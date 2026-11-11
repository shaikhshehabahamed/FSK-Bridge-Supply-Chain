"""
Step 6: Implied Safety Stock Inference (IE-specific contribution)
==================================================================
Uses the inventory balance identity and observed production continuity
during the port closure to back-calculate the effective buffer stock
(in days-of-supply) maintained by automotive JIT supply chains.

Formal model
─────────────
During a port closure of D days, if a firm's production rate is P_t and
the port normally supplies fraction θ of its parts needs:

  Production_actual = P_normal × [1 − θ · max(0, Closure − Buffer)]

Solving for Buffer (the fraction of monthly demand covered by safety stock
+ rerouting capacity):

  Buffer = 1 − (P_normal − P_actual) / (θ · P_normal)

Implied safety stock in days-of-supply:
  SS_days = Buffer × D_closure   (D_closure ≈ 30 days for April 2024)

We compute this for a grid of θ values [0.20, 0.25, …, 0.55] to report
a central estimate and uncertainty range.

Baltimore's θ is triangulated from three sources:
  1. PortWatch RoRo share (portcalls)
  2. PortWatch import volume share (Container_Metrics)
  3. Industry reports (Maryland Port Administration: #1 RoRo port, 847k vehicles 2023)
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

master  = pd.read_csv(str(DATA_PROC / "master.csv"),
                       parse_dates=["date"], index_col="date")
pw_m    = pd.read_csv(str(DATA_RAW / "portwatch_monthly.csv"),
                       parse_dates=["date"])
balt_m  = pd.read_csv(str(DATA_RAW / "portwatch_baltimore_monthly.csv"),
                       parse_dates=["date"]).set_index("date")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6A: Estimate Baltimore's supply share θ
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 6A — Estimating Baltimore's automotive supply share (θ)")
print("=" * 70)

# ── Method 1: RoRo portcall share from PortWatch Daily_Ports_Data ─────────
# theta_portcalls and theta_vol (Method 2, below) are both DELIBERATELY
# restricted to PortWatch's own 3-port monitored set (Baltimore, Brunswick,
# Norfolk) with no adjustment for East Coast automotive RoRo volume moving
# through ports PortWatch doesn't track (e.g. Philadelphia, NY/NJ). That
# untracked-coverage concern is real, but is handled once, transparently,
# by Method 3 (theta_lit) below -- not baked silently into Methods 1/2 as
# well, which would double-count the same adjustment without disclosure.
pre_portcalls = pw_m[pw_m["date"].dt.year.isin([2019, 2020, 2021])].copy()
balt_rc  = pre_portcalls[pre_portcalls["port_label"]=="baltimore"]["portcalls_roro"].mean()
brun_rc  = pre_portcalls[pre_portcalls["port_label"]=="brunswick"]["portcalls_roro"].mean()
norf_rc  = pre_portcalls[pre_portcalls["port_label"]=="norfolk"]["portcalls_roro"].mean()

theta_portcalls = balt_rc / (balt_rc + brun_rc + norf_rc)
print(f"\nMethod 1 — RoRo portcall share (2019-2021):")
print(f"  Baltimore:  {balt_rc:.1f} calls/month")
print(f"  Brunswick:  {brun_rc:.1f} calls/month")
print(f"  Norfolk:    {norf_rc:.1f} calls/month")
print(f"  → Baltimore RoRo share (3-port): θ₁ = {theta_portcalls:.3f}")

# ── Method 2: PortWatch Container_Metrics import volume share ─────────────
pre_balt_vol = balt_m[balt_m.index.year.isin([2019,2020,2021])]["baltimore_import_vol"]
pre_balt_mean = pre_balt_vol.mean()
# Brunswick from PortWatch RoRo import_roro
brun_import = pre_portcalls[pre_portcalls["port_label"]=="brunswick"]["import_roro"].mean()
balt_import = pre_portcalls[pre_portcalls["port_label"]=="baltimore"]["import_roro"].mean()
theta_vol = balt_import / (balt_import + brun_import +
                           pre_portcalls[pre_portcalls["port_label"]=="norfolk"]["import_roro"].mean())
print(f"\nMethod 2 — RoRo import volume share:")
print(f"  Baltimore mean import_roro: {balt_import:,.0f}")
print(f"  Brunswick mean import_roro: {brun_import:,.0f}")
print(f"  → Baltimore volume share: θ₂ = {theta_vol:.3f}")

# ── Method 3: Conservative coverage-adjustment anchor ──────────────────────
# Methods 1-2 are restricted to PortWatch's monitored port set (Baltimore,
# Brunswick, Norfolk); automotive RoRo volume also moves through East Coast
# ports PortWatch does not track (e.g. Philadelphia, NY/NJ), which would
# mechanically inflate Baltimore's share if omitted. theta_lit is an author
# judgment discounting theta_1/theta_2 for this coverage gap -- it is NOT
# drawn from a specific external report and is disclosed as such in the
# manuscript's limitations. It is directionally anchored by Maryland Port
# Administration's verified, citable statistic that Baltimore was the #1 US
# vehicle-import port for 13 consecutive years (847,158 vehicles, 2023;
# mpa.maryland.gov / Governor of Maryland press release, Feb 2024).
theta_lit = 0.40   # conservative anchor, disclosed as an author judgment (see Limitations)

print(f"\nMethod 3 — Conservative coverage-adjustment anchor: θ₃ = {theta_lit:.3f}")
print(f"  (Author judgment, range 0.35-0.45; anchored to MPA's confirmed #1-port statistic, NOT a specific external market-share report)")

# ── Central estimate ──────────────────────────────────────────────────────
# theta_central is the mean of the two EMPIRICAL methods only (theta_portcalls,
# theta_vol). theta_lit (Method 3) is deliberately excluded from the central
# point estimate: pooling an undisclosed-source author judgment equally with
# two data-derived estimates would let a guess drive the headline
# safety-stock number on equal footing with real measurements, which is not
# defensible (fixed 2026-07-19 -- theta_central previously averaged all three
# and came out to 0.572; excluding theta_lit -- the smallest of the three
# values -- pulls theta_central UP to ~0.658). Since SS_days = D*(1 -
# apr_drop_pct/theta), a HIGHER theta implies a LARGER buffer (a higher
# assumed Baltimore exposure combined with the SAME observed production
# drop implies more of the shock was absorbed by inventory, not less) --
# the central safety-stock estimate moves from 15.7 to ~17.6 days. theta_lit
# still informs the reported sensitivity RANGE as an explicit conservative
# lower bound -- that is exactly the role it was designed for -- it is just
# no longer averaged into the central estimate on equal footing with data.
theta_empirical = np.array([theta_portcalls, theta_vol])
theta_central    = theta_empirical.mean()
theta_low        = min(theta_empirical.min(), theta_lit) * 0.90   # 10% conservative margin
theta_high       = theta_empirical.max() * 1.10                    # 10% liberal margin

print(f"\nCentral θ estimate (empirical methods 1-2 only): {theta_central:.3f}  "
      f"[{theta_low:.3f} – {theta_high:.3f}]")
print(f"  (Method 3 / theta_lit={theta_lit:.3f} informs the conservative low end of the")
print(f"   range above, but is excluded from the central point estimate -- see comment.)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6B: Production gaps during closure
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6B — Production gaps during closure months")
print("=" * 70)

# Use YoY comparison as the most defensible gap estimate
# (avoids ITS trend-extrapolation issues with short post-period)
prod = master["auto_prod"]
gaps = {}
for dt24, dt23, label in [
    ("2024-04-01", "2023-04-01", "April 2024 (full closure)"),
    ("2024-05-01", "2023-05-01", "May 2024  (partial closure)"),
    ("2024-06-01", "2023-06-01", "June 2024 (partial reopening)"),
]:
    p24 = prod.loc[dt24]; p23 = prod.loc[dt23]
    yoy = (p24 / p23 - 1)
    gaps[label] = {"actual": p24, "baseline": p23, "yoy": yoy,
                   "drop_pct": -yoy, "drop_k": p23 - p24}
    print(f"  {label}: actual={p24:.1f}k  baseline={p23:.1f}k  "
          f"YoY={yoy*100:+.1f}%  drop={p23-p24:.1f}k units")

# Focus on April 2024 — the peak disruption month
apr_drop_pct = gaps["April 2024 (full closure)"]["drop_pct"]
apr_baseline  = gaps["April 2024 (full closure)"]["baseline"]
D_closure = 30   # effective closure days in April 2024

print(f"\nFocus month: April 2024")
print(f"  Production baseline (2023 April): {apr_baseline:.1f}k units/month")
print(f"  Production decline:               {apr_drop_pct*100:.1f}%")
print(f"  Effective closure days (D):       {D_closure}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6C: Implied Safety Stock calculation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6C — Implied Safety Stock Inference")
print("=" * 70)
print()
print("Model: P_actual = P_normal × [1 − θ × max(0, Closure − Buffer)]")
print("Solving: Buffer_fraction = 1 − (ΔP/P_normal) / θ")
print("SS_days = Buffer_fraction × D_closure")
print()
print(f"{'θ (Baltimore share)':>22}  {'Buffer fraction':>18}  "
      f"{'SS (days)':>12}  {'Interpretation':>25}")
print("-" * 82)

theta_grid = np.arange(0.20, 0.81, 0.05)  # widened 2026-07-19 to cover
# theta_central (~0.66) and theta_high (~0.74) now that theta_lit no longer
# pulls the central estimate down toward the low end of the old 0.20-0.60 grid
ss_results = []

for theta in theta_grid:
    # Buffer fraction: share of monthly demand covered by safety stock + rerouting
    buffer_frac = 1.0 - (apr_drop_pct / theta)
    # Safety stock in days
    ss_days = buffer_frac * D_closure
    # Only meaningful if buffer_frac in [0, 1]
    valid = 0.0 <= buffer_frac <= 1.0
    flag  = " ← central" if abs(theta - theta_central) < 0.03 else \
            " ← low"    if abs(theta - theta_low)    < 0.03 else \
            " ← high"   if abs(theta - theta_high)   < 0.03 else ""
    interp = ("feasible" if valid else "infeasible (θ too small)")

    ss_results.append({"theta": theta, "buffer_frac": buffer_frac,
                        "ss_days": ss_days, "valid": valid})

    print(f"  θ = {theta:.2f}  →  Buffer = {buffer_frac:+.3f}  "
          f"  SS = {ss_days:+6.1f} days  {interp}{flag}")

df_ss = pd.DataFrame(ss_results)
valid_ss = df_ss[df_ss["valid"]]

# ── Central estimate ──────────────────────────────────────────────────────
# Use θ_central to get central SS estimate
buf_central = 1.0 - (apr_drop_pct / theta_central)
ss_central  = buf_central * D_closure

buf_low  = 1.0 - (apr_drop_pct / theta_high)   # higher θ → smaller buffer
buf_high = 1.0 - (apr_drop_pct / theta_low)    # lower θ → larger buffer
ss_range_lo = buf_low  * D_closure
ss_range_hi = buf_high * D_closure

print()
print("=" * 70)
print("CENTRAL ESTIMATE (θ = {:.2f}):".format(theta_central))
print(f"  Implied buffer fraction: {buf_central:.3f} ({buf_central*100:.1f}% of monthly demand)")
print(f"  Implied safety stock:    {ss_central:.1f} days-of-supply")
print(f"  Sensitivity range:       [{min(ss_range_lo,ss_range_hi):.1f}, {max(ss_range_lo,ss_range_hi):.1f}] days")
print()
print("INDUSTRY BENCHMARK COMPARISON:")
print("  Toyota recommended SS (JIT):          5–7 days")
print("  US auto industry pandemic target:     14–21 days (post-2020)")
print("  Our implied estimate:                 {:.1f} days  [{:.1f}, {:.1f}]".format(
    ss_central, ss_range_lo, ss_range_hi))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6D: Rerouting efficiency and residual gap
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6D — Rerouting efficiency analysis")
print("=" * 70)

hs8708 = master["hs8708_cif_B"]

# Comtrade-based rerouting signal: Apr 2024 imports vs Apr 2023
apr24_imports = hs8708.loc["2024-04-01"]
apr23_imports = hs8708.loc["2023-04-01"]
import_yoy    = (apr24_imports / apr23_imports - 1) * 100

# Pre-period monthly growth trend
pre_ct = master.loc[master["post"]==0, "hs8708_cif_B"]
pre_trend_monthly = stats.linregress(range(len(pre_ct)), pre_ct.values).slope
cf_growth = pre_trend_monthly * 12 / pre_ct.mean() * 100   # annualised %

rerouting_surplus_pct = import_yoy - cf_growth / 12  # above-trend import increase

print(f"\n  HS 8708 April 2024 YoY:          {import_yoy:+.1f}%")
print(f"  Pre-period monthly trend:         {pre_trend_monthly/pre_ct.mean()*100:+.2f}%/month")
print(f"  Rerouting surplus (above trend):  {rerouting_surplus_pct:+.1f}%")
print(f"  → Approximately {rerouting_surplus_pct:.0f}% of normal Baltimore supply")
print(f"    was absorbed by alternative ports within April 2024.")

# Recovery timeline from YoY
print("\n  Recovery timeline (YoY vs Apr 2024 baseline):")
for dt in ["2024-04-01","2024-05-01","2024-06-01","2024-07-01","2024-08-01","2024-09-01"]:
    prod_val = master.loc[dt, "auto_prod"]
    yoy_m = prod.loc[dt] / prod.loc[dt.replace("2024","2023")] - 1
    print(f"    {dt}:  prod={prod_val:.1f}k  YoY={yoy_m*100:+.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6E: Policy-relevant minimum safety stock calculation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6E — Minimum safety stock to prevent production decline")
print("=" * 70)

target_drops = [0.0, 0.05, 0.10]  # target maximum production decline
print(f"\nFor θ_central = {theta_central:.2f}, D_closure = {D_closure} days:")
print(f"{'Target max decline':>22}  {'Required SS (days)':>22}")
print("-" * 48)
for target in target_drops:
    required_buf = 1.0 - (target / theta_central)
    required_ss  = required_buf * D_closure
    print(f"  Δ ≤ {target*100:4.0f}%             →  {required_ss:>8.1f} days-of-supply required")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
df_ss.to_csv(str(DATA_PROC / "safety_stock_results.csv"), index=False)
_ss_clean = {
    "theta_central":        theta_central,
    "theta_low":            theta_low,
    "theta_high":           theta_high,
    "ss_central_days":      ss_central,
    "ss_range_low_days":    min(ss_range_lo, ss_range_hi),
    "ss_range_high_days":   max(ss_range_lo, ss_range_hi),
    "buffer_fraction":      buf_central,
    "apr_drop_pct":         apr_drop_pct,
    "rerouting_surplus_pct":rerouting_surplus_pct,
}
pd.DataFrame(list(_ss_clean.items()), columns=["Parameter","Value"]).to_csv(
    str(DATA_PROC / "ss_summary.csv"), index=False)
print(f"\nSaved: safety_stock_results.csv + ss_summary.csv")
print("Step 6 — Safety Stock Inference complete ✓")
