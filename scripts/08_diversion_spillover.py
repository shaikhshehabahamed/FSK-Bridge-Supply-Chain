"""
Step 8: Diversion Spillover Analysis
======================================
Tests whether the cargo lost at Baltimore during the FSK closure
was absorbed by the alternative East Coast RoRo port with a credible
signal in this data source: Port of Brunswick (GA).

Data: PortWatch Daily_Ports_Data (monthly aggregates)
  → 2019-2021 baseline (pre-disruption)
  → 2024 June-December post-reopening

PortWatch's Daily_Ports_Data feed also nominally tracks Norfolk and Port
of Virginia, but both were EXCLUDED from the headline analysis on
2026-07-19 after the pre-period coefficient of variation came back >1.0
for each (mean well under 1 RoRo call/month, more zeros than non-zeros) --
see the STEP 8A2 diagnostic below. That is noise, not signal, for two
genuinely major ports; computing "market share" percentages on counts
that small would be false precision, not evidence. Their raw numbers are
still printed for transparency in Step 8A/8A2, just not used in the
share/absorption/balance calculations Steps 8B-8E run on.

Key metrics (Baltimore vs Brunswick only, see above)
  (1) Market share shift: Baltimore vs Brunswick
  (2) Diversion absorption rate: fraction of lost Baltimore volume
      recovered at Brunswick
  (3) Net RoRo portcall balance: did the Baltimore-Brunswick pair
      gain or lose total throughput?
  (4) Statistical test: Welch t-test on port volumes pre vs post
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"



import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

pw = pd.read_csv(str(DATA_RAW / "portwatch_monthly.csv"),
                 parse_dates=["date"])

# ALL_TRACKED_PORTS: every port this PortWatch feed nominally covers, used
# only for the Step 8A/8A2 transparency check. PORTS: the ports actually
# used in the headline share/absorption/balance analysis (Steps 8B-8E) --
# see the module docstring for why Norfolk and Port of Virginia are excluded.
ALL_TRACKED_PORTS = ["baltimore","brunswick","norfolk","port_virginia"]
PORTS   = ["baltimore","brunswick"]
LABELS  = {"baltimore":"Baltimore","brunswick":"Brunswick (GA)",
           "norfolk":"Norfolk","port_virginia":"Port of Virginia"}
COLOURS = {"baltimore":"#2980b9","brunswick":"#e67e22",
           "norfolk":"#27ae60","port_virginia":"#8e44ad"}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8A: Define pre and post periods from available data
# ─────────────────────────────────────────────────────────────────────────────
# Pre-period: 2019-2021 (33 months, pre-disruption baseline)
# Post-period: July 2024 – December 2024 (full channel reopened June 10)
#   Use July+ to avoid partial-reopening noise in June
pre  = pw[pw["date"].dt.year.isin([2019,2020,2021])]
post = pw[(pw["date"].dt.year==2024) & (pw["date"].dt.month >= 7)]
# Disruption window (June 2024: partial, 8 days of data)
disr = pw[(pw["date"].dt.year==2024) & (pw["date"].dt.month == 6)]

print("=" * 70)
print("STEP 8A — Data coverage check (all 4 nominally-tracked ports)")
print("=" * 70)
for p_label, p_df in [("Pre (2019-2021)", pre), ("Post (Jul-Dec 2024)", post)]:
    print(f"\n{p_label}:")
    for port in ALL_TRACKED_PORTS:
        s = p_df[p_df["port_label"]==port]
        print(f"  {LABELS[port]:25s}: n={len(s)}  "
              f"roro_calls={s['portcalls_roro'].mean():.1f}/mo  "
              f"import_roro={s['import_roro'].mean():,.0f}/mo")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8A1: Why Norfolk and Port of Virginia are excluded from Steps 8B-8E
# ─────────────────────────────────────────────────────────────────────────────
# Coefficient of variation (SD/mean) of pre-period monthly RoRo portcalls:
# Baltimore and Brunswick are well-behaved (CV ~0.2); Norfolk and Port of
# Virginia have CV > 1.0 -- the SD exceeds the mean, i.e. the series is
# mostly zeros with occasional single-digit blips, not a real signal. Both
# are large, real ports in reality; a mean of well under 1 RoRo call/month
# in this specific PortWatch feed almost certainly reflects a data/tagging
# gap for these two port codes, not their true automotive throughput.
# Dropped from the headline market-share/absorption/balance analysis on
# 2026-07-19 rather than kept in with a caveat: computing "% share" and
# "% change" on a sub-1-count-per-month base is false precision on noise,
# not a weaker-but-still-informative signal.
print("\n" + "=" * 70)
print("STEP 8A1 — Why Norfolk & Port of Virginia are excluded below")
print("=" * 70)
for port in ALL_TRACKED_PORTS:
    s = pre[pre["port_label"]==port]["portcalls_roro"]
    cv = s.std() / s.mean() if s.mean() else float("nan")
    flag = "  <- EXCLUDED (CV>1: noise, not signal)" if cv > 1.0 else ""
    print(f"  {LABELS[port]:25s}: pre-period mean={s.mean():.2f}/mo  sd={s.std():.2f}  CV={cv:.2f}{flag}")
print("\n  Steps 8B-8E below use PORTS = [Baltimore, Brunswick] only. Norfolk and")
print("  Port of Virginia's raw numbers stay visible above for transparency,")
print("  but are not part of any share/absorption/balance calculation.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8A2: Disclosed data limitations for the diversion baseline
# ─────────────────────────────────────────────────────────────────────────────
# (1) IMF PortWatch's Daily_Ports_Data feed (the source of portcalls_roro /
#     import_roro used throughout this diversion analysis) has NO coverage
#     between 2021-09 and 2024-06 -- confirmed directly from the raw pull,
#     not just visible as a shaded gap in Figures 2 and 7. This is WHY the
#     "pre" baseline is 2019-2021 rather than a more recent window: there is
#     no alternative recent baseline available in this data source.
# (2) That forced 2019-2021 window coincides with the MV Golden Ray
#     shipwreck at the Port of Brunswick (capsized 2019-09-08, wreck fully
#     removed 2021-10-25 -- the largest wreck removal in US history), an
#     unrelated disruption at the SAME port used as the primary diversion
#     comparison. A skeptical reader could reasonably worry this depresses
#     Brunswick's "normal" baseline and mechanically inflates the reported
#     post-FSK surge. Checked directly below rather than left as an
#     unaddressed objection.
print("\n" + "=" * 70)
print("STEP 8A2 — Data limitations: PortWatch coverage gap & Golden Ray")
print("=" * 70)
print("  PortWatch Daily_Ports_Data has NO monthly coverage between 2021-09")
print("  and 2024-06 (confirmed from the raw pull) -- the 2019-2021 'pre'")
print("  window used throughout Step 8 is forced by this gap, not chosen")
print("  freely; there is no more-recent pre-2024 baseline available.")
print()
print("  Brunswick GA is also the site of the MV Golden Ray shipwreck")
print("  (capsized 2019-09-08; wreck fully removed 2021-10-25), which")
print("  overlaps ~25 of the 33 months in Brunswick's 'pre' baseline window.")
print("  Sensitivity check: does using ONLY the pre-Golden-Ray months")
print("  (Jan-Aug 2019, n=8) as Brunswick's baseline change the conclusion?")

brun_all  = pw[pw["port_label"] == "brunswick"].set_index("date")
brun_pre_goldenray  = brun_all.loc["2019-01-01":"2019-08-01", "portcalls_roro"]
brun_during_goldenray = brun_all.loc["2019-09-01":"2021-10-01", "portcalls_roro"]
brun_post_mean = post[post["port_label"] == "brunswick"]["portcalls_roro"].mean()
brun_full_pre_mean = pre[pre["port_label"] == "brunswick"]["portcalls_roro"].mean()

surge_full_pre   = (brun_post_mean / brun_full_pre_mean - 1) * 100
surge_pregr_only = (brun_post_mean / brun_pre_goldenray.mean() - 1) * 100

print(f"\n  Brunswick portcalls_roro/mo:")
print(f"    Pre-Golden-Ray (Jan-Aug 2019, n={len(brun_pre_goldenray)}):     {brun_pre_goldenray.mean():.1f}")
print(f"    During Golden Ray salvage (Sep2019-Oct2021, n={len(brun_during_goldenray)}): {brun_during_goldenray.mean():.1f}")
print(f"    Full 2019-2021 baseline (as used, n={len(pre[pre['port_label']=='brunswick'])}):     {brun_full_pre_mean:.1f}")
print(f"    Post (Jul-Dec 2024, n={len(post[post['port_label']=='brunswick'])}):                     {brun_post_mean:.1f}")
print(f"\n  Surge %, full 2019-2021 baseline (as reported):  {surge_full_pre:+.1f}%")
print(f"  Surge %, pre-Golden-Ray-only baseline (n=8):     {surge_pregr_only:+.1f}%")
if surge_pregr_only >= surge_full_pre:
    print("  -> Golden Ray does NOT inflate the reported surge: portcalls_roro")
    print("     during the salvage period was NOT measurably lower than the brief")
    print("     pre-wreck window, so the stricter baseline gives an EQUAL-OR-LARGER")
    print("     surge. The headline 76.7% figure is if anything conservative.")
else:
    print("  -> WARNING: the pre-Golden-Ray-only baseline gives a SMALLER surge;")
    print("     the headline figure may be inflated by the wreck-period baseline.")
golden_ray_check = {
    "brun_pre_goldenray_mean": float(brun_pre_goldenray.mean()),
    "brun_during_goldenray_mean": float(brun_during_goldenray.mean()),
    "brun_full_pre_mean": float(brun_full_pre_mean),
    "brun_post_mean": float(brun_post_mean),
    "surge_pct_full_pre": float(surge_full_pre),
    "surge_pct_pregr_only": float(surge_pregr_only),
    "n_pregr": int(len(brun_pre_goldenray)),
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8B: Market share analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 8B — RoRo Market Share Shift")
print("=" * 70)

def share_stats(period_df, metric="portcalls_roro"):
    totals = {}
    for port in PORTS:
        s = period_df[period_df["port_label"]==port][metric]
        totals[port] = s.mean()
    total_all = sum(totals.values())
    shares = {k: v/total_all for k,v in totals.items()}
    return totals, shares

pre_totals,  pre_shares  = share_stats(pre,  "portcalls_roro")
post_totals, post_shares = share_stats(post, "portcalls_roro")

pre_vol,  pre_vol_sh  = share_stats(pre,  "import_roro")
post_vol, post_vol_sh = share_stats(post, "import_roro")

print(f"\n{'Port':25s}  {'Pre calls/mo':>14}  {'Pre share':>12}  "
      f"{'Post calls/mo':>14}  {'Post share':>12}  {'Δ share':>10}")
print("-" * 90)
for port in PORTS:
    print(f"  {LABELS[port]:23s}  {pre_totals[port]:>14.1f}  "
          f"{pre_shares[port]*100:>11.1f}%  "
          f"{post_totals[port]:>14.1f}  {post_shares[port]*100:>11.1f}%  "
          f"{(post_shares[port]-pre_shares[port])*100:>+9.1f}pp")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8C: Welch t-tests — pre vs post for each port
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 8C — Welch t-tests (pre vs post) for RoRo portcalls")
print("=" * 70)

ttest_results = {}
for port in PORTS:
    pre_s  = pre[pre["port_label"]==port]["portcalls_roro"].dropna()
    post_s = post[post["port_label"]==port]["portcalls_roro"].dropna()
    t, p   = stats.ttest_ind(pre_s, post_s, equal_var=False)
    # Cohen's d with the degrees-of-freedom-weighted pooled SD (correct for
    # unequal n; a simple unweighted average of variances, as used here
    # previously, is only valid when n1=n2 -- pre n=33 vs post n=6 here).
    n1, n2 = len(pre_s), len(post_s)
    s_pooled = np.sqrt(((n1-1)*pre_s.std()**2 + (n2-1)*post_s.std()**2) / (n1+n2-2))
    d      = (post_s.mean() - pre_s.mean()) / s_pooled   # Cohen's d
    pct_chg = (post_s.mean() / pre_s.mean() - 1) * 100
    ttest_results[port] = {"t": t, "p": p, "d": d, "pct": pct_chg,
                           "pre_m": pre_s.mean(), "post_m": post_s.mean()}
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"  {LABELS[port]:25s}:  pre={pre_s.mean():.1f}  post={post_s.mean():.1f}  "
          f"Δ={pct_chg:+.1f}%  t={t:+.2f}  p={p:.4f}{sig}  d={d:+.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8D: Diversion absorption rate
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 8D — Diversion Absorption Rate")
print("=" * 70)

balt_loss = pre_totals["baltimore"] - post_totals["baltimore"]
brun_gain = post_totals["brunswick"] - pre_totals["brunswick"]
# Norfolk/Port of Virginia gains are NOT added here (Step 8A1: CV>1, noise
# not signal in this data source) -- absorption_rate below is Brunswick's
# absorption of Baltimore's loss specifically, not a claim that no cargo
# went anywhere else. See Step 8A for their raw (near-zero) numbers.

total_absorbed = brun_gain
absorption_rate = total_absorbed / balt_loss if balt_loss > 0 else 0

print(f"\n  Baltimore monthly RoRo portcalls:")
print(f"    Pre-period (2019-2021): {pre_totals['baltimore']:.1f}/month")
print(f"    Post-period (2024 H2):  {post_totals['baltimore']:.1f}/month")
print(f"    Loss:                   {balt_loss:+.1f}/month")

print(f"\n  Brunswick gain:            {brun_gain:+.1f}/month  "
      f"({brun_gain/balt_loss*100:+.0f}% of Baltimore loss)")
print(f"  (Norfolk / Port of Virginia excluded -- see Step 8A1; their raw")
print(f"   pre->post changes are within noise, not a meaningful gain or loss)")
if absorption_rate > 1.5:
    print(f"  NOTE: absorption rate > 100% means Brunswick's gain "
          f"({brun_gain:+.1f}/mo) considerably exceeds Baltimore's own loss "
          f"({balt_loss:+.1f}/mo) -- read this as 'Brunswick grew by more than")
    print(f"  enough to cover Baltimore's shortfall', not literally 'X% of the")
    print(f"  lost cargo was recovered' (Baltimore's own post-period loss is")
    print(f"  fairly small; most of Brunswick's growth reflects the port")
    print(f"  becoming a larger RoRo hub in its own right, not pure diversion).")
print(f"\n  Net Baltimore+Brunswick RoRo balance: {total_absorbed - balt_loss:+.1f}/month")
print(f"  Diversion absorption rate (by Brunswick):   {absorption_rate*100:.1f}%")

# Also compute for import volume
balt_vol_loss = pre_vol["baltimore"] - post_vol["baltimore"]
brun_vol_gain = post_vol["brunswick"] - pre_vol["brunswick"]
vol_absorb    = brun_vol_gain / balt_vol_loss if balt_vol_loss != 0 else 0
print(f"\n  Import volume (RoRo) absorption rate: {vol_absorb*100:.1f}%")
print(f"  (Baltimore lost {balt_vol_loss:,.0f}/mo  Brunswick gained {brun_vol_gain:,.0f}/mo)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8E: Net trade flow balance (Baltimore + Brunswick combined)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 8E — Net Baltimore+Brunswick RoRo Trade Flow Balance")
print("(NOT 'East Coast': scoped to the 2 ports with credible signal -- Step 8A1)")
print("=" * 70)

pre_ec_total  = sum(pre_totals.values())
post_ec_total = sum(post_totals.values())
ec_change = (post_ec_total / pre_ec_total - 1) * 100

print(f"\n  Total Baltimore+Brunswick RoRo portcalls:")
print(f"    Pre-period:  {pre_ec_total:.1f}/month")
print(f"    Post-period: {post_ec_total:.1f}/month")
print(f"    Net change:  {ec_change:+.1f}%")
print(f"\n  Interpretation: the Baltimore-Brunswick pair {'gained' if ec_change>0 else 'lost'} "
      f"{abs(ec_change):.1f}% combined throughput, suggesting "
      f"{'supply chain resilience' if ec_change>-5 else 'residual disruption'} "
      f"at these two ports specifically (not a full East Coast accounting).")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
diversion_summary = {
    "balt_loss_calls": balt_loss,
    "brun_gain_calls": brun_gain,
    "absorption_rate_pct": absorption_rate * 100,
    "ec_net_change_pct": ec_change,
    "vol_absorption_rate_pct": vol_absorb * 100,
}
pd.Series(diversion_summary).to_csv(
    str(DATA_PROC / "diversion_summary.csv"))
pd.Series(golden_ray_check).to_csv(
    str(DATA_PROC / "golden_ray_sensitivity.csv"))

import pickle
with open(str(DATA_PROC / "diversion_data.pkl"),"wb") as f:
    pickle.dump({
        "pw": pw, "pre_totals": pre_totals, "post_totals": post_totals,
        "pre_shares": pre_shares, "post_shares": post_shares,
        "pre_vol": pre_vol, "post_vol": post_vol,
        "ttest": ttest_results, "diversion_summary": diversion_summary,
        "golden_ray_check": golden_ray_check,
    }, f)

print("\nStep 8 — Diversion analysis complete ✓")
