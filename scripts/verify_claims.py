"""
Independent verification of every key claim in the paper.
Run from scratch against current data files.
"""
import pandas as pd, numpy as np, pickle
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from pathlib import Path
from scipy import stats

BASE = Path(__file__).parent.parent
TAB_MAIN = BASE/"paper_assets/tables/main"
TAB_SUPP = BASE/"paper_assets/tables/supplementary"
master = pd.read_csv(BASE/"data/processed/master.csv",
                     parse_dates=["date"], index_col="date")

MONTH_DUMMIES = [f"m{m:02d}" for m in range(2,13)]
results = {}
errors  = []

def check(claim, actual, expected, tol=0.01):
    ok = abs(float(actual) - float(expected)) <= tol
    results[claim] = ("✓" if ok else "✗", actual, expected)
    if not ok:
        errors.append(f"{claim}: got {actual}, expected {expected}")

print("="*65)
print("INDEPENDENT VERIFICATION OF ALL KEY PAPER CLAIMS")
print("="*65)

# CLAIM 1: Baltimore port collapse -99.5%
balt_pre = master[master["post"]==0]["balt_import_vol"].mean()
balt_apr = master.loc["2024-04-01","balt_import_vol"]
balt_pct = (balt_apr/balt_pre - 1)*100
check("Baltimore collapse %", balt_pct, -99.5, tol=0.1)

# CLAIM 2: Production YoY -27.3%
p24 = master.loc["2024-04-01","auto_prod"]
p23 = master.loc["2023-04-01","auto_prod"]
yoy_prod = (p24/p23 - 1)*100
check("Production YoY Apr 2024 %", yoy_prod, -27.3, tol=0.1)

# CLAIM 3: HS8708 imports +8.2% YoY
h24 = master.loc["2024-04-01","hs8708_cif_B"]
h23 = master.loc["2023-04-01","hs8708_cif_B"]
yoy_hs = (h24/h23 - 1)*100
check("HS8708 YoY Apr 2024 %", yoy_hs, 8.2, tol=0.1)

# CLAIM 4: Safety stock ~17.6 days (theta_central = mean of the 2 EMPIRICAL
# methods only, since 2026-07-19 -- theta_lit, an undisclosed-source author
# judgment, was removed from the central-estimate average; it previously
# pooled equally with real data and pulled theta_central down to 0.572 /
# ss_central to 15.7d. theta_lit still informs the low end of the range.)
ss = pd.read_csv(BASE/"data/processed/ss_summary.csv")
ss_d = dict(zip(ss["Parameter"], ss["Value"].astype(float)))
check("Safety stock central (days)", ss_d["ss_central_days"], 17.57, tol=0.1)
check("SS theta central", ss_d["theta_central"], 0.6584, tol=0.002)
check("SS range low (days)", ss_d["ss_range_low_days"], 7.3, tol=0.1)
check("SS range high (days)", ss_d["ss_range_high_days"], 19.0, tol=0.1)
# Verify arithmetic
buf_check = 1 - ss_d["apr_drop_pct"] / ss_d["theta_central"]
ss_check  = buf_check * 30
check("SS arithmetic verify (days)", ss_check, ss_d["ss_central_days"], tol=0.05)

# CLAIM 5: Brunswick +76.7%
t5 = pd.read_csv(TAB_MAIN/"table4_diversion.csv")
brun = t5[t5["Port"]=="Brunswick (GA)"]["Pct_change"].values[0]
check("Brunswick surge %", brun, 76.7, tol=0.1)
brun_p = t5[t5["Port"]=="Brunswick (GA)"]["p_value"].values[0]
brun_d = t5[t5["Port"]=="Brunswick (GA)"]["Cohen_d"].values[0]
check("Brunswick p-value < 0.001", float(brun_p < 0.001), 1.0, tol=0.0)
# Cohen's d with the correct dof-weighted pooled SD for unequal n (n_pre=33
# vs n_post=6); a naive unweighted average-of-variances formula previously
# gave 4.54, a ~20% overstatement.
check("Brunswick Cohen d ~3.79", abs(brun_d), 3.79, tol=0.05)

# CLAIM 6: Market share shift (Baltimore vs Brunswick only, since 2026-07-19
# -- Norfolk/Port of Virginia dropped from the headline diversion analysis,
# confirmed CV>1 noise; shares now sum to 100% over the 2-port dyad instead
# of 4 ports, so they shifted slightly from the old 63.5/48.0/35.0/51.5 figures)
check("Table 5 has exactly 2 ports (Baltimore, Brunswick only)", len(t5), 2, tol=0)
balt_pre_share = t5[t5["Port"]=="Baltimore"]["Pre_share_pct"].values[0]
balt_post_share= t5[t5["Port"]=="Baltimore"]["Post_share_pct"].values[0]
brun_pre_share = t5[t5["Port"]=="Brunswick (GA)"]["Pre_share_pct"].values[0]
brun_post_share= t5[t5["Port"]=="Brunswick (GA)"]["Post_share_pct"].values[0]
check("Baltimore pre share %",  balt_pre_share,  64.5, tol=0.2)
check("Baltimore post share %", balt_post_share, 48.2, tol=0.2)
check("Brunswick pre share %",  brun_pre_share,  35.5, tol=0.2)
check("Brunswick post share %", brun_post_share, 51.8, tol=0.2)

# CLAIM 7: False-break F-statistic = 11.01
with open(BASE/"data/processed/robustness.pkl","rb") as f:
    rob = pickle.load(f)
check("False-break F-stat", rob["true_F"], 11.01, tol=0.05)

# CLAIM 8: ITS Model A β₂ for PPI and auto_prod
with open(BASE/"data/processed/its_results.pkl","rb") as f:
    its = pickle.load(f)

for outcome, b2_exp, p2_exp, label in [
    ("ln_auto_prod",    0.126, 0.424, "AutoProd Model A β₂"),
    ("ppi_mv_parts",   -1.279, 0.128, "PPI Model A β₂"),
    ("ln_hs8708_cif_B", 0.019, 0.541, "HS8708 Model A β₂"),
]:
    fit, cn, _ = its["model_a"][outcome]
    b2 = fit.params[cn.index("post")]
    p2 = fit.pvalues[cn.index("post")]
    check(f"{label} estimate", b2, b2_exp, tol=0.002)
    check(f"{label} p-value",  p2, p2_exp, tol=0.002)

# CLAIM 9: AR-augmented PPI β₂ = -0.388, p=0.007
t2 = pd.read_csv(TAB_MAIN/"table2_its_results.csv")
ppi_ar = t2[(t2["Model"]=="C: AR(1)-augmented ITS (HAC-4)") &
             (t2["Outcome"]=="PPI: MV Parts [PRICE]") &
             (t2["Coefficient"]=="beta2 level shift")]
check("PPI AR-augmented β₂", ppi_ar["Estimate"].values[0], -0.388, tol=0.002)
check("PPI AR-augmented p",  ppi_ar["p_value"].values[0],   0.007, tol=0.002)

# CLAIM 10: Lost passenger cars (5-month cumulative, Apr-Aug 2024) = 162,018
# DAUPSA = "Thousands of Units, Seasonally Adjusted" (monthly, NOT SAAR) per
# FRED's own series page -- actual monthly units = raw_value * 1,000.
w = pd.read_csv(TAB_MAIN/"table5_welfare_estimate.csv")
wd = dict(zip(w["Parameter"], w["Value"]))
check("Lost passenger cars", float(str(wd["lost_passenger_cars"]).replace(",","")),
      162018, tol=50)
check("Production loss $B", float(wd["production_loss_passenger_B"]), 7.777, tol=0.02)

# CLAIM 11: Event study HS8708 pre-trend flat (0 sig pre-period coefs)
with open(BASE/"data/processed/event_study.pkl","rb") as f:
    _ev_pkl = pickle.load(f)
ev = _ev_pkl["outcomes"]
pretrend_tests = _ev_pkl["pretrend_tests"]
hs_coefs = ev["ln_hs8708_cif_B"]["coefs"]
pre_sig = sum(1 for k,v in hs_coefs.items() if k < -1 and v["pval"] < 0.05)
check("HS8708 pre-period sig coefs", pre_sig, 0, tol=0)

# CLAIM 13: Pre-trend joint F-test (H0: flat pre-trend, k=-12..-2) -- disclosed
# for all 4 outcomes, not just the ones that pass. HS8708 and PPI parts are
# flat (supports parallel trends); auto_prod (primary outcome) and the
# petrochem placebo are NOT flat and must be treated as a limitation.
check("HS8708 pre-trend flat (F-test)",   float(pretrend_tests["ln_hs8708_cif_B"]["flat"]), 1.0, tol=0)
check("PPI parts pre-trend flat (F-test)", float(pretrend_tests["ppi_mv_parts"]["flat"]),    1.0, tol=0)
check("AutoProd pre-trend NOT flat (F-test)", float(pretrend_tests["ln_auto_prod"]["flat"]), 0.0, tol=0)
check("Petrochem placebo pre-trend NOT flat (F-test)", float(pretrend_tests["ppi_petrochem"]["flat"]), 0.0, tol=0)
check("AutoProd pre-trend F-stat", pretrend_tests["ln_auto_prod"]["F"], 6.01, tol=0.05)
check("Petrochem placebo pre-trend F-stat", pretrend_tests["ppi_petrochem"]["F"], 2.32, tol=0.05)

# CLAIM 10b: April-2024-only lost passenger cars (the figure quoted in the
# manuscript's Introduction and Section 5.2, distinct from CLAIM 10's 5-month
# Apr-Aug cumulative figure). DAUPSA_apr23=151.615, DAUPSA_apr24=110.269
# (thousands of units/month); lost = (151.615-110.269)*1000 = 41,346.
apr23_prod = master.loc["2023-04-01","auto_prod"]
apr24_prod = master.loc["2024-04-01","auto_prod"]
april_only_lost = (apr23_prod - apr24_prod) * 1000
check("April-2024-only lost passenger cars", april_only_lost, 41346, tol=5)

# CLAIM 11b: Inventory carrying cost of the implied JIT buffer (values shift
# from the theta_central fix above -- larger implied buffer -> larger buffer
# value, carrying cost, avoided loss, and break-even years, all consistently)
cc = pd.read_csv(BASE/"data/processed/carrying_cost_results.csv")
cc_d = dict(zip(cc["Parameter"], cc["Value"].astype(float)))
check("Buffer value, central ($B)", cc_d["buffer_value_central_B"], 4.2631, tol=0.01)
check("Annual carrying cost, central ($B/yr)", cc_d["annual_carry_cost_central_B"], 0.9592, tol=0.01)
check("Avoided loss, passenger-car ($B)", cc_d["avoided_value_passenger_B"], 2.8067, tol=0.01)
check("Break-even years, passenger-car scope", cc_d["breakeven_years_observed_passenger"], 2.926, tol=0.01)
check("Break-even years, full-fleet scope", cc_d["breakeven_years_observed_fullfleet"], 11.704, tol=0.01)
# Verify the analytical identity: observed-buffer and incremental-buffer break-even
# years are mathematically forced to be equal (Section 4.3's identity is linear
# in the shortfall fraction) — confirms this is a model property, not a bug.
check("Break-even identity (observed == incremental, passenger)",
      cc_d["breakeven_years_observed_passenger"], cc_d["breakeven_years_incremental_passenger"], tol=0.001)

# CLAIM 11c: PPI first-differenced robustness check (non-stationarity concern
# on the paper's ONE significant coefficient, Model C beta2=-0.388,p=0.007).
# This is disclosed honestly in the manuscript: the effect does NOT survive
# first-differencing (beta2=-0.159, p=0.234, ns), so it is presented as
# suggestive/specification-dependent, not a robust, settled finding.
t3 = pd.read_csv(TAB_SUPP/"tableS1_robustness.csv")
ppi_fd = t3[t3["Specification"]=="PPI first-differenced spec (non-stationarity robustness)"]
check("PPI first-differenced beta2", ppi_fd["beta2"].values[0], -0.15933, tol=0.001)
check("PPI first-differenced p", ppi_fd["beta2_p"].values[0], 0.2335, tol=0.001)

# CLAIM 12: Unit root — auto_prod stationary, PPI non-stationary
t3b = pd.read_csv(TAB_SUPP/"tableS2_unit_root_tests.csv")
auto_stat = t3b[t3b["Series"]=="ln(Auto Production)"]["conclusion"].values[0]
ppi_stat  = t3b[t3b["Series"]=="PPI: MV Parts"]["conclusion"].values[0]
check("AutoProd stationary", float(auto_stat=="stationary"), 1.0, tol=0)
check("PPI non-stationary",  float(ppi_stat=="non-stationary"), 1.0, tol=0)

# CLAIM 13: Multiple-testing correction (Table 2b, Benjamini-Hochberg FDR on
# the beta2 coefficient family across 4 outcomes x 3 models = 12 tests).
# The paper's ONLY nominally-significant beta2 (PPI, Model C, p=0.007) does
# NOT survive FDR correction -- this must stay disclosed as 0 survivors, not
# silently reported as "1 significant finding" without the correction.
t2b = pd.read_csv(TAB_MAIN/"table3_multiple_testing_correction.csv")
check("Beta2 family size (4 outcomes x 3 models)", len(t2b), 12, tol=0)
check("Beta2 raw-significant count (p<0.05)", int(t2b["Significant_raw_p05"].sum()), 1, tol=0)
check("Beta2 BH-FDR-significant count (q<0.05)", int(t2b["Significant_BH_q05"].sum()), 0, tol=0)

# CLAIM 14: March 2024 partial-treatment-month sensitivity (Table 3, Test 7).
# Collapse occurred day 26/31 (~81% of March pre-collapse) but Model A/B pool
# all of March into "post" -- this checks the primary beta2 estimate is not
# an artifact of that pooling (point estimate should stay non-significant
# and same-signed, not flip to significant when March is dropped).
march_excl = t3[t3["Specification"]=="March 2024 excluded (partial-treatment-month sensitivity)"]
check("March-excluded beta2", march_excl["beta2"].values[0], 0.0783, tol=0.002)
check("March-excluded beta2 p (still ns)", float(march_excl["beta2_p"].values[0] > 0.05), 1.0, tol=0)

# CLAIM 15: Golden Ray shipwreck baseline-confound sensitivity check for the
# Brunswick diversion "surge" figure (Table 5b / Step 8A2). The stricter
# pre-Golden-Ray-only baseline must give an EQUAL-OR-LARGER surge than the
# reported 76.7% -- confirming the headline figure is not inflated by using
# a wreck-suppressed baseline (if this ever flips, Table 5's headline number
# needs re-examination, not silent continued use).
t5b = pd.read_csv(TAB_SUPP/"tableS4_golden_ray_sensitivity.csv")
t5b_d = dict(zip(t5b["Check"], t5b["Value"]))
surge_full = float(str(t5b_d["Surge % using full 2019-2021 baseline (as reported in Table 4)"]).rstrip("%"))
surge_pregr = float(str(t5b_d["Surge % using pre-Golden-Ray-only baseline (sensitivity check)"]).rstrip("%"))
check("Golden Ray sensitivity: full-baseline surge %", surge_full, 76.7, tol=0.1)
check("Golden Ray check: pre-GR-only surge >= reported surge", float(surge_pregr >= surge_full), 1.0, tol=0)

# CLAIM 16: Welfare Channel 2 (renumbered from Channel 3 on 2026-07-19, when
# the old Channel 2 "rerouting logistics premium" was dropped entirely --
# unsourced $800-1,200/vehicle author guess, already excluded from every
# headline figure before removal) short-run/long-run wage-loss range.
w6 = pd.read_csv(TAB_MAIN/"table5_welfare_estimate.csv")
w6_d = dict(zip(w6["Parameter"], w6["Value"]))
check("logistics_premium keys fully removed from welfare table",
      float("logistics_premium_lo_B" not in w6_d and "logistics_premium_hi_B" not in w6_d), 1.0, tol=0)
check("Wage loss, short-run (elast=0.25), $B", float(w6_d["wage_loss_shortrun_B"]), 1.823, tol=0.02)
check("Wage loss, long-run (elast=0.80), $B",  float(w6_d["wage_loss_longrun_B"]),  5.834, tol=0.02)
check("Conservative total welfare estimate, $B", float(w6_d["conservative_total_B"]), 9.60, tol=0.05)
check("Liberal total welfare estimate, $B",      float(w6_d["liberal_total_B"]),      13.61, tol=0.05)

# CLAIM 17: dead ppi_mv_parts_wpu derivation removed from master.csv (2026-07-19)
check("ppi_mv_parts_wpu removed from master.csv", float("ppi_mv_parts_wpu" not in master.columns), 1.0, tol=0)

# CLAIM 18: paper_assets/ output structure is complete after the 2026-07-19
# visual-design pass (7 main + 6 supplementary figures, each PDF+PNG; 7 main
# + 6 supplementary tables, each CSV+TEX) -- catches a script silently
# failing to save into the new structure.
FIG_MAIN = BASE/"paper_assets/figures/main"
FIG_SUPP = BASE/"paper_assets/figures/supplementary"
check("Main figures: 7 PDFs", len(list(FIG_MAIN.glob("*.pdf"))), 7, tol=0)
check("Main figures: 7 PNGs", len(list(FIG_MAIN.glob("*.png"))), 7, tol=0)
check("Supplementary figures: 6 PDFs", len(list(FIG_SUPP.glob("*.pdf"))), 6, tol=0)
check("Supplementary figures: 6 PNGs", len(list(FIG_SUPP.glob("*.png"))), 6, tol=0)
check("Main tables: 7 CSVs", len(list(TAB_MAIN.glob("*.csv"))), 7, tol=0)
check("Main tables: 7 TEX", len(list(TAB_MAIN.glob("*.tex"))), 7, tol=0)
check("Supplementary tables: 6 CSVs", len(list(TAB_SUPP.glob("*.csv"))), 6, tol=0)
check("Supplementary tables: 6 TEX", len(list(TAB_SUPP.glob("*.tex"))), 6, tol=0)

# ── PRINT RESULTS ─────────────────────────────────────────────────────────────
print()
all_pass = True
for claim, (status, actual, expected) in results.items():
    if status == "✗":
        all_pass = False
    print(f"  {status}  {claim:45s}  got={actual:.4g}  exp={expected:.4g}")

print()
if all_pass:
    print(f"  ALL {len(results)} CLAIMS VERIFIED ✓")
    print()
    print("  The paper's key claims are reproducible from the current codebase.")
else:
    print(f"  FAILURES: {len(errors)}")
    for e in errors:
        print(f"    ✗ {e}")
