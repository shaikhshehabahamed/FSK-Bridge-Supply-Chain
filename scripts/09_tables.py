"""
Step 9: Generate all publication-ready result tables (CSV + LaTeX).
Run independently after all previous steps are complete.

Tables are classified main vs. supplementary and numbered accordingly
(Table 1-7 main; Table S1-S6 supplementary), mirroring the figure
classification in Steps 4b/5b/6b/7b/8b/8d/10/11b. Saved to
paper_assets/tables/{main,supplementary}/ as both .csv (data) and .tex
(ready for \\input{} in the manuscript).
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from _style import TAB_MAIN, TAB_SUPP

ROOT   = Path(__file__).parent.parent


def save_table(df, stem, category, caption=""):
    """Save one table to paper_assets/tables/{main,supplementary}/ as .csv + .tex.
    Numeric columns are pre-rounded (round(x, N)) at construction time in every
    caller below; float_format="%.4g" here just makes to_latex() *display*
    that already-chosen precision instead of padding with trailing zeros."""
    target_dir = TAB_MAIN if category == "main" else TAB_SUPP
    csv_path = target_dir / f"{stem}.csv"
    tex_path = target_dir / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=True, float_format="%.4g",
                            caption=caption or stem.replace("_", " "),
                            label=f"tab:{stem}"))
    print(f"  Saved {stem}: .csv + .tex -> {target_dir.name}/")


master  = pd.read_csv(ROOT/"data/processed/master.csv",
                      parse_dates=["date"], index_col="date")
PRE  = master[master["post"]==0]
DISR = master[master["disruption"]==1]
POST = master[(master["post"]==1) & (master["disruption"]==0)]

# TABLE 1 (main) — Descriptive statistics
rows = []
for col, label, unit in [
    ("auto_prod",      "Domestic Auto Production (DAUPSA)", "Thousand vehicles/month, SA"),
    ("ppi_mv_parts",   "PPI: Motor Vehicle Parts Mfg",      "Index Dec2003=100"),
    ("balt_import_vol","Baltimore Import Volume",            "PortWatch units"),
    ("hs8708_cif_B",   "HS 8708 CIF Imports",               "USD billions/mo"),
    ("vehicle_sales",  "Total Vehicle Sales",               "M units SAAR/mo"),
    ("ppi_petrochem",  "PPI: Petrochemical Mfg [PLACEBO]",  "Index Dec2003=100"),
]:
    for phase, df in [("Pre (2019-2023)", PRE),
                      ("Disruption (Mar-May 2024)", DISR),
                      ("Recovery (Jun-Dec 2024)", POST)]:
        s = df[col].dropna()
        rows.append({
            "Series": label, "Unit": unit, "Phase": phase, "n": len(s),
            "Mean": round(s.mean(),4), "SD": round(s.std(),4),
            "Min": round(s.min(),4), "Max": round(s.max(),4),
            "Change_vs_Pre_pct": round((s.mean()/PRE[col].mean()-1)*100,2)
                                  if "Pre" not in phase else 0.0,
        })
save_table(pd.DataFrame(rows), "table1_descriptive_statistics", "main",
           "Descriptive statistics by study phase")

# TABLE 2 (main) — ITS results (Models A, B, C)
with open(ROOT/"data/processed/its_results.pkl","rb") as f:
    its = pickle.load(f)
import statsmodels.api as _sm
MONTH_DUMMIES_T2 = [f"m{m:02d}" for m in range(2,13)]

def _fmt_row2(outcome_label, coef_label, model_label, b, se, lo, hi, p, r2, n, caveat=""):
    return {"Outcome": outcome_label, "Model": model_label,
            "Coefficient": coef_label,
            "Estimate": round(b,5), "SE_HAC4": round(se,5),
            "CI_lo": round(lo,5), "CI_hi": round(hi,5),
            "p_value": round(p,4),
            "Sig": "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns",
            "R2_adj": round(r2,3), "n": int(n), "Caveat": caveat}

OUTCOMES_T2 = {
    "ln_auto_prod"   : "ln(Auto Production) [PRIMARY]",
    "ppi_mv_parts"   : "PPI: MV Parts [PRICE]",
    "ln_hs8708_cif_B": "ln(HS 8708 CIF) [TRADE]",
    "ppi_petrochem"  : "PPI: Petrochem [PLACEBO]",
}
# Every ITS point estimate below rests on a flat-pre-trend identifying
# assumption. Table S6's formal joint F-test shows that assumption FAILS for
# these two outcomes (auto production, the primary outcome, and the
# petrochem placebo) -- so their ITS coefficients are flagged in every table
# and figure they appear in, not just in the Figure 2 caption where this was
# previously disclosed only once and inconsistently.
OUTCOMES_T2_CAVEAT = {
    "ln_auto_prod": ("Pre-trend NOT flat (Table S6: F=6.01, p<0.001) -- ITS counterfactual "
                      "unreliable for this outcome; see Fig. 2 YoY comparison and Table 5/Sec 4.2 "
                      "for the estimates this paper actually relies on for production."),
    "ppi_petrochem": ("Pre-trend NOT flat (Table S6: F=2.32, p=0.027) -- placebo/falsification "
                       "check is imperfect; interpret the 'no placebo effect' result with this caveat."),
}
rows2 = []
for outcome, label in OUTCOMES_T2.items():
    caveat = OUTCOMES_T2_CAVEAT.get(outcome, "")
    # Model A
    fit, cn, _ = its["model_a"][outcome]
    ci = fit.conf_int()
    for key, klabel in [("t","beta1 pre-trend"),("post","beta2 level shift"),("t_post","beta3 slope")]:
        i = cn.index(key)
        rows2.append(_fmt_row2(label, klabel, "A: Two-phase ITS (HAC-4)",
            fit.params[i], fit.bse[i], ci[i,0], ci[i,1], fit.pvalues[i], fit.rsquared_adj, fit.nobs, caveat))
    # Model B
    fit3, cn3, _ = its["model_b"][outcome]
    ci3 = fit3.conf_int()
    for key, klabel in [("t","beta1 pre-trend"),("disruption","beta2 disruption level"),
                        ("t_post","beta3 slope change"),("recovery","beta4 recovery level")]:
        i = cn3.index(key)
        rows2.append(_fmt_row2(label, klabel, "B: Three-phase ITS (HAC-4)",
            fit3.params[i], fit3.bse[i], ci3[i,0], ci3[i,1], fit3.pvalues[i], fit3.rsquared_adj, fit3.nobs, caveat))
    # Model C: AR(1)-augmented
    df_ar = master[[outcome,"t","post","t_post","covid_shock","uaw_strike"]+MONTH_DUMMIES_T2].copy()
    df_ar["lag1"] = df_ar[outcome].shift(1)
    df_ar = df_ar.dropna()
    cols_ar = ["t","post","t_post","covid_shock","uaw_strike"]+MONTH_DUMMIES_T2+["lag1"]
    X_ar = _sm.add_constant(df_ar[cols_ar])
    cn_ar = ["const"]+cols_ar
    res_ar = _sm.OLS(df_ar[outcome].values, X_ar.values).fit(
        cov_type="HAC", cov_kwds={"maxlags":4,"use_correction":True})
    ci_ar = res_ar.conf_int()
    for key, klabel in [("t","beta1 pre-trend"),("post","beta2 level shift"),("t_post","beta3 slope")]:
        i = cn_ar.index(key)
        rows2.append(_fmt_row2(label, klabel, "C: AR(1)-augmented ITS (HAC-4)",
            res_ar.params[i], res_ar.bse[i], ci_ar[i,0], ci_ar[i,1],
            res_ar.pvalues[i], res_ar.rsquared_adj, res_ar.nobs, caveat))
save_table(pd.DataFrame(rows2), "table2_its_results", "main",
           "ITS segmented-regression results, Models A-C")

# TABLE 3 (main) — Multiple-testing correction (Benjamini-Hochberg FDR)
# ---------------------------------------------------------------------
# Table 2 reports 4 outcomes x 3 models x up to 4 coefficients each -- a
# large family of tests, and the paper's ENTIRE positive result (PPI,
# Model C, beta2 p=0.007) is a single nominally-significant p-value drawn
# from that family. Without a multiple-testing correction, a sharp referee
# will (rightly) ask whether that one "hit" is what you'd expect from
# chance alone. This applies Benjamini-Hochberg FDR control to the natural
# primary family: the beta2 "level shift / disruption level" coefficient
# (the paper's central causal quantity) across all outcome x model
# combinations in Table 2. beta1 (pre-trend, a nuisance/control term) and
# beta3/beta4 (secondary, already discussed as largely non-significant) are
# excluded from the family to avoid diluting power against a family that
# was never the basis of any headline claim.
from statsmodels.stats.multitest import multipletests
_beta2_rows = [r for r in rows2 if r["Coefficient"] in
               ("beta2 level shift", "beta2 disruption level")]
_p_raw = [r["p_value"] for r in _beta2_rows]
_reject05, _q05, _, _ = multipletests(_p_raw, alpha=0.05, method="fdr_bh")
_reject10, _q10, _, _ = multipletests(_p_raw, alpha=0.10, method="fdr_bh")
rows2b = []
for r, p, q, rej05, rej10 in zip(_beta2_rows, _p_raw, _q05, _reject05, _reject10):
    rows2b.append({
        "Outcome": r["Outcome"], "Model": r["Model"],
        "p_value_raw": round(p, 4),
        "q_value_BH": round(q, 4),
        "Significant_raw_p05": p < 0.05,
        "Significant_BH_q05": bool(rej05),
    })
save_table(pd.DataFrame(rows2b), "table3_multiple_testing_correction", "main",
           "Benjamini-Hochberg FDR correction across the beta2 coefficient family")
_n_raw_sig = sum(1 for p in _p_raw if p < 0.05)
_n_bh_sig  = int(sum(_reject05))
print(f"    -> {_n_raw_sig}/{len(_p_raw)} beta2 coefficients significant at raw p<0.05; "
      f"{_n_bh_sig}/{len(_p_raw)} survive Benjamini-Hochberg FDR correction at q<0.05.")

# TABLE S1 (supplementary) — Robustness (10 rows: 8 ITS specs + false-break + first-differenced)
with open(ROOT/"data/processed/robustness.pkl","rb") as f:
    rob = pickle.load(f)

_sig = lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
specs = [
    "Main (HAC-4, full sample, 2019-2024)",
    "HAC-2 lags","HAC-6 lags","HAC-8 lags",
    "COVID-excluded (2020-03 to 2021-06 dropped)",
    "Restricted pre-period (2021-01 to 2024-02)",
    "Levels specification (auto_prod, coefficient /100 for display)",
    "March 2024 excluded (partial-treatment-month sensitivity)",
]
rows3 = []
for spec, (lbl, b2, se2, p2, b3, se3, p3) in zip(specs, rob["rows"]):
    rows3.append({"Specification":spec,
                  "beta2":round(b2,5),"beta2_se":round(se2,5),"beta2_p":round(p2,4),"beta2_sig":_sig(p2),
                  "beta3":round(b3,5),"beta3_se":round(se3,5),"beta3_p":round(p3,4),"beta3_sig":_sig(p3)})
# False-break row -- percentile computed dynamically from the stored placebo
# F-stat distribution (rob["pct_rank"]) rather than a hardcoded "100th
# pctile" string, so this stays correct if the underlying data ever changes.
_pctile_label = f"F={rob['true_F']:.1f} ({rob['pct_rank']:.0f}th pctile of {len(rob['df_plac'])} placebo dates)"
rows3.append({"Specification":"False-break test (Baltimore vol, 48 placebo dates)",
              "beta2":round(rob["true_b2"],4),"beta2_se":"n/a","beta2_p":"n/a",
              "beta2_sig":_pctile_label,
              "beta3":"n/a","beta3_se":"n/a","beta3_p":"n/a","beta3_sig":"n/a"})
# First-differenced row
import statsmodels.api as _sm2
_df_fd = master[["ln_auto_prod","post","t_post","covid_shock","uaw_strike"]
                + [f"m{m:02d}" for m in range(2,13)]].copy()
_df_fd["dy"] = _df_fd["ln_auto_prod"].diff()
_df_fd = _df_fd.dropna()
_cols_fd = ["post","t_post","covid_shock","uaw_strike"] + [f"m{m:02d}" for m in range(2,13)]
_X_fd = _sm2.add_constant(_df_fd[_cols_fd])
_cn_fd = ["const"] + _cols_fd
_res_fd = _sm2.OLS(_df_fd["dy"].values, _X_fd.values).fit(
    cov_type="HAC", cov_kwds={"maxlags":4,"use_correction":True})
_b2f = _res_fd.params[_cn_fd.index("post")]
_se2f = _res_fd.bse[_cn_fd.index("post")]
_p2f = _res_fd.pvalues[_cn_fd.index("post")]
_b3f = _res_fd.params[_cn_fd.index("t_post")]
_se3f = _res_fd.bse[_cn_fd.index("t_post")]
_p3f = _res_fd.pvalues[_cn_fd.index("t_post")]
rows3.append({"Specification":"First-differenced spec (non-stationarity robustness)",
              "beta2":round(_b2f,5),"beta2_se":round(_se2f,5),"beta2_p":round(_p2f,4),"beta2_sig":_sig(_p2f),
              "beta3":round(_b3f,5),"beta3_se":round(_se3f,5),"beta3_p":round(_p3f,4),"beta3_sig":_sig(_p3f)})

# PPI first-differenced row (PPI: MV Parts is the series with the strongest
# non-stationarity signature -- Table S2 ADF p=0.72 -- AND the paper's one
# statistically significant ITS coefficient (Model C), so this outcome, not
# just auto_prod, needs its own explicit non-stationarity robustness check.
_df_fd_ppi = master[["ppi_mv_parts","post","t_post","covid_shock","uaw_strike"]
                     + [f"m{m:02d}" for m in range(2,13)]].copy()
_df_fd_ppi["dy"] = _df_fd_ppi["ppi_mv_parts"].diff()
_df_fd_ppi = _df_fd_ppi.dropna()
_X_fd_ppi = _sm2.add_constant(_df_fd_ppi[_cols_fd])
_res_fd_ppi = _sm2.OLS(_df_fd_ppi["dy"].values, _X_fd_ppi.values).fit(
    cov_type="HAC", cov_kwds={"maxlags":4,"use_correction":True})
_b2f_ppi = _res_fd_ppi.params[_cn_fd.index("post")]
_se2f_ppi = _res_fd_ppi.bse[_cn_fd.index("post")]
_p2f_ppi = _res_fd_ppi.pvalues[_cn_fd.index("post")]
_b3f_ppi = _res_fd_ppi.params[_cn_fd.index("t_post")]
_se3f_ppi = _res_fd_ppi.bse[_cn_fd.index("t_post")]
_p3f_ppi = _res_fd_ppi.pvalues[_cn_fd.index("t_post")]
rows3.append({"Specification":"PPI first-differenced spec (non-stationarity robustness)",
              "beta2":round(_b2f_ppi,5),"beta2_se":round(_se2f_ppi,5),"beta2_p":round(_p2f_ppi,4),"beta2_sig":_sig(_p2f_ppi),
              "beta3":round(_b3f_ppi,5),"beta3_se":round(_se3f_ppi,5),"beta3_p":round(_p3f_ppi,4),"beta3_sig":_sig(_p3f_ppi)})

save_table(pd.DataFrame(rows3), "tableS1_robustness", "supplementary",
           "Placebo, lag-sensitivity, and specification robustness checks")

# TABLE S2 (supplementary) — ADF unit root tests
from statsmodels.tsa.stattools import adfuller as _adf
_pre = master[master["post"]==0]
_adf_rows = []
for _col, _label in [
    ("ln_auto_prod","ln(Auto Production)"),("ppi_mv_parts","PPI: MV Parts"),
    ("ln_hs8708_cif_B","ln(HS 8708 CIF)"),("ppi_petrochem","PPI Petrochem [PLACEBO]"),
]:
    _s = _pre[_col].dropna()
    _stat,_pv,_lags,_nobs,_cvs,_ = _adf(_s, maxlag=4, autolag="AIC")
    _adf_rows.append({"Series":_label,"ADF_statistic":round(_stat,3),"p_value":round(_pv,4),
                      "critical_1pct":round(_cvs["1%"],3),"n_lags":_lags,"n_obs":_nobs,
                      "conclusion":"stationary" if _pv<0.05 else "non-stationary"})
save_table(pd.DataFrame(_adf_rows), "tableS2_unit_root_tests", "supplementary",
           "Augmented Dickey-Fuller unit-root tests, pre-period series")

# TABLE S3 (supplementary) — Safety-stock sensitivity grid
ss = pd.read_csv(ROOT/"data/processed/safety_stock_results.csv")
save_table(ss, "tableS3_safety_stock_grid", "supplementary",
           "Implied safety-stock sensitivity across the theta grid")

# TABLE 6 (main) — Safety-stock summary (compact)
ss_sum = pd.read_csv(ROOT/"data/processed/ss_summary.csv")
save_table(ss_sum, "table6_safety_stock_summary", "main",
           "Implied safety-stock inference, summary")

# TABLE 4 (main) — Diversion (Baltimore vs Brunswick only -- Norfolk and Port
# of Virginia were dropped from the headline diversion analysis on
# 2026-07-19; see Step 8A1 in 08_diversion_spillover.py for the CV>1 noise
# diagnostic)
with open(ROOT/"data/processed/diversion_data.pkl","rb") as f:
    div = pickle.load(f)
PORTS = ["baltimore","brunswick"]
LABS  = {"baltimore":"Baltimore","brunswick":"Brunswick (GA)",
         "norfolk":"Norfolk","port_virginia":"Port of Virginia"}
rows5 = []
for p in PORTS:
    t = div["ttest"][p]
    rows5.append({
        "Port": LABS[p],
        "Pre_calls_mo": round(div["pre_totals"][p],1),
        "Post_calls_mo": round(div["post_totals"][p],1),
        "Pct_change": round(t["pct"],1),
        "t_stat": round(t["t"],2), "p_value": round(t["p"],4),
        "Cohen_d": round(t["d"],2),
        "Pre_share_pct": round(div["pre_shares"][p]*100,1),
        "Post_share_pct": round(div["post_shares"][p]*100,1),
        "Share_change_pp": round((div["post_shares"][p]-div["pre_shares"][p])*100,1),
    })
save_table(pd.DataFrame(rows5), "table4_diversion", "main",
           "Port-network diversion: Baltimore vs. Brunswick")

# TABLE S4 (supplementary) — Golden Ray baseline sensitivity check for Brunswick
gr = div["golden_ray_check"]
rows5b = [{
    "Check": "Brunswick RoRo portcalls/mo, pre-Golden-Ray only (Jan-Aug 2019, n=8)",
    "Value": round(gr["brun_pre_goldenray_mean"], 1)}, {
    "Check": "Brunswick RoRo portcalls/mo, during Golden Ray salvage (Sep2019-Oct2021)",
    "Value": round(gr["brun_during_goldenray_mean"], 1)}, {
    "Check": "Brunswick RoRo portcalls/mo, full 2019-2021 baseline (as used in Table 4)",
    "Value": round(gr["brun_full_pre_mean"], 1)}, {
    "Check": "Brunswick RoRo portcalls/mo, post (Jul-Dec 2024)",
    "Value": round(gr["brun_post_mean"], 1)}, {
    "Check": "Surge % using full 2019-2021 baseline (as reported in Table 4)",
    "Value": f"{gr['surge_pct_full_pre']:+.1f}%"}, {
    "Check": "Surge % using pre-Golden-Ray-only baseline (sensitivity check)",
    "Value": f"{gr['surge_pct_pregr_only']:+.1f}%"},
]
save_table(pd.DataFrame(rows5b), "tableS4_golden_ray_sensitivity", "supplementary",
           "Brunswick baseline sensitivity to the Golden Ray shipwreck")

# TABLE 5 (main) — Welfare estimate
welfare = pd.read_csv(ROOT/"data/processed/welfare_estimate.csv")
save_table(welfare, "table5_welfare_estimate", "main",
           "Back-of-envelope welfare estimate")

# TABLE S5 (supplementary) — Event study coefficients
import pickle as _pkl
with open(ROOT/"data/processed/event_study.pkl","rb") as _f:
    _ev_pkl = _pkl.load(_f)
_ev = _ev_pkl["outcomes"]
_pretrend_tests = _ev_pkl["pretrend_tests"]
_SHOCK = pd.Timestamp("2024-03-01")
_rows7 = []
for _outcome, _data in _ev.items():
    for _k, _c in sorted(_data["coefs"].items()):
        _dt = _SHOCK + pd.DateOffset(months=_k)
        _rows7.append({
            "Outcome": _data["label"], "k": _k,
            "Calendar_month": str(_dt.date())[:7],
            "Estimate": round(_c["est"],5),
            "CI_lo": round(_c["lo"],5), "CI_hi": round(_c["hi"],5),
            "p_value": round(_c["pval"],4),
            "Sig": "***" if _c["pval"]<0.001 else "**" if _c["pval"]<0.01
                   else "*" if _c["pval"]<0.05 else "",
            "Phase": "Pre" if _k < -1 else "Reference" if _k==-1 else "Post",
        })
save_table(pd.DataFrame(_rows7), "tableS5_event_study_coefficients", "supplementary",
           "Full event-study coefficient table (all outcomes, k=-12..+9)")

# TABLE S6 (supplementary) — Pre-trend joint F-test (H0: flat pre-trend,
# k=-12..-2), one row per outcome. Reported for ALL outcomes, not just the
# ones that pass: this is the formal statistical test of the ITS/event-study
# identifying assumption, and it FAILS for the primary outcome (auto
# production) and for the placebo (petrochem PPI). That is a first-order
# limitation, not a cosmetic diagnostic -- see the manuscript's Limitations
# section.
_rows7b = []
for _outcome, _pt in _pretrend_tests.items():
    _rows7b.append({
        "Outcome": _pt["label"],
        "F_statistic": round(_pt["F"], 3),
        "p_value": round(_pt["p"], 4),
        "Flat_pretrend_at_10pct": _pt["flat"],
        "Interpretation": "Supports parallel trends" if _pt["flat"]
                           else "VIOLATES flat pre-trend -- interpret ITS estimates for this series with caution",
    })
save_table(pd.DataFrame(_rows7b), "tableS6_pretrend_ftest", "supplementary",
           "Pre-trend joint F-test, identifying-assumption check")

# TABLE 7 (main) — Carrying cost
# 11_carrying_cost.py computes this analysis; the publication-ready table is
# reconstructed here (from the same underlying results CSV) so all table
# numbering/output stays centralized in this one script.
_ccr = pd.read_csv(ROOT/"data/processed/carrying_cost_results.csv")
_ccr_d = dict(zip(_ccr["Parameter"], _ccr["Value"].astype(float)))
_sssum = dict(zip(ss_sum["Parameter"], ss_sum["Value"].astype(float)))
cc = pd.DataFrame([
    {"Metric": "Baltimore supply share (theta, central)", "Value": f"{_sssum['theta_central']:.3f}"},
    {"Metric": "Implied safety stock, SS (days)", "Value": f"{_sssum['ss_central_days']:.1f}  [{_sssum['ss_range_low_days']:.1f}, {_sssum['ss_range_high_days']:.1f}]"},
    {"Metric": "Full-fleet parts value ($B/day)", "Value": f"{_ccr_d['daily_parts_value_B_fullfleet']:.3f}"},
    {"Metric": "Buffer value ($B)", "Value": f"{_ccr_d['buffer_value_central_B']:.2f}  [{_ccr_d['buffer_value_lo_B']:.2f}, {_ccr_d['buffer_value_hi_B']:.2f}]"},
    {"Metric": "Carrying-cost rate assumed (%/yr)", "Value": f"{_ccr_d['carry_rate_central']*100:.1f}  [{_ccr_d['carry_rate_low']*100:.0f}, {_ccr_d['carry_rate_high']*100:.0f}]"},
    {"Metric": "Annual carrying cost of observed buffer ($B/yr)", "Value": f"{_ccr_d['annual_carry_cost_central_B']:.3f}  [{_ccr_d['annual_carry_cost_lo_B']:.3f}, {_ccr_d['annual_carry_cost_hi_B']:.3f}]"},
    {"Metric": "Incremental buffer to fully avert shortfall (days)", "Value": f"{_ccr_d['incremental_days_to_full_coverage']:.1f}"},
    {"Metric": "Incremental annual carrying cost ($B/yr)", "Value": f"{_ccr_d['incremental_annual_carry_cost_central_B']:.3f}  [{_ccr_d['incremental_annual_carry_cost_lo_B']:.3f}, {_ccr_d['incremental_annual_carry_cost_hi_B']:.3f}]"},
    {"Metric": "One-time value avoided by observed buffer, passenger-car ($B)", "Value": f"{_ccr_d['avoided_value_passenger_B']:.3f}"},
    {"Metric": "One-time value avoided by observed buffer, full-fleet est. ($B)", "Value": f"{_ccr_d['avoided_value_fullfleet_B']:.3f}"},
    {"Metric": "Remaining loss despite buffer, passenger-car ($B)", "Value": f"{_ccr_d['remaining_value_passenger_B']:.3f}"},
    {"Metric": "Break-even years (avoided loss / annual carrying cost), passenger-car scope", "Value": f"{_ccr_d['breakeven_years_observed_passenger']:.2f}"},
    {"Metric": "Break-even years (avoided loss / annual carrying cost), full-fleet scope", "Value": f"{_ccr_d['breakeven_years_observed_fullfleet']:.2f}"},
])
save_table(cc, "table7_carrying_cost", "main",
           "Inventory carrying cost of the implied JIT buffer")

n_main = len(list(TAB_MAIN.glob("*.csv")))
n_supp = len(list(TAB_SUPP.glob("*.csv")))
print(f"\nAll tables written: {n_main} main + {n_supp} supplementary "
      f"(paper_assets/tables/)")
