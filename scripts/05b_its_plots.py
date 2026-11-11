"""
Step 5b: ITS Visualisation
Figure 2 (main): Four-panel ITS — Panel A replaced with YoY comparison for
          auto production (ITS counterfactual for production is unreliable
          due to a steep, COVID-era pre-trend; see Table 7b).
Figure 3 (main): Coefficient forest plot.
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_PROC = ROOT / "data" / "processed"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import statsmodels.api as sm
import pickle, warnings
warnings.filterwarnings("ignore")

import _style
from _style import COLOR, INK_PRIMARY, INK_SECONDARY, INK_MUTED, save_figure, style_axes, event_lines
_style.apply_style()

master = pd.read_csv(str(DATA_PROC / "master.csv"),
                     parse_dates=["date"], index_col="date")
with open(str(DATA_PROC / "its_results.pkl"), "rb") as f:
    res = pickle.load(f)
store_a = res["model_a"]

MONTH_DUMMIES = [f"m{m:02d}" for m in range(2, 13)]

# ── Pre-trend joint F-test (flat pre-trend, H0: all pre-period event-time
# dummies = 0), computed self-contained here rather than imported from
# 08c_event_study.py's pickle: 08c runs at Step 8 in run_all.py, AFTER this
# script (Step 5b), so depending on its output would silently skip this
# caveat on a clean from-scratch `python run_all.py` run. This duplicates
# ~15 lines of 08c's event-time-dummy machinery, deliberately, to avoid that
# step-order hazard. Results match Table 7b exactly (both use the same
# design and HAC(4) covariance).
def _pretrend_flat_test(outcome):
    SHOCK = pd.Timestamp("2024-03-01")
    ek = ((master.index.year - SHOCK.year) * 12 + (master.index.month - SHOCK.month))
    df = master.copy()
    df["event_k"] = ek
    k_vals = [k for k in range(-12, 10) if k != -1]
    cols = ["t", "covid_shock"] + MONTH_DUMMIES
    Xb = sm.add_constant(df[cols].copy())
    for k in k_vals:
        Xb[f"ev_{k:+03d}"] = (df["event_k"] == k).astype(int)
    cn = Xb.columns.tolist()
    y = df[outcome].dropna()
    X = Xb.loc[y.index]
    if np.linalg.matrix_rank(X.values) < X.shape[1]:
        return None  # defensive: skip rather than trust a rank-deficient fit
    fit = sm.OLS(y.values, X.values).fit(
        cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": True})
    pre_cols = [f"ev_{k:+03d}" for k in range(-12, -1)]
    R = np.zeros((len(pre_cols), len(cn)))
    for i, c in enumerate(pre_cols):
        R[i, cn.index(c)] = 1
    ft = fit.f_test(R)
    return {"F": float(ft.fvalue), "p": float(ft.pvalue), "flat": float(ft.pvalue) > 0.10}

_pretrend_tests = {o: _pretrend_flat_test(o) for o in
                   ["ln_auto_prod", "ppi_mv_parts", "ln_hs8708_cif_B", "ppi_petrochem"]}

COLLAPSE = pd.Timestamp("2024-03-26")
FULL_OP  = pd.Timestamp("2024-06-10")

def build_X(df):
    cols = ["t","post","t_post","covid_shock","uaw_strike"] + MONTH_DUMMIES
    return sm.add_constant(df[cols].copy()), ["const"] + cols

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 2 (main): ITS observed vs counterfactual, 4 outcomes
# ═════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# ── PANEL A: YoY comparison for auto production (replaces ITS CF) ─────────
ax = axes.flat[0]
prod = master["auto_prod"]
months_2024 = master.loc["2024-01-01":"2024-12-01"]
months_2023 = master.loc["2023-01-01":"2023-12-01"]

ax.plot(months_2023.index + pd.DateOffset(years=1), months_2023["auto_prod"].values,
        color=INK_MUTED, lw=1.6, ls="--", zorder=2, label="2023 actual (lagged 1yr)")
ax.plot(months_2024.index, months_2024["auto_prod"].values,
        color=COLOR["auto_prod"], lw=2.0, zorder=3, label="2024 actual")
event_lines(ax, COLLAPSE, FULL_OP, label=False)

apr24 = prod.loc["2024-04-01"]; apr23 = prod.loc["2023-04-01"]
ax.annotate(f"Apr 2024: {apr24:.1f}k vs {apr23:.1f}k in 2023\n(−27.3% YoY)",
            xy=(pd.Timestamp("2024-04-01"), apr24),
            xytext=(pd.Timestamp("2024-05-20"), 108),
            fontsize=_style.FS_ANNOT, color=COLOR["shock"], fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color=COLOR["shock"], lw=1.1,
                             connectionstyle="arc3,rad=-0.15"))
ax.set_title("(A)  Auto production: 2024 vs 2023 (YoY)", fontsize=_style.FS_TITLE, pad=5)
ax.set_ylabel("Production (DAUPSA, k units/mo, SA)", fontsize=_style.FS_AXIS)
ax.set_xlim(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
ax.set_ylim(100, 160)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,3,5,7,9,11]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.legend(fontsize=_style.FS_LEGEND, loc="upper right")

# ── PANELS B, C, D: Standard ITS CFs (valid for these series) ─────────────
PANELS = [
    ("ppi_mv_parts",    "ppi_mv_parts",   "(B)  PPI: motor vehicle parts",       "Index (Dec 2003 = 100)"),
    ("ln_hs8708_cif_B", "hs8708",         "(C)  ln(HS 8708 CIF imports) [null]", "ln(USD billions)"),
    ("ppi_petrochem",   "ppi_petrochem",  "(D)  PPI: petrochemicals [placebo]",  "Index (Dec 2003 = 100)"),
]

for ax, (outcome, role, title, ylabel) in zip(axes.flat[1:], PANELS):
    fit, cn, Xdf = store_a[outcome]
    params = fit.params
    Xdf_full, cn_full = build_X(master)
    fitted = Xdf_full.values @ params
    Xcf = Xdf_full.copy(); Xcf["post"] = 0; Xcf["t_post"] = 0
    cf = Xcf.values @ params

    y_obs = master[outcome]
    ax.scatter(master.index, y_obs, color=INK_MUTED, s=12, alpha=0.55, zorder=3, label="Observed")
    ax.plot(master.index, fitted, color=COLOR[role], lw=1.8, zorder=4, label="ITS fitted")
    post_mask = master["post"].values == 1
    ax.plot(master.index[post_mask], cf[post_mask], color=COLOR["shock"],
            lw=1.6, ls="--", zorder=4, label="Counterfactual")
    ax.fill_between(master.index[post_mask], fitted[post_mask], cf[post_mask],
                    alpha=0.14, color=COLOR["shock"], zorder=2)
    event_lines(ax, COLLAPSE, FULL_OP, label=False)
    ax.set_title(title, fontsize=_style.FS_TITLE, pad=5)
    ax.set_ylabel(ylabel, fontsize=_style.FS_AXIS)
    ax.set_xlim(pd.Timestamp("2019-01-01"), pd.Timestamp("2024-12-31"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=_style.FS_FOOTNOTE, loc="upper left", ncol=2)

patches = [
    mpatches.Patch(color=COLOR["shock"], alpha=0.06, label="Disruption window"),
    plt.Line2D([0], [0], color=COLOR["shock"], lw=1.6, ls="--", label="FSK collapse (Mar 26)"),
    plt.Line2D([0], [0], color=COLOR["recovery"], lw=1.6, ls="--", label="Full reopening (Jun 10)"),
]
fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=_style.FS_LEGEND,
           frameon=False, bbox_to_anchor=(0.5, -0.03))

for ax in axes.flat:
    style_axes(ax)

plt.tight_layout(rect=[0, 0.03, 1, 1])
save_figure(fig, "figure2_its_results", category="main")

# ═════════════════════════════════════════════════════════════════════════
# FIGURE 3 (main): Coefficient forest plot
# ═════════════════════════════════════════════════════════════════════════
OUTCOMES = {
    "ln_auto_prod":     ("ln(Auto production) [primary]", "auto_prod"),
    "ppi_mv_parts":      ("PPI, MV parts [price]",          "ppi_mv_parts"),
    "ln_hs8708_cif_B":   ("ln(HS 8708) [trade / null]",      "hs8708"),
    "ppi_petrochem":      ("PPI, petrochem [placebo]",        "ppi_petrochem"),
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

# Outcomes whose flat-pre-trend identifying assumption is violated (Table 7b)
# get a dagger marker on their y-axis label rather than being shown as if
# every outcome's ITS estimate carries equal weight.
_ytick_labels = []
for outcome, (label, role) in OUTCOMES.items():
    pt = _pretrend_tests.get(outcome)
    _ytick_labels.append(label + "  †" if (pt and not pt["flat"]) else label)

for ax_i, (coef_key, coef_label) in enumerate([
    ("post", "β2 — immediate level shift"),
    ("t_post", "β3 — slope change post-shock"),
]):
    ax = axes[ax_i]
    ys = list(range(len(OUTCOMES)))[::-1]
    for yi, (outcome, (label, role)) in zip(ys, OUTCOMES.items()):
        fit, cn, _ = store_a[outcome]
        idx = cn.index(coef_key); v = fit.params[idx]
        ci = fit.conf_int(); lo, hi = ci[idx, 0], ci[idx, 1]
        p = fit.pvalues[idx]; col = COLOR[role]; sig = p < 0.05
        ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="o", color=col,
                    markersize=8 if sig else 6, capsize=4.5, capthick=1.4, lw=1.4,
                    markerfacecolor=col if sig else "white", markeredgewidth=1.4, zorder=3)
    ax.axvline(0, color=INK_SECONDARY, lw=1.1, ls="--", alpha=0.6, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(_ytick_labels[::-1], fontsize=_style.FS_AXIS)
    ax.set_xlabel(coef_label, fontsize=_style.FS_AXIS)
    ax.set_title(f"({chr(65 + ax_i)})", fontsize=_style.FS_TITLE, pad=6, loc="left")
    style_axes(ax, grid_axis="x")
    ax.scatter([], [], color=INK_SECONDARY, s=60, label="p < 0.05")
    ax.scatter([], [], color=INK_SECONDARY, s=36, facecolors="white",
               edgecolors=INK_SECONDARY, lw=1.4, label="p ≥ 0.05")
    ax.legend(fontsize=_style.FS_FOOTNOTE, loc="best")

fig.text(0.5, -0.06,
          "† Pre-trend joint F-test (Table 7b) rejects flatness at the 10% level for this outcome — "
          "its ITS point estimate rests on a violated identifying assumption and should be read with that caveat.",
          ha="center", fontsize=_style.FS_FOOTNOTE, color=INK_SECONDARY, style="italic", wrap=True)

plt.tight_layout()
save_figure(fig, "figure3_coefficient_forest", category="main")
print("Step 5b complete")
