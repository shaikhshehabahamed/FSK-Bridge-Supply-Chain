"""
Step 7b: Robustness visualisation
Figure S2 (supplementary): Three-panel robustness figure
  (A) False-break test — F-stat distribution vs true shock
  (B) Robustness coefficient chart (beta2 across specs)
  (C) Placebo outcome comparison (auto_prod vs petrochemical PPI)
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_PROC = ROOT / "data" / "processed"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import statsmodels.api as sm
import pickle, warnings
warnings.filterwarnings("ignore")

import _style
from _style import COLOR, CRITICAL, WARNING, GOOD, INK_PRIMARY, INK_SECONDARY, INK_MUTED, save_figure, style_axes, event_lines
_style.apply_style()

master = pd.read_csv(str(DATA_PROC / "master.csv"),
                     parse_dates=["date"], index_col="date")

with open(str(DATA_PROC / "robustness.pkl"), "rb") as f:
    rob = pickle.load(f)
with open(str(DATA_PROC / "its_results.pkl"), "rb") as f:
    its = pickle.load(f)

df_plac = rob["df_plac"]
true_F  = rob["true_F"]
true_b2 = rob["true_b2"]
lag_res = rob["lag_results"]
rows    = rob["rows"]

MONTH_DUMMIES = [f"m{m:02d}" for m in range(2, 13)]
TRUE_SHOCK    = pd.Timestamp("2024-03-01")
COLLAPSE      = pd.Timestamp("2024-03-26")
FULL_OP       = pd.Timestamp("2024-06-10")

fig = plt.figure(figsize=(12.5, 4.4))
gs  = gridspec.GridSpec(1, 3, wspace=0.42)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

# ── Panel A: False-break test (Baltimore port volume) ───────────────────────
ax = axes[0]
ax.hist(df_plac["F"], bins=18, color=INK_MUTED, alpha=0.65, edgecolor="white",
        linewidth=0.8, label="Placebo date F-stats\n(2020-01 – 2023-12)", zorder=2)
ax.axvline(true_F, color=CRITICAL, lw=2.2, ls="--", zorder=4,
           label=f"True shock (Mar 2024)\nF = {true_F:.1f}")
pct95 = df_plac["F"].quantile(0.95)
ax.axvline(pct95, color=WARNING, lw=1.4, ls=":", zorder=3, label=f"95th pctile = {pct95:.1f}")

pct_rank = rob.get("pct_rank", (df_plac["F"] < true_F).mean() * 100)
ax.set_xlabel("Joint F-statistic (β2 = β3 = 0)", fontsize=_style.FS_AXIS)
ax.set_ylabel("Frequency", fontsize=_style.FS_AXIS)
ax.set_title(f"(A)  False-break test: Baltimore import volume\nTrue shock F at {pct_rank:.0f}th pctile of {len(df_plac)} placebo dates",
             fontsize=_style.FS_TITLE, pad=6)
ax.legend(fontsize=_style.FS_FOOTNOTE, loc="center right")
ax.text(0.05, 0.90, f"All {len(df_plac)} placebo\ndates: F < {df_plac['F'].max():.1f}",
        transform=ax.transAxes, fontsize=_style.FS_FOOTNOTE, color=INK_SECONDARY,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#e1e0d9"))
style_axes(ax)

# ── Panel B: Robustness coefficient chart ────────────────────────────────────
ax = axes[1]
spec_labels = ["Main\n(HAC-4)", "HAC-2", "HAC-6", "HAC-8", "COVID\nexcluded",
               "Pre: 2021+\n(restricted)", "Levels\n(÷100)", "March 2024\nexcluded"]
b2_vals  = [r[1] for r in rows]
se2_vals = [r[2] for r in rows]
p2_vals  = [r[3] for r in rows]
ci_width = [1.96 * se for se in se2_vals]

y = np.arange(len(spec_labels))[::-1]
for yi, b2, p2, cw in zip(y, b2_vals, p2_vals, ci_width):
    col  = CRITICAL if p2 < 0.05 else WARNING if p2 < 0.10 else INK_MUTED
    fill = col if p2 < 0.10 else "white"
    ax.errorbar(b2, yi, xerr=cw, fmt="o", color=col, markersize=8, capsize=4,
                capthick=1.3, lw=1.3, markerfacecolor=fill, markeredgewidth=1.5, zorder=3)

ax.axvline(0, color=INK_SECONDARY, lw=1.1, ls="--", alpha=0.6, zorder=2)
ax.set_yticks(y)
ax.set_yticklabels(spec_labels, fontsize=_style.FS_FOOTNOTE)
ax.set_xlabel("β2 — immediate level shift", fontsize=_style.FS_AXIS)
ax.set_title("(B)  Robustness: β2 across specifications\nln(Auto production) · HAC SEs",
             fontsize=_style.FS_TITLE, pad=6)
ax.legend(handles=[
    mpatches.Patch(color=CRITICAL, label="p < 0.05"),
    mpatches.Patch(color=WARNING, label="p < 0.10"),
    mpatches.Patch(facecolor="white", edgecolor=INK_MUTED, linewidth=1.3, label="p ≥ 0.10"),
], fontsize=_style.FS_FOOTNOTE, loc="lower right")
style_axes(ax, grid_axis="y")

# ── Panel C: Placebo outcome side-by-side (single shared axis — both series
# are already indexed to Jan 2019 = 100, so a second y-scale would be
# decorative, not informative; a dual-axis chart is avoided) ────────────────
ax = axes[2]

def run_simple_its(outcome_series, master_df):
    df = master_df.copy()
    cols = ["t", "post", "t_post", "covid_shock", "uaw_strike"] + MONTH_DUMMIES
    X = sm.add_constant(df[cols])
    y = outcome_series.loc[df.index].dropna()
    X = X.loc[y.index]
    res = sm.OLS(y.values, X.values).fit(cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": True})
    fitted = X.values @ res.params
    Xcf = X.copy(); Xcf["post"] = 0; Xcf["t_post"] = 0
    cf = Xcf.values @ res.params
    return fitted, cf, y.index, res

fit_auto, cf_auto, idx_auto, res_auto = run_simple_its(np.log(master["auto_prod"]), master)
fit_plac, cf_plac, idx_plac, res_plac = run_simple_its(master["ppi_petrochem"], master)

ref_auto = np.exp(fit_auto[0])
ref_plac = fit_plac[0]

ax.scatter(idx_auto, np.exp(master.loc[idx_auto, "ln_auto_prod"]) / ref_auto * 100,
           color=COLOR["auto_prod"], s=10, alpha=0.55, zorder=3)
ax.plot(idx_auto, np.exp(fit_auto) / ref_auto * 100, color=COLOR["auto_prod"], lw=1.6,
        zorder=4, label="Auto production (ITS fit)")

ax.scatter(idx_plac, master.loc[idx_plac, "ppi_petrochem"] / ref_plac * 100,
           color=INK_MUTED, s=10, alpha=0.45, zorder=2)
ax.plot(idx_plac, fit_plac / ref_plac * 100, color=INK_MUTED, lw=1.4,
        zorder=3, label="Petrochem PPI (placebo)")

event_lines(ax, COLLAPSE, FULL_OP, label=False)

cn_ref = list(sm.add_constant(master[["t","post","t_post","covid_shock","uaw_strike"] + MONTH_DUMMIES]).columns)
b2_auto = res_auto.params[cn_ref.index("post")]; p2_auto = res_auto.pvalues[cn_ref.index("post")]
b2_plac = res_plac.params[cn_ref.index("post")]; p2_plac = res_plac.pvalues[cn_ref.index("post")]

ax.set_ylabel("Index (Jan 2019 = 100)", fontsize=_style.FS_AXIS)
ax.set_title("(C)  Outcome vs. placebo comparison\nAuto production vs. petrochem PPI", fontsize=_style.FS_TITLE, pad=6)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

textstr = f"Auto β2={b2_auto:+.3f} (p={p2_auto:.2f})\nPlacebo β2={b2_plac:+.1f} (p={p2_plac:.2f})"
ax.text(0.03, 0.06, textstr, transform=ax.transAxes, fontsize=_style.FS_FOOTNOTE,
        va="bottom", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#e1e0d9"))
ax.legend(fontsize=_style.FS_FOOTNOTE, loc="upper right")
style_axes(ax)

plt.tight_layout(rect=[0, 0, 1, 1])
save_figure(fig, "figureS2_robustness", category="supplementary")
print("Step 7b complete")
