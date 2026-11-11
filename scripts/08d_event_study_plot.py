"""
Step 8d: Event-Study Plot — Figure 6 (main)
Two-panel:
  (A) Auto production + HS8708 imports (primary outcomes)
  (B) PPI MV parts + Petrochem placebo (price + falsification)
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_PROC = ROOT / "data" / "processed"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pickle, warnings
warnings.filterwarnings("ignore")

import _style
from _style import COLOR, INK_PRIMARY, INK_SECONDARY, INK_MUTED, save_figure, style_axes
_style.apply_style()

with open(str(DATA_PROC / "event_study.pkl"), "rb") as f:
    _ev_pkl = pickle.load(f)
ev = _ev_pkl["outcomes"]
pretrend_tests = _ev_pkl["pretrend_tests"]

K_MIN, K_MAX, K_REF = -12, 9, -1

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

PANELS = [
    [("ln_auto_prod", "ln(Auto production)", COLOR["auto_prod"], "o"),
     ("ln_hs8708_cif_B", "ln(HS 8708 imports)", COLOR["hs8708"], "s")],
    [("ppi_mv_parts", "PPI: MV parts", COLOR["ppi_mv_parts"], "o"),
     ("ppi_petrochem", "PPI: petrochem [placebo]", COLOR["ppi_petrochem"], "s")],
]
TITLES = [
    "(A)  Production & trade outcomes",
    "(B)  Price outcomes & placebo",
]
# Pre-trend joint F-test results as a compact caption line under each title,
# not as floating arrow-annotations -- stable regardless of figure size, and
# states the identifying-assumption test result plainly rather than pointing
# at an approximate spot on a noisy pre-period series. Built from
# pretrend_tests (loaded from event_study.pkl above) rather than hardcoded,
# so this stays correct if the underlying data ever changes -- same
# stale-hardcoded-number bug class fixed elsewhere in this pipeline
# (11_carrying_cost.py, 08b/06_safety_stock.py; see project history).
def _pt_str(outcome_key, short_label):
    d = pretrend_tests[outcome_key]
    verdict = "flat" if d["flat"] else "NOT flat"
    p_str = "p<0.001" if d["p"] < 0.001 else f"p={d['p']:.3f}"
    dagger = ", †" if not d["flat"] else ""
    return f"{short_label} pre-trend {verdict} (F={d['F']:.2f}, {p_str}{dagger})"

SUBTITLES = [
    f"{_pt_str('ln_auto_prod', 'Auto Prod')} · {_pt_str('ln_hs8708_cif_B', 'HS 8708')}",
    f"{_pt_str('ppi_mv_parts', 'PPI parts')} · {_pt_str('ppi_petrochem', 'Placebo')}",
]

for ax, panel_series, title, subtitle in zip(axes, PANELS, TITLES, SUBTITLES):
    for outcome, label, colour, marker in panel_series:
        coefs = ev[outcome]["coefs"]
        k_vals = sorted(coefs.keys())
        ests   = [coefs[k]["est"] for k in k_vals]
        lo_err = [coefs[k]["est"] - coefs[k]["lo"] for k in k_vals]
        hi_err = [coefs[k]["hi"] - coefs[k]["est"] for k in k_vals]
        pvals  = [coefs[k]["pval"] for k in k_vals]

        marker_cols = [colour if k >= 0 else
                       INK_MUTED if pvals[i] >= 0.05 else colour
                       for i, k in enumerate(k_vals)]

        for i, (k, est, lo, hi, p, mc) in enumerate(
                zip(k_vals, ests, lo_err, hi_err, pvals, marker_cols)):
            alpha = 1.0 if k >= 0 else 0.65
            ax.errorbar(k, est, yerr=[[lo], [hi]], fmt=marker, color=mc,
                        markersize=6 if k >= 0 else 4.5,
                        capsize=3.5, capthick=1.1, lw=1.2,
                        alpha=alpha, zorder=3, label=label if i == 0 else "")
        ax.plot(k_vals, ests, color=colour, lw=1.1, alpha=0.35, zorder=2)

    ax.axvline(-0.5, color=COLOR["shock"], lw=1.8, ls="--", alpha=0.85, zorder=4,
               label="Shock (k=0 = Mar 2024)")
    ax.axhline(0, color=INK_SECONDARY, lw=0.9, alpha=0.5, zorder=1)
    ax.axvspan(K_MIN - 0.5, -1.5, color=COLOR["baltimore"], alpha=0.05, zorder=0,
               label="Pre-shock (k < −1)")
    ax.axvspan(-0.5, K_MAX + 0.5, color=COLOR["shock"], alpha=0.05, zorder=0,
               label="Post-shock (k ≥ 0)")
    ax.scatter([-1], [0], color=INK_PRIMARY, s=55, zorder=5, marker="D",
               label="Reference (k=−1, Feb 2024)")

    month_at = {-12:"Mar'23", -10:"May'23", -8:"Jul'23", -6:"Sep'23", -4:"Nov'23",
                -2:"Jan'24", 0:"Mar'24", 2:"May'24", 4:"Jul'24", 6:"Sep'24", 8:"Nov'24"}
    ax.set_xticks(sorted(month_at.keys()))
    ax.set_xticklabels([month_at[k] for k in sorted(month_at.keys())],
                       fontsize=_style.FS_FOOTNOTE, rotation=35, ha="right")

    ax.set_xlim(K_MIN - 0.8, K_MAX + 0.5)
    ax.set_xlabel("Months relative to FSK collapse", fontsize=_style.FS_AXIS)
    ax.set_ylabel("Coefficient (relative to Feb 2024)", fontsize=_style.FS_AXIS)
    ax.set_title(title, fontsize=_style.FS_TITLE, pad=22)
    ax.text(0.5, 1.11, subtitle, transform=ax.transAxes, ha="center",
            fontsize=_style.FS_FOOTNOTE, color=INK_SECONDARY)
    ax.legend(fontsize=_style.FS_FOOTNOTE, loc="upper left", ncol=1)
    style_axes(ax)

# One pointing annotation per panel, each calling out a specific estimate --
# the pre-trend flat/not-flat verdicts themselves are stated plainly in the
# subtitle above (Table 7b has the full numbers), not chased around the plot
# with arrows that collide with the legend at different figure sizes.
ax_a = axes[0]
ax_a.set_ylim(-1.5, 2.6)
ax_a.annotate("HS 8708 Dec 2024: −0.18**\n(p=0.003) → mild post-shock\nimport deceleration",
              xy=(9, ev["ln_hs8708_cif_B"]["coefs"][9]["est"]),
              xytext=(5.8, -1.35),
              fontsize=_style.FS_FOOTNOTE, color=COLOR["hs8708"], ha="left",
              arrowprops=dict(arrowstyle="->", color=COLOR["hs8708"], lw=1))

ax_b = axes[1]
ax_b.set_ylim(-145, 35)
ax_b.annotate("PPI MV parts: flat throughout →\nNO price effect (p > 0.40 all months)",
              xy=(3, ev["ppi_mv_parts"]["coefs"][3]["est"]),
              xytext=(-1, 26),
              fontsize=_style.FS_FOOTNOTE, color=COLOR["ppi_mv_parts"], ha="left",
              arrowprops=dict(arrowstyle="->", color=COLOR["ppi_mv_parts"], lw=1))

fig.text(0.5, -0.02,
          "† Pre-trend joint F-test (Table 7b) rejects flatness at the 10% level — see Figs. 2–3 for how this caveat propagates.",
          ha="center", fontsize=_style.FS_FOOTNOTE, color=INK_SECONDARY, style="italic")

plt.tight_layout()
save_figure(fig, "figure6_event_study", category="main")
