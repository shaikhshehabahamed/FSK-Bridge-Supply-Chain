"""
Step 11b: Inventory Carrying Cost Visualisation
Figure 7 (main): Carrying cost of the implied JIT buffer vs. the one-time
production loss it avoided during the FSK closure.
  (A) Annual carrying cost of the observed buffer vs. the one-time
      avoided loss from the April 2024 closure (passenger-car and
      full-fleet scope)
  (B) Break-even sensitivity: how many years of carrying cost the
      avoided loss offsets, across the 15-30% carrying-cost-rate range
"""
from pathlib import Path as _Path
ROOT      = _Path(__file__).parent.parent
DATA_PROC = ROOT / "data" / "processed"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

import _style
from _style import CRITICAL, GOOD, INK_PRIMARY, INK_SECONDARY, save_figure, style_axes
_style.apply_style()

cc = pd.read_csv(str(DATA_PROC / "carrying_cost_results.csv"))
cc = cc.set_index("Parameter")["Value"].astype(float)
ss_central_days = pd.read_csv(str(DATA_PROC / "ss_summary.csv"),
                               index_col=0, header=None).squeeze()["ss_central_days"]
ss_central_days = float(ss_central_days)

buffer_value_central = cc["buffer_value_central_B"]
carry_central   = cc["annual_carry_cost_central_B"]
carry_lo        = cc["annual_carry_cost_lo_B"]
carry_hi        = cc["annual_carry_cost_hi_B"]
avoided_pass    = cc["avoided_value_passenger_B"]
avoided_full    = cc["avoided_value_fullfleet_B"]
breakeven_pass  = cc["breakeven_years_observed_passenger"]
breakeven_full  = cc["breakeven_years_observed_fullfleet"]
rate_lo, rate_c, rate_hi = cc["carry_rate_low"], cc["carry_rate_central"], cc["carry_rate_high"]

C_COST  = CRITICAL
C_BEN_P = GOOD
C_BEN_F = "#8fd18f"   # lighter step of GOOD -- same hue family, upper-bound scope
C_SHADE = "#eaf1fb"

fig = plt.figure(figsize=(10.5, 4.6))
gs = gridspec.GridSpec(1, 2, wspace=0.36)
axes = [fig.add_subplot(gs[0, i]) for i in range(2)]


# ── Panel A: cost vs. one-time avoided loss ─────────────────────────────────
ax = axes[0]
labels = ["Annual carrying\ncost (buffer)", "Avoided loss\n(passenger-car)", "Avoided loss\n(full-fleet est.)"]
values = [carry_central, avoided_pass, avoided_full]
colors = [C_COST, C_BEN_P, C_BEN_F]

ax.bar(labels, values, color=colors, alpha=0.92, edgecolor="white", width=0.58, zorder=3)
ax.errorbar([0], [carry_central], yerr=[[carry_central - carry_lo], [carry_hi - carry_central]],
            fmt="none", ecolor=INK_PRIMARY, elinewidth=1.4, capsize=4.5, zorder=4)

label_y_offset = [carry_hi - carry_central + 0.2, 0.15, 0.15]
for i, (v, dy) in enumerate(zip(values, label_y_offset)):
    ax.text(i, v + dy, f"${v:.2f}B", ha="center", va="bottom",
            fontsize=_style.FS_AXIS, fontweight="bold", color=INK_PRIMARY)

ax.set_ylabel("USD, billions", fontsize=_style.FS_AXIS)
ax.set_title(f"(A)  Annual cost vs. one-time avoided loss\nApril 2024; carrying-cost rate {rate_lo*100:.0f}–{rate_hi*100:.0f}%, central {rate_c*100:.1f}%",
             fontsize=_style.FS_TITLE, pad=6)
ax.set_ylim(0, max(values) * 1.35)
ax.text(0.02, 0.93,
        f"Avoided loss = {breakeven_pass:.1f}–{breakeven_full:.1f} yrs of carrying cost\n(passenger-car – full-fleet scope)",
        transform=ax.transAxes, ha="left", va="top", fontsize=_style.FS_FOOTNOTE,
        style="italic", color=INK_SECONDARY)
style_axes(ax)

# ── Panel B: break-even sensitivity across the carrying-cost-rate range ─────
ax = axes[1]
rate_range = np.linspace(0.10, 0.35, 100)
breakeven_pass_curve = avoided_pass / (buffer_value_central * rate_range)
breakeven_full_curve = avoided_full / (buffer_value_central * rate_range)

ax.plot(rate_range * 100, breakeven_pass_curve, color=C_BEN_P, lw=2.1, label="Passenger-car scope")
ax.plot(rate_range * 100, breakeven_full_curve, color=C_BEN_F, lw=2.1, label="Full-fleet scope (upper bound)")

ax.axvspan(rate_lo * 100, rate_hi * 100, color=C_SHADE, alpha=0.9, zorder=1,
           label=f"Assumed rate range [{rate_lo*100:.0f}–{rate_hi*100:.0f}%]")
ax.axvline(rate_c * 100, color=INK_SECONDARY, lw=1.1, ls="--", alpha=0.7)
ax.scatter([rate_c * 100], [breakeven_pass], color=C_BEN_P, s=75, zorder=5,
           edgecolor="white", linewidth=1.1, label=f"Central: {breakeven_pass:.1f} yrs (pax-car)")
ax.scatter([rate_c * 100], [breakeven_full], color=C_BEN_F, s=75, zorder=5,
           edgecolor="white", linewidth=1.1)

ax.set_xlabel("Assumed annual carrying-cost rate (%)", fontsize=_style.FS_AXIS)
ax.set_ylabel("Break-even years\n(avoided loss / annual cost)", fontsize=_style.FS_AXIS)
ax.set_title("(B)  Break-even sensitivity to carrying-cost rate\nRequired recurrence interval for the buffer to pay for itself",
             fontsize=_style.FS_TITLE, pad=6)
ax.legend(fontsize=_style.FS_FOOTNOTE, loc="upper right")
ax.set_xlim(10, 35)
style_axes(ax)

plt.tight_layout(rect=[0, 0, 1, 1])
save_figure(fig, "figure7_carrying_cost", category="main")
print("Step 11b complete")
