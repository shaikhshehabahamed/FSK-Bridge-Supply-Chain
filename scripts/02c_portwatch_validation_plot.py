"""
Step 2c: PortWatch Validation Plot
Shows:
  (Left)  Baltimore import volume — COMPLETE 2019-2024+ series (shock clearly visible)
  (Right) RoRo portcalls: Baltimore vs Brunswick 2019-2021 baseline
Figure S5 (supplementary).
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_RAW  = ROOT / "data" / "raw"

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np

import _style
from _style import COLOR, INK_SECONDARY, GRIDLINE, save_figure, style_axes, event_lines
_style.apply_style()

balt_monthly = pd.read_csv(str(DATA_RAW / "portwatch_baltimore_monthly.csv"),
                           parse_dates=["date"])
portwatch_m = pd.read_csv(str(DATA_RAW / "portwatch_monthly.csv"),
                          parse_dates=["date"])

COLLAPSE = pd.Timestamp("2024-03-26")
FULL_OP  = pd.Timestamp("2024-06-10")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

# ── Panel A: Baltimore import volume 2019-2024 ──────────────────────────────
ax = axes[0]
series = balt_monthly[balt_monthly["date"] <= "2024-12-31"].copy()
series["balt_M"] = series["baltimore_import_vol"] / 1e6
pre_mean = series[(series.date.dt.year >= 2019) & (series.date.dt.year <= 2023) &
                  ~((series.date.dt.year == 2020) & series.date.dt.month.isin([4, 5, 6]))]["balt_M"].mean()

ax.plot(series["date"], series["balt_M"], color=COLOR["baltimore"], lw=1.6, zorder=2)
event_lines(ax, COLLAPSE, FULL_OP, label=False)
ax.axhline(pre_mean, color=INK_SECONDARY, lw=1, ls=":", alpha=0.7, zorder=2)

apr_val = series[series.date == "2024-04-01"]["balt_M"].values[0]
ax.scatter(pd.Timestamp("2024-04-01"), apr_val, color=COLOR["shock"], s=40, zorder=5,
           edgecolor="white", linewidth=1.0)
ax.annotate("Apr 2024\n−99.5%", xy=(pd.Timestamp("2024-04-01"), apr_val),
            xytext=(pd.Timestamp("2022-06-01"), 1.1),
            fontsize=_style.FS_ANNOT, color=COLOR["shock"],
            arrowprops=dict(arrowstyle="->", color=COLOR["shock"], lw=1))

ax.set_title("(A)  Baltimore import volume (est., DWT-equiv.)\nIMF PortWatch Container_Metrics",
             fontsize=_style.FS_TITLE, pad=6)
ax.set_ylabel("Monthly import volume (millions)", fontsize=_style.FS_AXIS)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlim(pd.Timestamp("2019-01-01"), pd.Timestamp("2024-12-31"))
ax.text(0.02, 0.05, f"Pre-period mean (2019–2023): {pre_mean:.2f}M",
        transform=ax.transAxes, fontsize=_style.FS_FOOTNOTE, color=INK_SECONDARY)
style_axes(ax)

# ── Panel B: RoRo portcalls Baltimore vs Brunswick ──────────────────────────
ax2 = axes[1]
_full_months = pd.date_range("2019-01-01", "2024-12-01", freq="MS")
for port, colour, label in [("baltimore", COLOR["baltimore"], "Baltimore"),
                             ("brunswick", COLOR["brunswick"], "Brunswick")]:
    s = portwatch_m[portwatch_m["port_label"] == port].set_index("date")[["portcalls_roro"]]
    s = s.reindex(_full_months)  # genuine break across the coverage gap, not a straight connector
    ax2.plot(s.index, s["portcalls_roro"], color=colour, lw=1.6,
             marker="o", markersize=3, label=label, zorder=2)

ax2.axvspan(pd.Timestamp("2021-09-01"), pd.Timestamp("2024-06-01"),
            facecolor=GRIDLINE, alpha=0.6, zorder=1, hatch="//", edgecolor="white")
ax2.text(pd.Timestamp("2022-09-01"), 10, "PortWatch\ndata gap\n(2021–2024)",
         fontsize=_style.FS_FOOTNOTE, color=INK_SECONDARY, ha="center")
event_lines(ax2, COLLAPSE, FULL_OP, label=True)

ax2.set_title("(B)  RoRo vessel port calls: Baltimore vs. Brunswick\nIMF PortWatch Daily_Ports_Data",
              fontsize=_style.FS_TITLE, pad=6)
ax2.set_ylabel("Monthly RoRo port calls", fontsize=_style.FS_AXIS)
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.legend(fontsize=_style.FS_FOOTNOTE, loc="upper left")
style_axes(ax2)

plt.tight_layout(rect=[0, 0, 1, 1])
save_figure(fig, "figureS5_portwatch_validation", category="supplementary")
