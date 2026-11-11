"""
Step 1b: FRED Data Validation Plot
Confirms data quality and documents pre-shock baselines visually.
Figure S4 (supplementary).
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_RAW  = ROOT / "data" / "raw"

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import _style
from _style import COLOR, AQUA, MAGENTA, INK_SECONDARY, save_figure, style_axes, event_lines
_style.apply_style()

df = pd.read_csv(str(DATA_RAW / "fred_raw.csv"),
                 parse_dates=["date"], index_col="date")

COLLAPSE = pd.Timestamp("2024-03-26")
FULL_OP  = pd.Timestamp("2024-06-10")

fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.5))

panels = [
    ("auto_production",  "Domestic auto production\n(DAUPSA, k units/mo, SA)", "Production (k units/mo)", True, COLOR["auto_prod"]),
    ("ppi_mv_parts_mfg",  "PPI: motor vehicle parts mfg\n(PCU33633363, Dec 2003=100)", "Index", False, COLOR["ppi_mv_parts"]),
    ("ppi_mv_parts_comm", "PPI commodity: motor vehicle parts\n(WPU1412, 1982=100)", "Index", False, COLOR["ppi_mv_parts"]),
    ("new_orders_mv",     "Mfrs new orders: motor vehicles & parts\n(AMTMNO, $M)", "$M", False, AQUA),
    ("cpi_mv_parts",      "CPI: motor vehicle parts & equipment\n(CUUR0000SETC)", "Index", False, MAGENTA),
    ("ppi_petrochem",     "PPI: petrochemical mfg [placebo]\n(PCU325110325110, Dec 2003=100)", "Index", False, COLOR["ppi_petrochem"]),
]

for ax, (col, title, ylabel, highlight, colour) in zip(axes.flat, panels):
    s = df[col].dropna()
    ax.plot(s.index, s.values, color=colour, lw=1.5, zorder=2)
    event_lines(ax, COLLAPSE, FULL_OP, label=False)

    if highlight:
        for dt in ["2024-04-01", "2024-05-01"]:
            val = df.loc[dt, col]
            ax.scatter(pd.Timestamp(dt), val, color=COLOR["shock"], s=35, zorder=5,
                       edgecolor="white", linewidth=1.0)

    ax.set_title(title, fontsize=_style.FS_FOOTNOTE, pad=5)
    ax.set_ylabel(ylabel, fontsize=_style.FS_FOOTNOTE)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(s.index.min(), s.index.max())
    style_axes(ax)
    ax.tick_params(labelsize=_style.FS_FOOTNOTE - 0.5)

patches = [
    mpatches.Patch(color=COLOR["shock"], alpha=0.06, label="Disruption window (Mar 26 – Jun 10, 2024)"),
    plt.Line2D([0], [0], color=COLOR["shock"], lw=1.4, ls="--", label="Collapse date (Mar 26)"),
    plt.Line2D([0], [0], color=COLOR["recovery"], lw=1.4, ls="--", label="Full reopening (Jun 10)"),
]
fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=_style.FS_LEGEND,
           frameon=False, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.03, 1, 1])
save_figure(fig, "figureS4_fred_validation", category="supplementary")
