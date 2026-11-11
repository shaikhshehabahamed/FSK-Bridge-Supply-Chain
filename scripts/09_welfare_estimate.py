"""
Step 9: Back-of-Envelope Welfare Estimate
==========================================
Converts the observed production shortfall (vehicles not built)
into dollar terms using average transaction prices and industry
employment data.

Two channels (a third, a rerouting logistics-cost premium, was dropped
2026-07-19 -- see the note above Channel 2's section below):
  (1) Lost vehicle output ($B)   — YoY units deficit x avg transaction price
  (2) Employment-hours lost ($B) — auto sector employment x output elasticity,
                                    reported as a short-run/long-run range
"""
from pathlib import Path as _Path
ROOT    = _Path(__file__).parent.parent
DATA_PROC = ROOT / "data" / "processed"



import pandas as pd
import numpy as np

master = pd.read_csv(str(DATA_PROC / "master.csv"),
                     parse_dates=["date"], index_col="date")


print("=" * 70)
print("WELFARE ESTIMATE — FSK Bridge Collapse Production Shortfall")
print("=" * 70)

# ── Parameters ────────────────────────────────────────────────────────────────
ATP           = 48_000       # average transaction price, new vehicle, 2024 (USD)
                              # source: Cox Automotive / Kelley Blue Book Mar 2024
PARTS_PER_VEH = 12_000       # approximate parts content per vehicle (industry estimate)
AUTO_EMPLOY   = 1_000_000    # direct auto mfg employment (BLS, Q1 2024)
HOURS_PER_MO  = 173          # avg monthly work hours per employee
OUTPUT_ELAST_LONGRUN  = 0.80  # output-employment elasticity for mfg (BLS historical,
                               # estimated over PERMANENT output changes)
OUTPUT_ELAST_SHORTRUN = 0.25  # short-run "labor hoarding" elasticity: this shock was
                               # known ex ante to be temporary (a channel reopening on
                               # a published timeline), and the industry was mid-recovery
                               # from the Oct-Nov 2023 UAW strike, both of which push
                               # firms toward retaining idled workers (paid downtime,
                               # maintenance, retraining) rather than laying off and
                               # re-hiring -- the standard labor-hoarding response to a
                               # short, foreseeable-end supply shock (see e.g. Fair 1985;
                               # Burnside, Eichenbaum & Rebelo 1993 on short-run labor
                               # hoarding). 0.25 is an illustrative low anchor, not a
                               # series-specific estimate; the true short-run elasticity
                               # for this event is not directly observed.

# ── Channel 1: Lost vehicle output ────────────────────────────────────────────
print("\n── CHANNEL 1: Lost vehicle production")
print("   (Observed YoY deficit × ATP)")
print()

prod = master["auto_prod"]
months_disruption = [
    ("2024-04-01", "2023-04-01", "April 2024",    30),
    ("2024-05-01", "2023-05-01",  "May 2024",       31),
    ("2024-06-01", "2023-06-01",  "June 2024",      30),
    ("2024-07-01", "2023-07-01",  "July 2024",      31),
    ("2024-08-01", "2023-08-01",  "August 2024",    31),
]

total_lost_units = 0
total_value_B    = 0

print(f"  {'Month':15s}  {'Actual':>10}  {'Baseline':>10}  "
      f"{'Lost (k)':>10}  {'Value $B':>10}")
print("  " + "-" * 60)

for dt24, dt23, label, days in months_disruption:
    try:
        v24 = prod.loc[dt24]
        v23 = prod.loc[dt23]
    except KeyError:
        continue
    # DAUPSA units (confirmed from FRED series page): "Thousands of Units,
    # Seasonally Adjusted" -- a MONTHLY count, NOT a seasonally-adjusted
    # annual rate. actual monthly units = DAUPSA_value * 1,000 (no /12).
    actual_mo   = v24 * 1_000
    baseline_mo = v23 * 1_000
    lost        = baseline_mo - actual_mo
    value_B     = (lost * ATP) / 1e9

    total_lost_units += max(lost, 0)
    total_value_B    += max(value_B, 0)

    print(f"  {label:15s}  {actual_mo:>10,.0f}  {baseline_mo:>10,.0f}  "
          f"  {lost:>+9,.0f}  {value_B:>+9.3f}B")

print(f"\n  TOTAL lost units (Apr–Aug 2024):  {total_lost_units:>12,.0f} vehicles")
print(f"  TOTAL production value lost:      ${total_value_B:.2f} billion")

# NOTE: an earlier draft of this script included a "Channel 2: rerouting
# logistics premium" ($800-1,200/vehicle x diverted units). It was dropped
# entirely during the 2026-07-19 audit rather than kept as a labeled
# sensitivity channel: the $800-1,200/vehicle figure was an unsourced author
# guess with no grounding in any published report, and the script's own
# comments already excluded it from every headline figure. A number not
# trusted enough to use is not worth computing and printing with false
# precision -- removed rather than caveated in place.

# ── Channel 2: Employment-hours ────────────────────────────────────────────────
print("\n── CHANNEL 2: Employment-hours impact")
print("   Reported as a [short-run, long-run] RANGE, not a point estimate: a")
print("   textbook output-employment elasticity (0.80, BLS historical) is")
print("   estimated over PERMANENT output changes and is likely a substantial")
print("   OVERSTATEMENT here, since the shock was known ex ante to be")
print("   temporary and the industry was fresh off the Oct-Nov 2023 UAW")
print("   strike -- both push firms toward labor hoarding (idling workers on")
print("   maintenance/retraining rather than laying off) instead of shedding")
print("   headcount. 0.25 is an illustrative short-run lower anchor.")
# DAUPSA unit: value × 1,000 = monthly cars (already a monthly count, not SAAR)
lost_output_pct = total_lost_units / ((prod.loc["2024-04-01":"2024-08-01"].mean() * 1_000) * 5)

def _channel3(elast):
    implied_emp_reduction = AUTO_EMPLOY * lost_output_pct * elast
    lost_hours = implied_emp_reduction * HOURS_PER_MO * 5   # 5 months
    lost_wages_B = (lost_hours * 30) / 1e9                  # $30/hr avg mfg wage (BLS)
    return implied_emp_reduction, lost_hours, lost_wages_B

emp_shortrun, hours_shortrun, wages_shortrun_B = _channel3(OUTPUT_ELAST_SHORTRUN)
emp_longrun,  hours_longrun,  wages_longrun_B  = _channel3(OUTPUT_ELAST_LONGRUN)

print(f"\n  Lost output as % of 5-month production:  {lost_output_pct*100:.1f}%")
print(f"  {'':30s}  {'Short-run (elast=0.25)':>24}  {'Long-run (elast=0.80)':>24}")
print(f"  Implied employment reduction   {emp_shortrun:>20,.0f} workers  {emp_longrun:>20,.0f} workers")
print(f"  Lost employment-hours (Apr-Aug){hours_shortrun:>20,.0f} hours    {hours_longrun:>20,.0f} hours")
print(f"  Lost wage income (@ $30/hr)    {'$'+format(wages_shortrun_B,'.2f')+'B':>24}  {'$'+format(wages_longrun_B,'.2f')+'B':>24}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("WELFARE SUMMARY")
print("=" * 70)
print(f"\n  Direct production shortfall:  ${total_value_B:.2f}B  ({total_lost_units:,.0f} vehicles)")
print(f"  Lost wage income:             ${wages_shortrun_B:.2f}B (short-run) – ${wages_longrun_B:.2f}B (long-run)")
print(f"\n  CONSERVATIVE TOTAL:          ~${total_value_B + wages_shortrun_B:.1f}B  "
      f"(short-run labor-hoarding wage estimate)")
print(f"  LIBERAL TOTAL:               ~${total_value_B + wages_longrun_B:.1f}B  "
      f"(long-run BLS elasticity wage estimate)")
print()
print("  Notes:")
print("  - Production shortfall uses YoY deficit (same month prior year as baseline)")
print("  - ATP = $48,000 (Cox Automotive / KBB average, Q1 2024)")
print("  - Wage-loss channel spans a short-run (labor-hoarding, elast=0.25) to")
print("    long-run (BLS historical, elast=0.80) range; the long-run figure almost")
print("    certainly overstates actual employment impact for a shock of this")
print("    duration and foreseeable end -- see Channel 2 discussion above")
print("  - Does NOT include downstream dealer inventory effects, consumer surplus")
print("    loss, or a rerouting/logistics-cost channel (dropped 2026-07-19: was an")
print("    unsourced per-vehicle guess, not a defensible estimate)")
print("  - Does NOT include insurance, crane removal, or bridge reconstruction costs")
print("  - Reconstruction cost is a separate, one-time infrastructure cost, not part")
print("    of this supply-chain welfare estimate: MDTA's initial 2024 estimate was")
print("    $1.7-1.9B, revised to $4.3-5.2B by late 2025 as design/engineering work")
print("    progressed (Maryland Transportation Authority, updated cost/schedule")
print("    release) -- cited here only for scale comparison, not summed into any total")

# ── Save ────────────────────────────────────────────────────────────────────
welfare = {
    "lost_passenger_cars":         int(total_lost_units),
    "production_loss_passenger_B": round(total_value_B, 3),
    "lost_vehicles_all_est":       int(total_lost_units / 0.25),
    "production_loss_all_B_est":   round((total_lost_units / 0.25) * ATP / 1e9, 2),
    "wage_loss_shortrun_B":        round(wages_shortrun_B, 3),
    "wage_loss_longrun_B":         round(wages_longrun_B, 3),
    "output_elast_shortrun":       OUTPUT_ELAST_SHORTRUN,
    "output_elast_longrun":        OUTPUT_ELAST_LONGRUN,
    "conservative_total_B":        round(total_value_B + wages_shortrun_B, 2),
    "liberal_total_B":             round(total_value_B + wages_longrun_B, 2),
    "atp_usd":                     int(ATP),
    "months_covered":              "Apr-Aug 2024",
    "note_scope":    "DAUPSA=domestic autos only (BEA definition, excludes light trucks); unit=thousands of units/month, seasonally adjusted (NOT an annual rate) -- source: FRED series page for DAUPSA",
    "note_scaling":  "Multiply passenger-car figures by 4 for full-fleet estimate",
    "note_wage_loss": "Wage-loss channel is a [short-run labor-hoarding, long-run BLS] range, not a point estimate -- the long-run (elast=0.80) figure likely overstates actual impact for this known-temporary shock; see Channel 2 discussion in script output",
    "note_dropped_channel": "A 'rerouting logistics premium' channel was dropped 2026-07-19: it was an unsourced $800-1,200/vehicle author guess with no grounding in a published report, already excluded from headline figures before removal",
}
pd.DataFrame(list(welfare.items()), columns=["Parameter","Value"]).to_csv(
    str(DATA_PROC / "welfare_estimate.csv"), index=False)
print("\nSaved: welfare_estimate.csv")
print("Step 9 — Welfare estimate complete ✓")
