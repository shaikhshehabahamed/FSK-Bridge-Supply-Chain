"""
Step 11: Inventory Carrying Cost of the Implied JIT Buffer
============================================================
Just-in-time inventory theory (Ohno, 1988) treats safety stock as a
deliberate economic trade-off, not a free insurance policy: every day
of buffer held against disruption risk is a day of parts inventory
that ties up capital, occupies storage, and is exposed to obsolescence
and damage risk (Silver, Pyke & Peterson, 1998). Step 6 estimated that
the industry held approximately 17.6 days of usable buffer stock
during the FSK closure; this step prices that buffer in dollar terms
and weighs its annual carrying cost against the one-time production
loss the buffer actually avoided, closing the loop between the paper's
empirical safety-stock estimate and the carrying-cost side of JIT
theory that motivates near-zero inventory in the first place.

Formal approach
----------------
1. Value of one day of industry-wide automotive parts flow:
     V_day = (baseline full-fleet monthly production) x PARTS_PER_VEH / 30
   PARTS_PER_VEH = $12,000 is the same approximate parts-content-per-
   vehicle estimate used in Step 9 (09_welfare_estimate.py); industry
   cost breakdowns for material/component content of a light vehicle
   fall in the $8,000-$12,000 range (vehicledatabases.com; Cox
   Automotive cost breakdowns), so this figure is treated as an
   upper-of-range, order-of-magnitude estimate, not a precise figure.

2. Dollar value of the observed buffer:
     Buffer_value = V_day x SS_days          (central and [lo, hi] range)

3. Annual carrying cost of holding that buffer as standing policy:
     Carrying_cost = Buffer_value x carrying_rate
   carrying_rate spans 15-30% of average inventory value per year, the
   standard planning range cited in APICS/ASCM supply-chain benchmarking
   guidance and in Silver, Pyke & Peterson (1998); 22.5% (the midpoint of
   the commonly-cited 20-25% "typical" band) is used as the central rate.

4. Benefit side, using the paper's own production-continuity identity
   (Section 4.3): with the observed buffer, April 2024 production fell
   apr_drop_pct = 27.3%; with NO buffer at all (b=0), the same identity
   implies production would have fallen by the full exposure theta_central
   (i.e., Pa = Pn(1-theta)). The buffer's one-time avoided loss is the
   difference between these two counterfactual production shortfalls,
   monetised at ATP (same average transaction price used in Step 9).

5. Break-even framing: the one-time avoided loss is compared against the
   buffer's recurring ANNUAL carrying cost to show how many years of
   carrying cost a single FSK-scale event's avoided loss offsets, and the
   same logic is applied to the incremental buffer (~17.6 -> 30 days) that
   would have been required to eliminate the April 2024 shortfall entirely.

All monetary figures are passenger-car-scope primary estimates (DAUPSA),
with a full-fleet (/0.25) scale-up reported as an upper bound, mirroring
the scaling convention already used in Step 9 and the paper's README.

Limitations (disclosed in the manuscript, not hidden here):
  - PARTS_PER_VEH and the 15-30% carrying-cost rate are industry
    benchmark estimates, not firm-specific disclosures.
  - The analysis is scoped to April 2024 (the single full-closure month
    the SS identity is calibrated to), not the full Apr-Aug disruption
    window used in Step 9's welfare estimate.
  - This prices only the FSK-disruption-avoidance channel of the buffer's
    value; safety stock also hedges demand variability, quality defects,
    and minor supplier delays, none of which are monetised here, so the
    break-even year counts below are a lower bound on the buffer's total
    economic value.
"""
from pathlib import Path as _Path
ROOT      = _Path(__file__).parent.parent
DATA_PROC = ROOT / "data" / "processed"

import pandas as pd
import numpy as np

master = pd.read_csv(str(DATA_PROC / "master.csv"), parse_dates=["date"], index_col="date")
ss     = pd.read_csv(str(DATA_PROC / "ss_summary.csv"))
ss_d   = dict(zip(ss["Parameter"], ss["Value"].astype(float)))
welfare = pd.read_csv(str(DATA_PROC / "welfare_estimate.csv"))
w_d = dict(zip(welfare["Parameter"], welfare["Value"]))

print("=" * 70)
print("STEP 11 — INVENTORY CARRYING COST OF THE IMPLIED JIT BUFFER")
print("=" * 70)

# ── Parameters ────────────────────────────────────────────────────────────────
PARTS_PER_VEH = 12_000            # USD, approx. parts content per vehicle (Step 9 convention)
ATP           = float(w_d["atp_usd"])   # USD, average transaction price (same as Step 9)
D_CLOSURE     = 30                # days, April 2024 closure window (Step 6 convention)
FULL_FLEET_SCALE = 0.25           # passenger cars as share of total light-vehicle production

CARRY_RATE_LOW      = 0.15
CARRY_RATE_CENTRAL  = 0.225
CARRY_RATE_HIGH     = 0.30

theta_central = ss_d["theta_central"]
ss_central    = ss_d["ss_central_days"]
ss_lo         = ss_d["ss_range_low_days"]
ss_hi         = ss_d["ss_range_high_days"]
apr_drop_pct  = ss_d["apr_drop_pct"]

print(f"\nInputs pulled from prior steps:")
print(f"  theta_central (Step 6):      {theta_central:.4f}")
print(f"  SS_central (Step 6):         {ss_central:.2f} days   [{ss_lo:.2f}, {ss_hi:.2f}]")
print(f"  April 2024 YoY drop (Step 6): {apr_drop_pct*100:.2f}%")
print(f"  ATP (Step 9):                ${ATP:,.0f}")
print(f"  PARTS_PER_VEH:               ${PARTS_PER_VEH:,.0f}  (industry estimate, $8k-$12k range)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11A — Baseline monthly production (passenger car + full fleet)
# ─────────────────────────────────────────────────────────────────────────────
prod = master["auto_prod"]
pre_mean_daupsa = prod.loc[master["post"] == 0].mean()

# April-2023 baseline (matches Step 6's "focus month" scope exactly)
# DAUPSA units (FRED series page): "Thousands of Units, Seasonally Adjusted"
# -- a MONTHLY count, not a seasonally-adjusted annual rate. Actual monthly
# units = DAUPSA_value * 1,000 (no /12).
apr23_daupsa = prod.loc["2023-04-01"]
apr23_passenger_monthly = apr23_daupsa * 1_000
apr23_fullfleet_monthly = apr23_passenger_monthly / FULL_FLEET_SCALE

print("\n" + "-" * 70)
print("STEP 11A — Baseline monthly production (April 2023 reference month)")
print("-" * 70)
print(f"  Passenger cars/month:  {apr23_passenger_monthly:,.0f}")
print(f"  Full-fleet est./month: {apr23_fullfleet_monthly:,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11B — Dollar value of the observed buffer
# ─────────────────────────────────────────────────────────────────────────────
monthly_parts_value_B = apr23_fullfleet_monthly * PARTS_PER_VEH / 1e9
daily_parts_value_B   = monthly_parts_value_B / D_CLOSURE

buffer_value_central_B = daily_parts_value_B * ss_central
buffer_value_lo_B      = daily_parts_value_B * ss_lo
buffer_value_hi_B      = daily_parts_value_B * ss_hi

print("\n" + "-" * 70)
print("STEP 11B — Dollar value of the observed safety-stock buffer")
print("-" * 70)
print(f"  Full-fleet parts value/month: ${monthly_parts_value_B:.3f}B")
print(f"  Full-fleet parts value/day:   ${daily_parts_value_B:.3f}B")
print(f"  Buffer value (SS={ss_central:.1f}d, central): ${buffer_value_central_B:.3f}B")
print(f"  Buffer value range [{ss_lo:.1f}, {ss_hi:.1f}]d:  [${buffer_value_lo_B:.3f}B, ${buffer_value_hi_B:.3f}B]")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11C — Annual carrying cost of the observed buffer
# ─────────────────────────────────────────────────────────────────────────────
carry_central_B = buffer_value_central_B * CARRY_RATE_CENTRAL
carry_lo_B      = buffer_value_central_B * CARRY_RATE_LOW
carry_hi_B      = buffer_value_central_B * CARRY_RATE_HIGH

print("\n" + "-" * 70)
print("STEP 11C — Annual carrying cost of holding the observed buffer")
print("-" * 70)
print(f"  Carrying-cost rate assumed: {CARRY_RATE_LOW*100:.0f}-{CARRY_RATE_HIGH*100:.0f}% "
      f"(central {CARRY_RATE_CENTRAL*100:.1f}%) of buffer value per year")
print(f"  Annual carrying cost, central rate: ${carry_central_B:.3f}B/yr")
print(f"  Annual carrying cost, rate range:   [${carry_lo_B:.3f}B, ${carry_hi_B:.3f}B]/yr")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11D — Incremental buffer needed to fully eliminate the April shortfall
# ─────────────────────────────────────────────────────────────────────────────
incremental_days = D_CLOSURE - ss_central   # additional days to reach b=1 (zero decline)
incremental_buffer_value_B = daily_parts_value_B * incremental_days
incremental_carry_central_B = incremental_buffer_value_B * CARRY_RATE_CENTRAL
incremental_carry_lo_B      = incremental_buffer_value_B * CARRY_RATE_LOW
incremental_carry_hi_B      = incremental_buffer_value_B * CARRY_RATE_HIGH

print("\n" + "-" * 70)
print("STEP 11D — Incremental buffer to fully avert the April 2024 shortfall")
print("-" * 70)
print(f"  Additional days needed ({ss_central:.1f} -> {D_CLOSURE}d): {incremental_days:.1f} days")
print(f"  Incremental buffer value:             ${incremental_buffer_value_B:.3f}B")
print(f"  Incremental annual carrying cost, central: ${incremental_carry_central_B:.3f}B/yr")
print(f"  Incremental annual carrying cost, range:   [${incremental_carry_lo_B:.3f}B, ${incremental_carry_hi_B:.3f}B]/yr")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11E — Benefit side: production loss avoided by the observed buffer
# ─────────────────────────────────────────────────────────────────────────────
# Counterfactual identity (Section 4.3): with b=0 (no buffer/rerouting at all),
# Pa = Pn*(1-theta)  ->  the maximum possible decline is theta itself.
# With the OBSERVED buffer, the decline was only apr_drop_pct.
# Avoided fraction = theta - apr_drop_pct (production preserved BY the buffer).
# Remaining fraction = apr_drop_pct (loss that occurred despite the buffer).

avoided_fraction   = theta_central - apr_drop_pct
remaining_fraction = apr_drop_pct
max_possible_fraction = theta_central

if avoided_fraction < 0:
    raise ValueError(
        f"avoided_fraction={avoided_fraction:.4f} < 0: the observed decline "
        f"(apr_drop_pct={apr_drop_pct:.4f}) exceeds the model's max-possible "
        f"decline (theta_central={theta_central:.4f}). This would make the "
        f"buffer's 'avoided loss' negative, which is not a meaningful result -- "
        f"it means theta_central (Step 6) needs revisiting before this step's "
        f"break-even framing is valid, not that this script should silently "
        f"report a nonsensical figure."
    )

avoided_units_passenger   = apr23_passenger_monthly * avoided_fraction
avoided_units_fullfleet   = apr23_fullfleet_monthly * avoided_fraction
avoided_value_passenger_B = avoided_units_passenger * ATP / 1e9
avoided_value_fullfleet_B = avoided_units_fullfleet * ATP / 1e9

remaining_units_passenger   = apr23_passenger_monthly * remaining_fraction
remaining_value_passenger_B = remaining_units_passenger * ATP / 1e9
remaining_units_fullfleet   = apr23_fullfleet_monthly * remaining_fraction
remaining_value_fullfleet_B = remaining_units_fullfleet * ATP / 1e9

print("\n" + "-" * 70)
print("STEP 11E — Production loss avoided BY the observed buffer (vs. zero buffer)")
print("-" * 70)
print(f"  Max possible decline if buffer=0 (=theta): {max_possible_fraction*100:.1f}%")
print(f"  Observed decline with actual buffer:        {remaining_fraction*100:.1f}%")
print(f"  Avoided decline (buffer's benefit):         {avoided_fraction*100:.1f}%")
print(f"  Avoided units (passenger, Apr 2024):  {avoided_units_passenger:,.0f}")
print(f"  Avoided value (passenger, Apr 2024):  ${avoided_value_passenger_B:.3f}B")
print(f"  Avoided value (full-fleet est.):       ${avoided_value_fullfleet_B:.3f}B")
print(f"  Remaining loss despite buffer (passenger): ${remaining_value_passenger_B:.3f}B")
print(f"  Remaining loss despite buffer (full-fleet): ${remaining_value_fullfleet_B:.3f}B")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11F — Break-even framing
# ─────────────────────────────────────────────────────────────────────────────
# How many years of the OBSERVED buffer's carrying cost does this one event's
# avoided loss offset? (passenger-car scope, conservative/primary estimate)
breakeven_years_observed_passenger  = avoided_value_passenger_B / carry_central_B
breakeven_years_observed_fullfleet  = avoided_value_fullfleet_B / carry_central_B

# Same logic for the INCREMENTAL buffer (ss_central -> 30 days) against the
# loss that would have been fully eliminated by holding it.
# NOTE: because the underlying identity (Section 4.3) is exactly linear in
# the shortfall fraction, this ratio is mathematically identical to the
# observed-buffer break-even above (both collapse to s*ATP*theta/(PPV*rate),
# independent of apr_drop_pct) -- confirmed analytically and reproduced
# numerically below. This is reported explicitly in the manuscript as a
# property of the model rather than presented as two independent estimates.
breakeven_years_incremental_passenger = remaining_value_passenger_B / incremental_carry_central_B
breakeven_years_incremental_fullfleet = remaining_value_fullfleet_B / incremental_carry_central_B

print("\n" + "-" * 70)
print("STEP 11F — Break-even: one-time avoided loss vs. recurring annual carrying cost")
print("-" * 70)
print(f"  Observed buffer ({ss_central:.1f}d): avoided loss / annual carrying cost")
print(f"    Passenger-car scope: {breakeven_years_observed_passenger:.2f} years of carrying cost offset by this one event")
print(f"    Full-fleet scope:    {breakeven_years_observed_fullfleet:.2f} years of carrying cost offset by this one event")
print(f"  Incremental buffer (+{incremental_days:.1f}d to fully avert the shortfall):")
print(f"    Passenger-car scope: {breakeven_years_incremental_passenger:.2f} years of carrying cost offset by the loss it would have prevented")
print(f"    Full-fleet scope:    {breakeven_years_incremental_fullfleet:.2f} years of carrying cost offset by the loss it would have prevented")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
result = {
    "parts_per_veh_usd":                  PARTS_PER_VEH,
    "atp_usd":                            ATP,
    "carry_rate_low":                     CARRY_RATE_LOW,
    "carry_rate_central":                 CARRY_RATE_CENTRAL,
    "carry_rate_high":                    CARRY_RATE_HIGH,
    "monthly_parts_value_B_fullfleet":    round(monthly_parts_value_B, 4),
    "daily_parts_value_B_fullfleet":      round(daily_parts_value_B, 4),
    "buffer_value_central_B":             round(buffer_value_central_B, 4),
    "buffer_value_lo_B":                  round(buffer_value_lo_B, 4),
    "buffer_value_hi_B":                  round(buffer_value_hi_B, 4),
    "annual_carry_cost_central_B":        round(carry_central_B, 4),
    "annual_carry_cost_lo_B":             round(carry_lo_B, 4),
    "annual_carry_cost_hi_B":             round(carry_hi_B, 4),
    "incremental_days_to_full_coverage":  round(incremental_days, 2),
    "incremental_buffer_value_B":         round(incremental_buffer_value_B, 4),
    "incremental_annual_carry_cost_central_B": round(incremental_carry_central_B, 4),
    "incremental_annual_carry_cost_lo_B": round(incremental_carry_lo_B, 4),
    "incremental_annual_carry_cost_hi_B": round(incremental_carry_hi_B, 4),
    "avoided_fraction":                   round(avoided_fraction, 4),
    "remaining_fraction":                 round(remaining_fraction, 4),
    "avoided_value_passenger_B":          round(avoided_value_passenger_B, 4),
    "avoided_value_fullfleet_B":          round(avoided_value_fullfleet_B, 4),
    "remaining_value_passenger_B":        round(remaining_value_passenger_B, 4),
    "remaining_value_fullfleet_B":        round(remaining_value_fullfleet_B, 4),
    "breakeven_years_observed_passenger": round(breakeven_years_observed_passenger, 3),
    "breakeven_years_observed_fullfleet": round(breakeven_years_observed_fullfleet, 3),
    "breakeven_years_incremental_passenger": round(breakeven_years_incremental_passenger, 3),
    "breakeven_years_incremental_fullfleet": round(breakeven_years_incremental_fullfleet, 3),
}

pd.DataFrame(list(result.items()), columns=["Parameter", "Value"]).to_csv(
    str(DATA_PROC / "carrying_cost_results.csv"), index=False)

# The publication-ready table (formerly written here directly) is now built
# centrally in 09_tables.py (Table 7), from this same results CSV, so all
# table numbering/output lives in one place -- see save_table() there.

print("\nSaved: data/processed/carrying_cost_results.csv")
print("Step 11 — Inventory carrying cost analysis complete")
