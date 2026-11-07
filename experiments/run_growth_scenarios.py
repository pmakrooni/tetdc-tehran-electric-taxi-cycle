"""TETDC Stage 9c — Fleet growth x infrastructure scenarios (§17, §34-F).

Grid: N_EV in {470, 2000, 5000, 10000, 20000} x plug multiplier
{1,2,4,8,16} (x current 24 plugs / 2.2 MW). Engine: fleet_core.run_fleet
(validated against the stage-7 detailed run within Monte-Carlo noise).

Boundary condition (documented): 100% reliance on the 12 mapped public
sites — depot/home overnight charging is NOT modelled (no data); service
loss therefore bounds the opportunity-charging-only operating model.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fleet_core import run_fleet

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data", "processed", "growth_scenarios.json")

FLEETS = [470, 2000, 5000, 10000, 20000]
PLUG_MULTS = [1, 2, 4, 8, 16]
SEED = 42

rows = []
t0 = time.time()
for F in FLEETS:
    for pm in PLUG_MULTS:
        r = run_fleet(N_EV=F, plug_mult=pm, seed=SEED)
        rows.append(r)
        print(f"F={F:6d} plugs x{pm:2d} ({r['n_plugs_total']:4d}, {r['plug_MW_total']:6.1f} MW) "
              f"-> loss {r['service_loss_pct']:6.2f}%  queue med {r['queue_wait_min_median']:6.1f} min  "
              f"peak {r['peak_charging_MW']:6.2f} MW  E {r['fleet_energy_MWh_day']:8.1f} MWh/d")
print("grid done in", round(time.time() - t0, 1), "s")

# required infrastructure for <=5% service loss per fleet size
req = {}
for F in FLEETS:
    fr = [r for r in rows if r["N_EV"] == F]
    ok = [r for r in fr if r["service_loss_pct"] <= 5.0]
    if ok:
        best = min(ok, key=lambda r: r["plug_mult"])
        req[str(F)] = {"plug_mult": best["plug_mult"],
                       "n_plugs": best["n_plugs_total"],
                       "plug_MW": best["plug_MW_total"],
                       "plugs_per_1000_ev": round(best["n_plugs_total"] / F * 1000, 1)}
    else:
        req[str(F)] = {"plug_mult": None, "note": "not reached within 16x grid"}

out = {
    "purpose": "TETDC stage 9c — growth x infrastructure scenarios",
    "engine": "fleet_core.run_fleet (vectorised, validated vs stage-7 run)",
    "seed": SEED,
    "boundary_condition": "100% public-site opportunity charging; no depot/home charging data exists; service loss bounds this operating model",
    "grid": rows,
    "required_infrastructure_for_5pct_loss": req,
}
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("saved", OUT)
print(json.dumps(req, indent=1))
