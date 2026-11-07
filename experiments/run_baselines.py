"""TETDC Stage 8 — Baseline comparison (§13, §15): TETDC vs WLTC vs NEDC.

Authentic traces from the `wltp` package (UNECE GTR-15 reference
implementation, PyPI 'wltp' byankostis/wltp):
  - WLTC class 3b (PMR>34, vmax>=120 — correct class for C-SUV BEV), 1801 s
  - NEDC (4 x ECE-15 + EUDC), 1180 s
Type-approval convention: cycles run on a FLAT dyno (grade = 0).

Experimental matrix (per season):
  A. cycle-shape effect (both flat):
       TETDC-flat vs WLTC-flat vs NEDC-flat
  B. topography effect (same Tehran mixed-grade profile, S3 net-zero,
     mean |grade| 3.28% tiled in DISTANCE domain):
       TETDC-Tehran vs WLTC-Tehran vs NEDC-Tehran
  C. fleet-manager error (§15): using WLTC-flat (certification number)
     instead of TETDC-Tehran (reality) -> exact range & energy error.
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "digital_twin"))
from ev_model import EVDigitalTwin

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
OUT = os.path.join(PROC, "baseline_comparison.json")
SEED = 42

# ------------------------------------------------------- authentic traces
from wltp.cycles import class3, nedc
v_wltc = np.array(class3.class_data_b()["cycle"], dtype=float) / 3.6   # km/h -> m/s
v_nedc = np.array(nedc.cycle_data()["cycle"], dtype=float) / 3.6

cyc = pd.read_csv(os.path.join(PROC, "TETDC_1.csv"))
v_tetdc = cyc["v_ms"].to_numpy(dtype=float)

# sanity anchors (published values)
checks = {
    "wltc_len_s": len(v_wltc),       # expect 1801
    "wltc_dist_km": round(float(np.trapezoid(v_wltc, dx=1)) / 1000, 2),  # expect ~23.27
    "wltc_vmax_kmh": round(float(v_wltc.max()) * 3.6, 1),               # expect 131.3
    "nedc_len_s": len(v_nedc),       # expect 1180
    "nedc_dist_km": round(float(np.trapezoid(v_nedc, dx=1)) / 1000, 2),  # expect 11.01
}
print("authenticity anchors:", checks)

# ------------------------------------------------ Tehran mixed-grade tile
GRADE = 0.0328  # Tehran length-weighted mean |grade| (stage-2 DEM result)

def tehran_grade(v):
    """S3-style net-zero mixed grade in DISTANCE domain (stage-6d method)."""
    dist = np.cumsum(v)
    D = dist[-1]
    return np.where(dist <= D / 2, GRADE, -GRADE)

def kinematic_features(v, dt=1.0):
    a = np.gradient(v, dt)
    return {
        "duration_s": int(len(v)),
        "distance_km": round(float(np.trapezoid(v, dx=dt)) / 1000, 2),
        "v_mean_kmh": round(float(v.mean()) * 3.6, 2),
        "v_max_kmh": round(float(v.max()) * 3.6, 2),
        "a_max_ms2": round(float(a.max()), 3),
        "d_max_ms2": round(float(a.min()), 3),
        "a_rms_ms2": round(float(np.sqrt((a ** 2).mean())), 3),
        "P_idle_pct": round(float((v < 0.5).mean() * 100), 2),
    }

def run_cycle(twin, v, grade):
    r = twin.simulate(v, grade, dt=1.0, soc0=1.0)
    E = r["E_batt_kWh"]
    dist_km = r["distance_m"] / 1000
    # regen share: recovered braking energy / positive traction energy
    P_wheel = r["P_wheel"]
    E_trac = float(np.trapezoid(np.where(P_wheel > 0, P_wheel, 0), dx=1)) / 3.6e6
    E_regen = -float(np.trapezoid(np.where(P_wheel < 0, P_wheel, 0), dx=1)) / 3.6e6 * 0.70
    cons = E / dist_km * 100
    rng = (twin.E_use / 3.6e6) / (cons / 100) if cons > 0 else None
    return {
        "kWh_per_100km": round(float(cons), 2),
        "Wh_per_km": round(float(cons) * 10, 1),
        "energy_kWh_cycle": round(float(E), 3),
        "range_km": round(float(rng), 1) if rng else None,
        "regen_share_pct": round(E_regen / max(E_trac, 1e-9) * 100, 1),
        "distance_km": round(dist_km, 2),
    }

cycles = {
    "TETDC": v_tetdc,
    "WLTC": v_wltc,
    "NEDC": v_nedc,
}

res = {"authenticity_anchors": checks,
       "grade_mean_abs_pct": GRADE * 100,
       "sources": {
           "WLTC": "wltp PyPI package (UNECE GTR-15 reference impl), class 3b, 1801 s",
           "NEDC": "wltp PyPI package, 4xECE-15 + EUDC, 1180 s",
           "TETDC": "this project stage 6b (micro-trip recombination), 1920 s",
       },
       "kinematics": {k: kinematic_features(v) for k, v in cycles.items()},
       "seasons": {}}

for season in ["summer", "winter", "shoulder"]:
    twin = EVDigitalTwin(season=season)
    s = {}
    for name, v in cycles.items():
        s[f"{name}_flat"] = run_cycle(twin, v, 0.0)
        s[f"{name}_tehran"] = run_cycle(twin, v, tehran_grade(v))
    # §15 fleet-manager error: certification (WLTC-flat) vs reality (TETDC-Tehran)
    e_wltp, e_real = s["WLTC_flat"], s["TETDC_tehran"]
    s["fleet_manager_error"] = {
        "energy_error_pct": round((e_wltp["kWh_per_100km"] - e_real["kWh_per_100km"])
                                  / e_real["kWh_per_100km"] * 100, 1),
        "range_error_pct": round((e_wltp["range_km"] - e_real["range_km"])
                                 / e_real["range_km"] * 100, 1),
        "range_overestimate_km": round(e_wltp["range_km"] - e_real["range_km"], 1),
    }
    # decomposition: shape effect vs topography effect
    s["decomposition"] = {
        "shape_effect_kWh100km": round(s["TETDC_flat"]["kWh_per_100km"]
                                       - s["WLTC_flat"]["kWh_per_100km"], 2),
        "topo_effect_kWh100km": round(s["TETDC_tehran"]["kWh_per_100km"]
                                      - s["TETDC_flat"]["kWh_per_100km"], 2),
        "total_gap_kWh100km": round(s["TETDC_tehran"]["kWh_per_100km"]
                                    - s["WLTC_flat"]["kWh_per_100km"], 2),
    }
    res["seasons"][season] = s
    print(season, json.dumps(s["fleet_manager_error"]), json.dumps(s["decomposition"]))

json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("saved", OUT)

# quick view: summer table
print("\n--- summer ---")
for k, v in res["seasons"]["summer"].items():
    if isinstance(v, dict) and "kWh_per_100km" in v:
        print(f"{k:18s} {v['kWh_per_100km']:8.2f} kWh/100km  range {v['range_km']:7.1f} km  regen {v['regen_share_pct']}%")
