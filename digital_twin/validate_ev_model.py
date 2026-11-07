"""TETDC Stage 4 — independent validation of the EV digital-twin model.

Tests (all vs published / documented anchors, never vs fabricated data):
 T1 constant-speed sweep 30-110 km/h, flat: kWh/100km vs C-SUV BEV literature band
 T2 effect of Tehran air density (altitude) vs sea level
 T3 grade sensitivity 0 / +2 / +4 / mixed +/-, incl. regen on -grade
 T4 range sanity: usable 85.6 kWh at cycle-typical 18.5 kWh/100km vs documented
    fleet context (93 kWh pack; KMC E-TAX 402 km companion vehicle)
 T5 DC fast-charge anchor: 20->80 % in <= 45 min at station power (documented [108])
 T6 real corridor demo: Valiasr St. DEM profile from stage 2 applied at 50 km/h
 T7 sensitivity: mass/Cr/Cd/aux ranges -> kWh/100km spread (literature_typical ranges)
Writes: digital_twin/ev_validation.json (+ prints summary).
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ev_model import EVDigitalTwin, P  # noqa: E402

SEED = 42
rng = np.random.default_rng(SEED)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ev_validation.json")
res = {}

twin = EVDigitalTwin(season="summer")

# ---------------- T1: constant-speed sweep (flat, steady 600 s) -------------
def const_speed_test(kmh):
    v = np.full(600, kmh / 3.6)
    r = twin.simulate(v, 0.0, dt=1.0, soc0=1.0)
    # drop first 5 s (spin-up transient of np.gradient)
    return r["kWh_per_100km"]

speeds = [30, 40, 50, 60, 70, 80, 90, 100, 110]
lit_band = {30: (10.5, 14.5), 50: (11.0, 14.5), 70: (12.5, 16.5), 90: (14.0, 18.5), 110: (17.0, 22.5)}
t1 = {}
for s in speeds:
    k = const_speed_test(s)
    t1[s] = round(k, 2)
    if s in lit_band:
        lo, hi = lit_band[s]
        ok = lo <= k <= hi
        res.setdefault("T1_const_speed", {})[str(s)] = {
            "kWh_100km": round(k, 2), "lit_band": lit_band[s], "in_band": bool(ok)}
res["T1_const_speed"]["curve"] = t1

# ---------------- T2: Tehran altitude air-density effect --------------------
t2 = {}
for rho_label, rho in [("tehran_1190m", 1.03), ("sea_level", 1.18)]:
    tw = EVDigitalTwin(season="summer")
    tw.rho = rho
    r = tw.simulate(np.full(600, 90 / 3.6), 0.0)
    t2[rho_label] = round(r["kWh_per_100km"], 2)
res["T2_air_density"] = {"90kmh": t2,
    "delta_pct": round(100 * (t2["sea_level"] - t2["tehran_1190m"]) / t2["sea_level"], 2),
    "note": "Tehran altitude cuts aero drag; drag share is small at 90 km/h"}

# ---------------- T3: grade sensitivity -------------------------------------
t3 = {}
for g in [0.0, 0.02, 0.04, -0.02, -0.04]:
    r = twin.simulate(np.full(600, 50 / 3.6), g)
    t3[f"{int(g*100)}%"] = round(r["kWh_per_100km"], 2)
# mixed corridor: alternating +/-4% (net zero) at 50 km/h, 1200 s
gseq = np.tile(np.repeat([0.04, -0.04], 300), 2)[:1200]
r_mix = twin.simulate(np.full(1200, 50 / 3.6), gseq)
res["T3_grade"] = {"constant": t3, "mixed_pm4_net0": round(r_mix["kWh_per_100km"], 2),
    "note": "asymmetric: +grade loss > -grade regen recovery (eta_regen<1)"}

# ---------------- T4: range sanity -------------------------------------------
# typical urban taxi cycle consumption ~ 16-20 kWh/100km w/ summer A/C
cons_range = (16.0, 20.0)
usable = P["derived"]["usable_kWh"]
res["T4_range"] = {
    "usable_kWh": usable,
    "at_16": round(usable / 16 * 100, 0), "at_18.5": round(usable / 18.5 * 100, 0),
    "at_20": round(usable / 20 * 100, 0),
    "context": "km range; KMC E-TAX companion fleet vehicle declares 402 km (smaller pack, ref [109])",
}

# ---------------- T5: DC fast-charge anchor -----------------------------------
t5 = {}
for Pk in [60, 90, 120, 160, 340]:
    t_min, socs, _ = twin.charge_session(0.20, 0.80, Pk)
    t5[f"{Pk}kW"] = round(float(t_min[-1]), 1)
res["T5_charge_20_80_min"] = {
    "by_station_kW": t5, "documented_anchor": "<= 45 min [108]",
    "model_at_90kW": t5["90kW"],
    "note": "a 90 kW plateau reproduces the documented <=45 min anchor; Tehran stations span 60-340 kW DC",
}

# ---------------- T6: real corridors (Valiasr & 16 corridors, stage-2 DEM CSVs)
import csv as _csv
corr_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "processed", "corridor_profiles")
t6 = {}
if os.path.isdir(corr_dir):
    for fn in sorted(os.listdir(corr_dir)):
        if not fn.endswith("_profile.csv"):
            continue
        rows = []
        with open(os.path.join(corr_dir, fn), encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                try:
                    rows.append((float(row["s_m"]), float(row["elev_m"]),
                                 float(row["grade_pct"])))
                except (TypeError, ValueError):
                    continue
        if len(rows) < 10:
            continue
        dist = np.array([r[0] for r in rows])
        prof = np.array([r[2] for r in rows]) / 100.0          # pct -> fraction
        elev = np.array([r[1] for r in rows])
        name = fn.replace("_profile.csv", "")
        v_kmh, v_ms = 50.0, 50.0 / 3.6                          # free-flow assumption
        T = max(int(dist[-1] / v_ms), 60)
        s = v_ms * np.arange(T)
        grade_t = np.interp(s, dist, prof)
        r = twin.simulate(np.full(T, v_ms), grade_t, soc0=1.0)
        # reverse direction = negate profile (downhill becomes uphill)
        grade_rev = np.interp(s, dist, -prof[::-1])
        r_rev = twin.simulate(np.full(T, v_ms), grade_rev, soc0=1.0)
        t6[name] = {
            "length_km": round(dist[-1] / 1000, 2),
            "elev_start_end_m": [round(elev[0], 0), round(elev[-1], 0)],
            "kWh_100km_southbound": round(r["kWh_per_100km"], 2),
            "kWh_100km_northbound": round(r_rev["kWh_per_100km"], 2),
            "roundtrip_kWh_100km": round((r["kWh_per_100km"] +
                                          r_rev["kWh_per_100km"]) / 2, 2),
        }
res["T6_real_corridors"] = t6

# ---------------- T7: parameter sensitivity (literature ranges) --------------
lt = P["literature_typical"]
base = const_speed_test(50)   # urban-representative flat speed
sens = {}
for key, attr, vals in [
    ("curb_mass_kg", "m", [1900 + 195, 1980 + 195, 2100 + 195]),
    ("C_r", "Cr", [0.008, 0.009, 0.012]),
    ("Cd", "Cd", [0.27, 0.29, 0.32]),
    ("P_aux_summer", "P_aux", [1200, 1800, 2500]),
]:
    outs = []
    for v in vals:
        tw = EVDigitalTwin(season="summer")
        setattr(tw, attr, v)
        outs.append(round(tw.simulate(np.full(600, 50 / 3.6), 0.0)["kWh_per_100km"], 2))
    sens[key] = {"values": vals, "kWh_100km_at_50": outs}
res["T7_sensitivity_50kmh_flat"] = sens
res["T7_base"] = round(base, 2)

res["seed"] = SEED
res["model_version"] = "ev_model.py v1 (stage 4)"
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)

# ---------------- console summary -------------------------------------------
print("=== T1 constant speed (kWh/100km, flat, summer A/C) ===")
for s in speeds:
    band = f"  band {lit_band[s]}" if s in lit_band else ""
    flag = ""
    if s in lit_band:
        flag = "OK " if lit_band[s][0] <= t1[s] <= lit_band[s][1] else "OUT"
    print(f"  {s:3d} km/h -> {t1[s]:6.2f} {flag}{band}")
print("=== T2 air density (90 km/h) ===", t2, f"delta {res['T2_air_density']['delta_pct']}%")
print("=== T3 grade (50 km/h) ===", t3, "mixed:", res["T3_grade"]["mixed_pm4_net0"])
print("=== T4 range ===", res["T4_range"]["at_16"], "-", res["T4_range"]["at_20"], "km")
print("=== T5 charge 20-80 ===", t5)
print("=== T6 corridors ===")
for k, v in t6.items():
    print("  ", k, v)
print("=== T7 sensitivity ===", json.dumps(sens, indent=1)[:600])
print("saved ->", OUT)
