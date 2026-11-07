"""Terrain scenarios for the nominal vehicle on TETDC-1 (added 9 Oct 2026).

  python analysis/terrain_scenarios.py

For each season:
  * flat;
  * stylised round trip (first half of the distance uphill, second half downhill, net elevation 0)
    at the lower bound (1.86 %), the central value (2.57 %, midpoint of the bounds) and the upper bound
    (3.28 %) of the mean absolute grade;
  * one-way uphill and downhill at the central grade;
  * out-and-back drives over every usable corridor profile (status OK, length >= 3 km): the cycle
    distance is mapped onto the full corridor up and back, as for the Valiasr Street case in
    digital_twin/simulate_cycle.py.
Output: data/processed/terrain_scenarios.json
"""
import json, os, sys, re
import numpy as np, pandas as pd
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "digital_twin"))
from ev_model import EVDigitalTwin
P = os.path.join(BASE, "data", "processed")
v = pd.read_csv(os.path.join(P, "TETDC_1.csv")).v_ms.to_numpy()
dist = np.cumsum(v); D = dist[-1]
G_LO, G_HI = 0.0186, 0.0328; G_MID = round((G_LO + G_HI) / 2, 4)
stats = json.load(open(os.path.join(P, "corridor_stats.json"), encoding="utf-8"))
def kwh(twin, grade):
    return round(float(twin.simulate(v, grade, dt=1.0, soc0=1.0)["kWh_per_100km"]), 2)
def roundtrip(g):
    return np.where(dist <= D / 2, g, -g)
def corridor_grade(cp):
    s_arr = cp["s_m"].to_numpy(); g_arr = cp["grade_pct"].to_numpy() / 100; L = s_arr[-1]
    s_trip = (dist / D) * 2 * L
    return np.where(s_trip <= L, np.interp(s_trip, s_arr, g_arr), np.interp(2 * L - s_trip, s_arr, -g_arr[::-1]))
corridors = {}
for name, st in stats.items():
    if not isinstance(st, dict) or st.get("status") != "OK" or st.get("chain_length_km", 0) < 3:
        continue
    slug = re.sub(r"[^\w\u0600-\u06FF]+", "_", name).strip("_")      # same rule as extract_topography.py
    hit = os.path.join(P, "corridor_profiles", f"{slug}_profile.csv")
    if not os.path.exists(hit):
        continue
    cp = pd.read_csv(hit).dropna(subset=["grade_pct"])
    corridors[name] = dict(file=os.path.basename(hit), length_km=st["chain_length_km"],
                           mean_abs_grade_pct=st["mean_abs_grade_pct"],
                           net_rise_m=round(st["elev_max_m"] - st["elev_min_m"], 1), grade=corridor_grade(cp))
out = {"grades": {"lower": G_LO, "central": G_MID, "upper": G_HI}, "seasons": {}}
for season in ("summer", "winter", "shoulder"):
    tw = EVDigitalTwin(season=season)
    flat = kwh(tw, 0.0)
    r = {"flat": flat,
         "roundtrip_lower": kwh(tw, roundtrip(G_LO)), "roundtrip_central": kwh(tw, roundtrip(G_MID)),
         "roundtrip_upper": kwh(tw, roundtrip(G_HI)),
         "uphill_central": kwh(tw, G_MID), "downhill_central": kwh(tw, -G_MID)}
    r["topo_lower"] = round(r["roundtrip_lower"] - flat, 2)
    r["topo_central"] = round(r["roundtrip_central"] - flat, 2)
    r["topo_upper"] = round(r["roundtrip_upper"] - flat, 2)
    cr = {n: kwh(tw, c["grade"]) for n, c in corridors.items()}
    r["corridors"] = cr
    vals = np.array(list(cr.values()))
    r["corridor_topo_min"] = round(float(vals.min() - flat), 2); r["corridor_topo_max"] = round(float(vals.max() - flat), 2)
    r["corridor_topo_median"] = round(float(np.median(vals) - flat), 2)
    out["seasons"][season] = r
out["corridors"] = {n: {k: c[k] for k in ("file", "length_km", "mean_abs_grade_pct", "net_rise_m")} for n, c in corridors.items()}
out["n_corridors"] = len(corridors)
json.dump(out, open(os.path.join(P, "terrain_scenarios.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k != "corridors"}, indent=1, ensure_ascii=False))
