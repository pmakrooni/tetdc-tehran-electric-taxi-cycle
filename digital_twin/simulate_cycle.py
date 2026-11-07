"""TETDC Stage 6d — first full digital-twin integration: TETDC-1 drive cycle
through the validated EV model, on flat road and Tehran topography scenarios.

Scenarios:
  S0 flat            : grade 0 (reference)
  S1 northbound      : constant +3.28% (Tehran length-weighted mean |grade|)
  S2 southbound      : constant -3.28%
  S3 roundtrip mix   : alternating +/-3.28% every 300 s (net-zero elevation)
  S4 Valiasr corridor: real DEM profile (stage 2) stretched to cycle distance
Output: cycle_energy_results.json (kWh/100km, SOC, range estimates)
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ev_model import EVDigitalTwin

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CYC = os.path.join(BASE, "data", "processed", "TETDC_1.csv")
OUT = os.path.join(BASE, "data", "processed", "cycle_energy_results.json")

cyc = pd.read_csv(CYC)
v = cyc["v_ms"].to_numpy()
T = len(v)
res = {}

def run(name, grade, twin):
    r = twin.simulate(v, grade, dt=1.0, soc0=1.0)
    cons = r["kWh_per_100km"]
    rng_km = twin.E_use / 3.6e6 / (cons / 100) if cons > 0 else None
    res[name] = {
        "kWh_per_100km": round(float(cons), 2),
        "energy_kWh_cycle": round(float(r["E_batt_kWh"]), 3),
        "soc_end": round(float(r["soc"][-1]), 4),
        "distance_km": round(float(r["distance_m"]) / 1000, 2),
        "range_km_if_sustained": round(float(rng_km), 0) if rng_km else None,
    }
    return r

for season in ["summer", "winter", "shoulder"]:
    twin = EVDigitalTwin(season=season)
    s = {}
    # S0 flat
    run("S0_flat", 0.0, twin)
    s["S0_flat"] = res.pop("S0_flat")
    # S1/S2 sustained-grade stress scenarios
    run("S1_up_3.28pct", 0.0328, twin); s["S1_up_3.28pct"] = res.pop("S1_up_3.28pct")
    run("S2_down_3.28pct", -0.0328, twin); s["S2_down_3.28pct"] = res.pop("S2_down_3.28pct")
    # S3 net-zero mixed grade in DISTANCE domain (equal distance each way,
    #    so elevation gain == descent EXACTLY; time-domain tiling is biased
    #    because speed/idle differ across blocks)
    dist = np.cumsum(v)
    D = dist[-1]
    gmap = np.where(dist <= D / 2, 0.0328, -0.0328)
    # exact zero net: force equal climb/descent distance split
    run("S3_roundtrip_mix", gmap, twin); s["S3_roundtrip_mix"] = res.pop("S3_roundtrip_mix")
    # S4 real Valiasr DEM profile: out-and-back over the FULL corridor.
    #    cycle progress fraction f in [0,1] -> trip chainage s = f * 2L;
    #    s<=L : profile grade (up) ; s>L : -profile(2L-s) (down).
    #    climb = full corridor +506 m, descent = -506 m -> net elevation 0.
    corr = os.path.join(BASE, "data", "processed", "corridor_profiles",
                        "ولی_عصر_خیابان_profile.csv")
    if os.path.exists(corr):
        cp = pd.read_csv(corr).dropna(subset=["grade_pct"])
        s_arr = cp["s_m"].to_numpy()
        g_arr = cp["grade_pct"].to_numpy() / 100
        L = s_arr[-1]
        s_trip = (dist / D) * 2 * L
        gg = np.where(s_trip <= L,
                      np.interp(s_trip, s_arr, g_arr),
                      np.interp(2 * L - s_trip, s_arr, -g_arr[::-1]))
        run("S4_valiasr_updown", gg, twin); s["S4_valiasr_updown"] = res.pop("S4_valiasr_updown")
    res[f"season_{season}"] = s

res["cycle"] = {"length_s": int(T),
                "distance_km": round(float(v.sum() / 1000), 2),
                "v_mean_kmh": round(float(v.mean() * 3.6), 2),
                "source": "TETDC_1.csv (microtrip_recomb, err 5.0%)"}
res["key_finding"] = {
    "flat_summer_kWh_100km": res["season_summer"]["S0_flat"]["kWh_per_100km"],
    "roundtrip_mix_kWh_100km": res["season_summer"]["S3_roundtrip_mix"]["kWh_per_100km"],
    "topo_uplift_pct": round(100 * (res["season_summer"]["S3_roundtrip_mix"]["kWh_per_100km"] /
                                    res["season_summer"]["S0_flat"]["kWh_per_100km"] - 1), 1),
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
