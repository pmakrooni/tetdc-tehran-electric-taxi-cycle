"""TETDC Stage 9b — Sensitivity analysis (§20).

Factors on TETDC-1 energy/range (summer twin unless noted):
  1. Road grade (topography): S3 net-zero mixed profile scaled 0-2.5x
     (mean |grade| 0 -> 8.2%)
  2. Traffic congestion: REAL hour-of-day micro-trip groups from the Porto
     1 Hz sample (peak 7-10h vs off-peak midday 11-16h vs night 0-5h);
     20 Monte-Carlo random recombinations per group (~1920 s cycles)
  3. Auxiliary load (A/C): P_aux sweep 0-3500 W
  4. Tornado extras: rolling resistance, mass, drag (literature ranges)
Output: sensitivity_results.json
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "digital_twin"))
from ev_model import EVDigitalTwin

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
OUT = os.path.join(PROC, "sensitivity_results.json")
SEED = 42
rng = np.random.default_rng(SEED)

cyc = pd.read_csv(os.path.join(PROC, "TETDC_1.csv"))
v_cycle = cyc["v_ms"].to_numpy(dtype=float)

GRADE = 0.0328

def s3_grade(v, scale=1.0):
    """Net-zero mixed grade in DISTANCE domain (stage-6d S3 method)."""
    dist = np.cumsum(v)
    D = dist[-1]
    g = np.where(dist <= D / 2, GRADE * scale, -GRADE * scale)
    return g

def run(twin, v, grade):
    r = twin.simulate(v, grade, dt=1.0, soc0=1.0)
    cons = r["kWh_per_100km"]
    rng_km = (twin.E_use / 3.6e6) / (cons / 100) if cons > 0 else None
    return round(float(cons), 2), (round(float(rng_km), 1) if rng_km else None)

twin = EVDigitalTwin(season="summer")
res = {"purpose": "TETDC stage 9b — sensitivity analysis (§20)", "seed": SEED}

# ---------------- 1) road grade (topography) -------------------------------
sens_grade = []
for scale in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
    c, r = run(twin, v_cycle, s3_grade(v_cycle, scale))
    sens_grade.append({"mean_abs_grade_pct": round(scale * GRADE * 100, 2),
                       "kWh_100km": c, "range_km": r})
res["road_grade"] = sens_grade
base_c = sens_grade[2]["kWh_100km"]          # 3.28% = Tehran mean
res["road_grade_base"] = base_c

# ---------------- 2) congestion (hour-of-day micro-trip groups) ------------
mt = pd.read_parquet(os.path.join(PROC, "microtrips_1hz_sample.parquet"))
hour_map = mt.drop_duplicates("trip_id")[["trip_id"]].copy()
hour_map["hour"] = (hour_map.trip_id.astype("int64") // 10**9 + 3600) % 86400 // 3600
hmap = dict(zip(hour_map.trip_id, hour_map.hour))

V_STOP, STOP_S, MIN_S, MIN_M = 0.5, 3, 20, 100

def segment_trip(v):
    """stage-6a rule: stop = v<0.5 m/s held >=3 s; micro-trip = idle+motion."""
    stopmask = v < V_STOP
    stops = []
    i = 0
    while i < len(v):
        if stopmask[i]:
            j = i
            while j < len(v) and stopmask[j]:
                j += 1
            if j - i >= STOP_S:
                stops.append(i)
            i = j
        else:
            i += 1
    segs, start = [], 0
    for st in stops:
        seg = v[start:st + 1]                       # idle+motion up to stop
        if len(seg) >= MIN_S and seg.sum() >= MIN_M:
            segs.append(seg)
        start = st
    tail = v[start:]
    if len(tail) >= MIN_S and tail.sum() >= MIN_M:
        segs.append(tail)
    return segs

groups = {"peak_7_10": set(range(7, 11)),
          "offpeak_11_16": set(range(11, 17)),
          "night_0_5": set(range(0, 6))}
bank = {k: [] for k in groups}

for trip_id, g in mt.groupby("trip_id", sort=False):
    h = hmap.get(trip_id)
    for name, hs in groups.items():
        if h in hs:
            for seg in segment_trip(g["v_ms"].to_numpy()):
                bank[name].append(seg)
            break

congestion = {}
for name, segs in bank.items():
    if not segs:
        continue
    durs = np.array([len(s) for s in segs])
    feats = {
        "n_microtrips": len(segs),
        "v_mean_kmh": round(float(np.concatenate(segs).mean() * 3.6), 2),
        "idle_pct": round(float(np.concatenate([s[s < V_STOP] for s in segs]).size
                              / durs.sum() * 100), 2),
    }
    # 20 Monte-Carlo recombinations ~1920 s
    cons_f, cons_t = [], []
    for k in range(20):
        rng_k = np.random.default_rng(SEED + k)
        parts, tot = [], 0
        idxs = rng_k.integers(0, len(segs), size=int(1920 / 40))  # oversample
        for i in idxs:
            parts.append(segs[i]); tot += len(segs[i])
            if tot >= 1920:
                break
        v = np.concatenate(parts)
        c_f, _ = run(twin, v, 0.0)
        c_t, _ = run(twin, v, s3_grade(v))
        cons_f.append(c_f); cons_t.append(c_t)
    congestion[name] = {**feats,
        "kWh_100km_flat_mean": round(float(np.mean(cons_f)), 2),
        "kWh_100km_flat_std": round(float(np.std(cons_f)), 2),
        "kWh_100km_tehran_mean": round(float(np.mean(cons_t)), 2),
        "kWh_100km_tehran_std": round(float(np.std(cons_t)), 2)}
res["congestion"] = congestion

# ---------------- 3) auxiliary load (A/C) ----------------------------------
sens_aux = []
for paux in [0, 500, 1000, 1800, 2500, 3500]:
    tw = EVDigitalTwin(season="summer")
    tw.P_aux = paux
    c, r = run(tw, v_cycle, s3_grade(v_cycle))
    sens_aux.append({"P_aux_W": paux, "kWh_100km": c, "range_km": r})
res["auxiliary_load"] = sens_aux
res["aux_base"] = sens_aux[3]                   # 1800 W = summer base

# ---------------- 4) tornado extras (literature ranges) --------------------
lt = twin.p["literature_typical"]
tor = []
def sweep(attr, vals, label):
    outs = []
    for val in vals:
        tw = EVDigitalTwin(season="summer")
        setattr(tw, attr, val)
        c, _ = run(tw, v_cycle, s3_grade(v_cycle))
        outs.append(c)
    lo, hi = min(outs), max(outs)
    tor.append({"factor": label, "values": list(vals), "kWh_100km": outs,
                "spread": round(hi - lo, 2),
                "spread_pct_of_base": round((hi - lo) / base_c * 100, 1)})

sweep("Cr", [0.006, 0.009, 0.012], "rolling resistance C_r")
sweep("m", [1980 + 120, 1980 + 195, 1980 + 300], "total mass (kg)")
sweep("Cd", [0.27, 0.29, 0.32], "drag coefficient C_d")
# grade + aux spreads for the same tornado scale
g_lo, g_hi = sens_grade[0]["kWh_100km"], sens_grade[-1]["kWh_100km"]
a_lo, a_hi = sens_aux[0]["kWh_100km"], sens_aux[-1]["kWh_100km"]
tor += [
    {"factor": "road grade 0-8.2% (§20.1)", "spread": round(g_hi - g_lo, 2),
     "spread_pct_of_base": round((g_hi - g_lo) / base_c * 100, 1)},
    {"factor": "A/C 0-3.5 kW (§20.3)", "spread": round(a_hi - a_lo, 2),
     "spread_pct_of_base": round((a_hi - a_lo) / base_c * 100, 1)},
]
con_lo = min(v["kWh_100km_tehran_mean"] for v in congestion.values())
con_hi = max(v["kWh_100km_tehran_mean"] for v in congestion.values())
tor.append({"factor": "congestion peak-vs-night (§20.2)", "spread": round(con_hi - con_lo, 2),
            "spread_pct_of_base": round((con_hi - con_lo) / base_c * 100, 1)})
res["tornado"] = tor

json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("saved", OUT)
print(json.dumps({k: v for k, v in res.items() if k not in ()}, indent=1, ensure_ascii=False)[:3000])
