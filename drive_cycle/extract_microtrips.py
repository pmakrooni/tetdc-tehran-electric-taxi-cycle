"""TETDC Stage 6a — micro-trip bank construction from the Porto GPS sample.

Input : data/processed/porto_polylines_sample.parquet (48,415 trips, 15 s GPS)
Method (documented, no fabricated data):
  1) cumulative along-path distance s(t) from haversine steps (native 15 s grid)
  2) 1 Hz speed reconstruction: PCHIP monotone interpolation of s(t) -> v = ds/dt.
     PCHIP is shape-preserving (no position overshoot => no negative speeds);
     this is the standard sparse-GPS speed reconstruction used in drive-cycle
     literature. Kinematic features are computed at 1 Hz; sampling limitation
     (15 s GPS) is documented in the report.
  3) jitter suppression: 3-point median filter (GPS noise)
  4) micro-trip segmentation: stop = v < 0.5 m/s held >= 3 s;
     micro-trip = [preceding idle + motion] between stops
  5) features per prompt section 4: v_mean, v_max, a_max, d_max, P_idle, a_rms
     (+ duration, distance)
Output: microtrips_features.parquet, microtrips_1hz_sample.parquet (traces
        subsample for Markov), microtrip_bank_stats.json
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

SEED = 42
rng = np.random.default_rng(SEED)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(BASE, "data", "processed", "porto_polylines_sample.parquet")
OUTD = os.path.join(BASE, "data", "processed")
TRACES_N = 9000           # Markov trace subsample size
CHUNK = 2500

R_EARTH = 6371000.0
V_STOP = 0.5              # m/s stop threshold
STOP_HOLD = 3             # s
MIN_MT_S, MIN_MT_M = 10, 25
V_CAP = 33.0              # m/s (rule R6 companion)

def haversine_step(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))

def medfilt3(x):
    if len(x) < 3:
        return x
    y = x.copy()
    y[1:-1] = np.minimum(np.minimum(x[1:-1], x[:-2]), np.minimum(x[2:], x[1:-1]))
    # true 3-point median:
    y[1:-1] = np.median(np.stack([x[:-2], x[1:-1], x[2:]]), axis=0)
    return y

def trip_1hz_speed(lonlat):
    try:
        pts = np.stack([np.asarray(p, dtype=float) for p in lonlat])
    except (ValueError, TypeError):
        return None, None
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 4:
        return None, None
    steps = haversine_step(pts[:-1, 0], pts[:-1, 1], pts[1:, 0], pts[1:, 1])
    s15 = np.concatenate([[0.0], np.cumsum(steps)])
    t15 = 15.0 * np.arange(len(pts))
    # PCHIP on unique times (guard against duplicated timestamps)
    t_u, idx = np.unique(t15, return_index=True)
    s_u = s15[idx]
    if len(t_u) < 4:
        return None, None
    pc = PchipInterpolator(t_u, s_u)
    T = int(t_u[-1]) + 1
    t1 = np.arange(T, dtype=float)
    v = np.clip(pc(t1 + 0.5) - pc(t1 - 0.5), 0.0, V_CAP)  # central 1 Hz speed
    v = medfilt3(v)
    return v, s15[-1]

def microtrips(v, min_s=MIN_MT_S, min_m=MIN_MT_M):
    n = len(v)
    stopmask = v < V_STOP
    # stop events held >= STOP_HOLD
    ev = []
    i = 0
    while i < n:
        if stopmask[i]:
            j = i
            while j < n and stopmask[j]:
                j += 1
            if j - i >= STOP_HOLD:
                ev.append((i, j))
            i = j
        else:
            i += 1
    bounds = [0] + [e[0] for e in ev] + [n]
    out = []
    for b in range(len(bounds) - 1):
        a0, a1 = bounds[b], bounds[b + 1]
        seg = v[a0:a1]
        if len(seg) < min_s:
            continue
        d = float(seg.sum())
        if d < min_m:
            continue
        out.append((a0, a1, seg))
    return out

def features(seg):
    T = len(seg)
    acc = np.gradient(seg, 1.0)
    pos_a = acc[acc > 0.1]
    neg_a = acc[acc < -0.1]
    return {
        "duration_s": T,
        "distance_m": float(seg.sum()),
        "v_mean": float(seg.mean()),
        "v_max": float(seg.max()),
        "v_std": float(seg.std()),
        "a_max": float(acc.max()) if len(acc) else 0.0,
        "d_max": float(acc.min()) if len(acc) else 0.0,
        "a_pos_mean": float(pos_a.mean()) if len(pos_a) else 0.0,
        "a_neg_mean": float(neg_a.mean()) if len(neg_a) else 0.0,
        "P_idle": float((seg < V_STOP).mean() * 100),
        "a_rms": float(np.sqrt((acc ** 2).mean())),
    }

def main():
    df = pd.read_parquet(IN)
    n_trips = len(df)
    feat_rows, trace_rows = [], []
    funnel = {"trips": n_trips, "trips_ok": 0, "microtrips": 0,
              "dropped_short_gps": 0, "dropped_frag": 0}
    trace_ids = set(rng.choice(n_trips, size=min(TRACES_N, n_trips), replace=False))
    hour_all = pd.to_datetime(df["start_ts"], unit="s").dt.hour.values

    for c0 in range(0, n_trips, CHUNK):
        chunk = df.iloc[c0:c0 + CHUNK]
        for k, row in enumerate(chunk.itertuples(index=False)):
            gi = c0 + k
            v, dist15 = trip_1hz_speed(row.polyline_lonlat)
            if v is None or len(v) < 15:
                funnel["dropped_short_gps"] += 1
                continue
            funnel["trips_ok"] += 1
            for (a0, a1, seg) in microtrips(v):
                f = features(seg)
                f.update({"trip_id": row.trip_id, "taxi_id": row.taxi_id,
                          "hour": int(hour_all[gi])})
                feat_rows.append(f)
                funnel["microtrips"] += 1
            if gi in trace_ids:
                trace_rows.append(pd.DataFrame({
                    "trip_id": row.trip_id, "t_s": np.arange(len(v)), "v_ms": v}))
        print(f"  chunk {c0 + len(chunk)}/{n_trips}  microtrips={funnel['microtrips']}",
              flush=True)

    feats = pd.DataFrame(feat_rows)
    feats.to_parquet(os.path.join(OUTD, "microtrips_features.parquet"), index=False)
    traces = pd.concat(trace_rows, ignore_index=True)
    traces.to_parquet(os.path.join(OUTD, "microtrips_1hz_sample.parquet"), index=False)

    def qs(x):
        try:
            return {p: round(float(np.percentile(x, p)), 2) for p in (10, 50, 90, 99)}
        except Exception:
            return {}
    stats = {
        "seed": SEED,
        "input_trips": n_trips,
        "funnel": funnel,
        "traces_saved_for_markov": int(len(trace_rows)),
        "microtrip_features_percentiles": {
            "duration_s": qs(feats["duration_s"]),
            "distance_m": qs(feats["distance_m"]),
            "v_mean_ms": qs(feats["v_mean"]),
            "v_max_ms": qs(feats["v_max"]),
            "a_max_ms2": qs(feats["a_max"]),
            "d_max_ms2": qs(feats["d_max"]),
            "P_idle_pct": qs(feats["P_idle"]),
            "a_rms_ms2": qs(feats["a_rms"]),
        },
        "fleet_target_features": {
            "v_mean": round(float(feats["duration_s"].mul(feats["v_mean"]).sum()
                                  / feats["duration_s"].sum()), 3),
            "v_mean_unweighted": round(float(feats["v_mean"].mean()), 3),
            "P_idle": round(float(feats["duration_s"].mul(feats["P_idle"]).sum()
                                  / feats["duration_s"].sum()), 2),
            "a_rms": round(float(feats["duration_s"].mul(feats["a_rms"]).sum()
                                 / feats["duration_s"].sum()), 3),
            "note": "duration-weighted fleet targets used for cycle selection",
        },
        "method": "PCHIP 1 Hz reconstruction; stop>=3 s @ <0.5 m/s; median-3 filter",
    }
    with open(os.path.join(OUTD, "microtrip_bank_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    print(json.dumps(funnel, indent=1))
    print("targets:", json.dumps(stats["fleet_target_features"], indent=1))
    print("saved:", OUTD + "/microtrips_features.parquet",
          OUTD + "/microtrips_1hz_sample.parquet")

if __name__ == "__main__":
    main()
