#!/usr/bin/env python3.13
# -*- coding: utf-8 -*-
"""
TETDC Stage 3 — Porto taxi GPS preprocessing + trip-pattern extraction
=======================================================================
Input : data/raw/train.csv  (ECML/PKDD 2015 Porto taxi, 1,710,670 trips,
        442 taxis, 2013-07-01..2014-06-30, POLYLINE at 15 s sampling)
Output: data/processed/porto_trip_summary.parquet   (clean per-trip summary)
        data/processed/porto_polylines_sample.parquet (3% deterministic sample
        of clean trips' GPS traces — raw material for the future micro-trip bank)
        data/processed/trip_pattern_stats.json      (Porto patterns + Tehran
        calibration factors)

Cleaning rules (each documented in the funnel):
  R1  MISSING_DATA == True                      -> drop
  R2  n_points < 8  (trip < 105 s)              -> drop
  R3  distance  < 500 m                         -> drop (GPS noise floor)
  R4  implied step speed > 55 m/s anywhere      -> drop (GPS outlier)
  R5  < 90% of points inside Porto metro bbox   -> drop
      bbox: lat 41.00..41.35, lon -8.85..-8.35
  R6  avg speed > 33 m/s (120 km/h)             -> drop

NO synthetic data is generated: only real Porto GPS is used; Tehran numbers
enter solely as calibration SCALE factors (official aggregate statistics).

Deterministic (hash-based sampling, RANDOM_SEED = 42).
"""
import json
import os
import time

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO_ROOT, "data", "raw")
OUT = os.path.join(REPO_ROOT, "data", "processed")

# official aggregate statistics of Tehran (research report §4, refs [91][88])
TEHRAN = {
    "daily_taxi_trips": 3_500_000,
    "active_taxis": 80_000,
    "trips_per_taxi_per_day": 3_500_000 / 80_000,   # = 43.75
}

R_EARTH = 6371008.8
SAMPLE_MOD = 100          # keep trips with int(TRIP_ID) % 100 < 3  (~3%)
SAMPLE_KEEP = 3
PORTO_BBOX = (41.00, 41.35, -8.85, -8.35)   # lat_min, lat_max, lon_min, lon_max

CHUNK = 200_000


def haversine_m(lat1, lon1, lat2, lon2):
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def main():
    t0 = time.time()
    funnel = {"total": 0, "R1_missing": 0, "R2_short": 0, "R3_dist": 0,
              "R4_jump": 0, "R5_bbox": 0, "R6_speed": 0, "kept": 0}
    summaries = []
    sample_rows = []       # (trip_id, taxi_id, timestamp, np.array Nx2 float32)

    reader = pd.read_csv(
        os.path.join(RAW, "train.csv"),
        usecols=["TRIP_ID", "TAXI_ID", "TIMESTAMP", "DAY_TYPE",
                 "MISSING_DATA", "POLYLINE"],
        dtype={"TRIP_ID": str, "TAXI_ID": np.int32, "TIMESTAMP": np.int64,
               "DAY_TYPE": str, "MISSING_DATA": str, "POLYLINE": str},
        chunksize=CHUNK)

    for ci, chunk in enumerate(reader):
        for r in chunk.itertuples(index=False):
            funnel["total"] += 1
            if r.MISSING_DATA == "True":
                funnel["R1_missing"] += 1
                continue
            try:
                pts = json.loads(r.POLYLINE)
            except Exception:
                funnel["R1_missing"] += 1
                continue
            n = len(pts)
            if n < 8:
                funnel["R2_short"] += 1
                continue
            arr = np.asarray(pts, dtype=np.float64)      # (n, 2) = lon, lat
            lon, lat = arr[:, 0], arr[:, 1]
            dist = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]).sum()
            if dist < 500.0:
                funnel["R3_dist"] += 1
                continue
            duration = (n - 1) * 15.0
            step_v = dist / duration                     # mean step speed (m/s)
            dts = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]) / 15.0
            if dts.max() > 55.0:
                funnel["R4_jump"] += 1
                continue
            inb = ((lat >= PORTO_BBOX[0]) & (lat <= PORTO_BBOX[1]) &
                   (lon >= PORTO_BBOX[2]) & (lon <= PORTO_BBOX[3]))
            if inb.mean() < 0.9:
                funnel["R5_bbox"] += 1
                continue
            if step_v > 33.0:
                funnel["R6_speed"] += 1
                continue
            funnel["kept"] += 1
            summaries.append({
                "trip_id": r.TRIP_ID,
                "taxi_id": r.TAXI_ID,
                "start_ts": r.TIMESTAMP,
                "day_type": r.DAY_TYPE,
                "n_points": n,
                "distance_m": float(dist),
                "duration_s": float(duration),
                "avg_speed_mps": float(dist / duration),
                "start_lon": float(lon[0]), "start_lat": float(lat[0]),
                "end_lon": float(lon[-1]), "end_lat": float(lat[-1]),
            })
            if int(r.TRIP_ID) % SAMPLE_MOD < SAMPLE_KEEP:
                sample_rows.append((r.TRIP_ID, r.TAXI_ID, r.TIMESTAMP,
                                    arr.astype(np.float32)))
        print(f"  chunk {ci}: total={funnel['total']:,} kept={funnel['kept']:,} "
              f"({time.time() - t0:.0f}s)")

    df = pd.DataFrame(summaries)
    print(f"\nClean trips: {len(df):,} of {funnel['total']:,}")

    # ---------------- trip chaining (per taxi, chronological) ----------------
    df = df.sort_values(["taxi_id", "start_ts"]).reset_index(drop=True)
    end_ts = df["start_ts"] + df["duration_s"]
    same_taxi = df["taxi_id"].shift(-1) == df["taxi_id"]
    gap = (df["start_ts"].shift(-1) - end_ts).where(same_taxi)

    # shift boundary: gap > 6 h  ->  new shift (rest / overnight)
    new_shift = (~same_taxi) | (gap > 6 * 3600)
    shift_id = new_shift.cumsum()
    df["shift_id"] = shift_id

    # idle between consecutive trips inside a shift (positive gaps only)
    idle = gap[(gap > 0) & same_taxi]

    # trips per taxi per ACTIVE day (days with >=1 trip)
    df["date"] = pd.to_datetime(df["start_ts"], unit="s").dt.date
    per_day = df.groupby(["taxi_id", "date"]).size()
    hour = pd.to_datetime(df["start_ts"], unit="s").dt.hour
    hourly = df.groupby(hour).size()
    hourly_share = (hourly / hourly.sum()).round(4).to_dict()

    # weekday vs weekend profile
    dow = pd.to_datetime(df["start_ts"], unit="s").dt.dayofweek
    weekday_share = (df[dow < 5].groupby(hour[dow < 5]).size()
                     / (dow < 5).sum()).round(4).to_dict()
    weekend_share = (df[dow >= 5].groupby(hour[dow >= 5]).size()
                     / (dow >= 5).sum()).round(4).to_dict()

    # ---------------- Tehran calibration (pattern transfer) ----------------
    porto_tpd = float(per_day.mean())
    stats = {
        "source": "Porto ECML/PKDD 2015 taxi dataset (UCI ML Repo id 339)",
        "sampling_interval_s": 15,
        "period": "2013-07-01 .. 2014-06-30",
        "cleaning_funnel": funnel,
        "cleaning_rules": {
            "R1": "MISSING_DATA == True",
            "R2": "n_points < 8 (trip < 105 s)",
            "R3": "distance < 500 m",
            "R4": "step speed > 55 m/s (GPS outlier)",
            "R5": "<90% points inside Porto metro bbox",
            "R6": "avg speed > 33 m/s",
        },
        "n_taxis": int(df["taxi_id"].nunique()),
        "n_clean_trips": int(len(df)),
        "distance_km": {
            "p10": round(float(np.percentile(df["distance_m"], 10)) / 1000, 2),
            "median": round(float(np.percentile(df["distance_m"], 50)) / 1000, 2),
            "mean": round(float(df["distance_m"].mean()) / 1000, 2),
            "p90": round(float(np.percentile(df["distance_m"], 90)) / 1000, 2),
            "p99": round(float(np.percentile(df["distance_m"], 99)) / 1000, 2),
        },
        "duration_min": {
            "p10": round(float(np.percentile(df["duration_s"], 10)) / 60, 1),
            "median": round(float(np.percentile(df["duration_s"], 50)) / 60, 1),
            "mean": round(float(df["duration_s"].mean()) / 60, 1),
            "p90": round(float(np.percentile(df["duration_s"], 90)) / 60, 1),
        },
        "avg_speed_kmh": {
            "median": round(float(np.percentile(df["avg_speed_mps"], 50)) * 3.6, 1),
            "mean": round(float(df["avg_speed_mps"].mean()) * 3.6, 1),
            "p90": round(float(np.percentile(df["avg_speed_mps"], 90)) * 3.6, 1),
        },
        "trips_per_taxi_per_active_day": {
            "mean": round(porto_tpd, 2),
            "p50": round(float(per_day.median()), 2),
            "p90": round(float(np.percentile(per_day, 90)), 2),
        },
        "idle_between_trips_in_shift_s": {
            "n": int(len(idle)),
            "median": round(float(idle.median()), 0),
            "p75": round(float(np.percentile(idle, 75)), 0),
            "p90": round(float(np.percentile(idle, 90)), 0),
            "share_gt_10min": round(float((idle > 600).mean()), 4),
        },
        "shifts": {
            "n_shifts": int(shift_id.max()),
            "trips_per_shift_median": round(float(df.groupby("shift_id").size().median()), 1),
            "shift_duration_h_median": round(float(
                (df.groupby("shift_id")["duration_s"].sum() + 600).median() / 3600), 2),
        },
        "hourly_start_share": {int(k): v for k, v in hourly_share.items()},
        "hourly_start_share_weekday": {int(k): v for k, v in weekday_share.items()},
        "hourly_start_share_weekend": {int(k): v for k, v in weekend_share.items()},
        "tehran_calibration": {
            "official_daily_taxi_trips": TEHRAN["daily_taxi_trips"],
            "official_active_taxis": TEHRAN["active_taxis"],
            "tehran_trips_per_taxi_per_day": round(TEHRAN["trips_per_taxi_per_day"], 2),
            "porto_trips_per_taxi_per_active_day": round(porto_tpd, 2),
            "chain_scale_factor_trips_per_day":
                round(TEHRAN["trips_per_taxi_per_day"] / porto_tpd, 3),
            "method": ("Tehran enters ONLY as aggregate scale anchors "
                       "(3.5M trips/day, 80k taxis). Hourly shape, distance/"
                       "duration distributions and idle gaps are transferred "
                       "from Porto patterns — an assumption to be tested in "
                       "the sensitivity analysis of later stages."),
        },
        "random_seed": SEED,
        "sample_keep_fraction": SAMPLE_KEEP / SAMPLE_MOD,
    }

    df.drop(columns=["date"]).to_parquet(
        os.path.join(OUT, "porto_trip_summary.parquet"), index=False)

    with open(os.path.join(OUT, "trip_pattern_stats.json"), "w",
              encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    if sample_rows:
        sdf = pd.DataFrame({
            "trip_id": [r[0] for r in sample_rows],
            "taxi_id": np.array([r[1] for r in sample_rows], dtype=np.int32),
            "start_ts": np.array([r[2] for r in sample_rows], dtype=np.int64),
            "polyline_lonlat": [r[3].tolist() for r in sample_rows],
        })
        sdf.to_parquet(os.path.join(OUT, "porto_polylines_sample.parquet"),
                       index=False)
        stats["sample_trips_saved"] = int(len(sdf))
        with open(os.path.join(OUT, "trip_pattern_stats.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(stats, fh, ensure_ascii=False, indent=2)

    print(json.dumps({k: v for k, v in stats.items()
                      if k in ("cleaning_funnel", "n_taxis", "n_clean_trips",
                               "distance_km", "avg_speed_kmh")},
                     indent=2))
    print(f"\nDone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
