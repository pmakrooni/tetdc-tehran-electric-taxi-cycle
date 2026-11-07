#!/usr/bin/env python3.13
# -*- coding: utf-8 -*-
"""
TETDC Stage 3 — Calibrated trip-chain parameters for Tehran
============================================================
Reads trip_pattern_stats.json + porto_trip_summary.parquet, writes
data/processed/tehran_chain_params.json — the parameter file that later
stages (fleet simulation / digital twin scenarios) consume.

Tehran enters ONLY through official aggregate anchors:
  - 3.5M taxi trips/day, ~80k active taxis -> 43.75 trips/taxi/day
Everything else (hourly shape, distance/duration distributions) is a
TRANSFERRED Porto pattern — explicitly flagged as assumption.
"""
import json
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "data", "processed")

stats = json.load(open(os.path.join(OUT, "trip_pattern_stats.json"),
                       encoding="utf-8"))
df = pd.read_parquet(os.path.join(OUT, "porto_trip_summary.parquet"),
                     columns=["distance_m", "duration_s", "avg_speed_mps",
                              "start_ts"])

PCTS = [5, 10, 25, 50, 75, 90, 95]


def pct_table(vals, scale=1.0, nd=2):
    return {str(p): round(float(np.percentile(vals, p)) * scale, nd) for p in PCTS}


params = {
    "purpose": ("Tehran EV-taxi trip-chain generator parameters "
                "(roadmap stage 3 deliverable)"),
    "random_seed": 42,
    "tehran_official_anchors": {
        "daily_taxi_trips": 3_500_000,
        "active_taxis": 80_000,
        "trips_per_taxi_per_day": 43.75,
        "ev_fleet_current": "~470 (1000 KMC + 1752 Aion delivered, ~200 circulating)",
    },
    "porto_pattern_transfer": {
        "source": stats["source"],
        "n_clean_trips": stats["n_clean_trips"],
        "cleaning_funnel": stats["cleaning_funnel"],
        "hourly_start_share_weekday": stats["hourly_start_share_weekday"],
        "hourly_start_share_weekend": stats["hourly_start_share_weekend"],
        "trip_distance_km_percentiles": pct_table(df["distance_m"], 1e-3),
        "trip_duration_min_percentiles": pct_table(df["duration_s"], 1 / 60, 1),
        "trip_avg_speed_kmh_percentiles": pct_table(df["avg_speed_mps"], 3.6, 1),
        "porto_trips_per_taxi_per_active_day":
            stats["trips_per_taxi_per_active_day"]["mean"],
        "chain_scale_factor":
            stats["tehran_calibration"]["chain_scale_factor_trips_per_day"],
    },
    "assumptions_and_limits": [
        ("Hourly demand shape, trip-distance and trip-duration distributions "
         "are transferred from Porto (15 s GPS, 2013-14, 442 taxis) — no "
         "public Tehran equivalent exists."),
        ("Porto dataset records only a sample of each taxi's daily trips; "
         "inter-trip idle gaps are therefore inflated by unrecorded trips "
         "and must NOT be transferred directly. Idle modelling for Tehran "
         "will be re-parameterized at stage 6 (rank/queue based)."),
        ("Tehran enters only via aggregate anchors (3.5M trips/day, 80k "
         "taxis). The 43.75 trips/taxi/day implies shorter idle+queue times "
         "than Porto's recorded pattern — consistent with Tehran's "
         "line-taxi operating style."),
        ("No GPS data is fabricated: transferred distributions keep Porto's "
         "real values; scaling is applied at the aggregate level only."),
    ],
    "outputs_for_later_stages": {
        "network_edges_with_grade": "data/processed/edges_with_grade.csv",
        "sumo_network": "data/processed/tehran_sumo.net.xml",
        "corridor_profiles": "data/processed/corridor_profiles/",
        "polyline_sample_for_microtrips":
            "data/processed/porto_polylines_sample.parquet",
    },
}

with open(os.path.join(OUT, "tehran_chain_params.json"), "w",
          encoding="utf-8") as fh:
    json.dump(params, fh, ensure_ascii=False, indent=2)
print("Wrote tehran_chain_params.json")
print(json.dumps(params["porto_pattern_transfer"]["trip_distance_km_percentiles"],
                 indent=1))
print(json.dumps(params["porto_pattern_transfer"]["trip_duration_min_percentiles"],
                 indent=1))
