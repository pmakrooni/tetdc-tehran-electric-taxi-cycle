#!/usr/bin/env python3.13
# -*- coding: utf-8 -*-
"""
TETDC Stage 2 — Topography extraction from Copernicus DEM GLO-30
=================================================================
Input : data/raw/dem_N35_E050.tif, dem_N35_E051.tif  (Copernicus DSM 30 m, AWS open data)
        data/processed/tehran_network_nodes.csv / tehran_network_edges.csv (Stage 1)
Output: data/processed/edges_with_grade.csv     (per-edge road grade)
        data/processed/dem_summary.json         (elevation & grade statistics)
        data/processed/corridor_stats.json      (per-corridor grade statistics)
        data/processed/corridor_profiles/*.csv  (s, elevation, smoothed grade)

Method: sample DEM at each network node; edge grade = (H_v - H_u) / length.
Note  : Copernicus GLO-30 is a SURFACE model (DSM: buildings/canopy included).
        For corridor profiles the grade is smoothed with a centered 250 m window
        to suppress DSM roughness; |grade| > 30% flagged as artefact-suspect.
Deterministic. RANDOM_SEED = 42.
"""
import json
import os
import re

import numpy as np
import pandas as pd
import rasterio
import networkx as nx

SEED = 42
np.random.seed(SEED)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO_ROOT, "data", "raw")
OUT = os.path.join(REPO_ROOT, "data", "processed")
PROF = os.path.join(OUT, "corridor_profiles")
os.makedirs(PROF, exist_ok=True)

# ---------------------------------------------------------------- DEM helpers
TILES = [
    os.path.join(RAW, "dem_N35_E051.tif"),   # lon 51-52 E  <- most of Tehran
    os.path.join(RAW, "dem_N35_E050.tif"),   # lon 50-51 E  <- western margin
]
_srcs = [rasterio.open(t) for t in TILES]


def sample_elevation(lons, lats):
    """Sample elevation [m] for arrays of WGS84 coords (vectorized per tile)."""
    n = len(lons)
    out = np.full(n, np.nan, dtype=np.float64)
    for src in _srcs:
        b = src.bounds
        m = (lons >= b.left) & (lons <= b.right) & (lats >= b.bottom) & (lats <= b.top)
        if m.sum() == 0:
            continue
        coords = list(zip(lons[m], lats[m]))
        vals = np.array([float(v[0]) for v in src.sample(coords)])
        vals[vals < -1000] = np.nan          # nodata guard
        out[m] = vals
    return out


# ---------------------------------------------------------------- normalize names
_STRIP_WORDS = [
    "خیابان", "بزرگراه", "شهید", "آیت‌الله", "آیتالله", "اسلامی", "حضرت",
    "دکتر", "مهدی", "(ع)", "(ره)", "مینی",
]
_ZWNJ = "\u200c"


def norm_name(s):
    if not s or not isinstance(s, str):
        return ""
    s = s.replace(_ZWNJ, " ")
    s = re.sub(r"\([^)]*\)", " ", s)          # remove parenthetical suffixes
    for w in _STRIP_WORDS:
        s = s.replace(w + " ", " ").replace(" " + w, " ").replace(w, " ")
    s = s.replace("آل احمد", "جلالآلاحمد")     # unify جلال آل‌احمد
    return re.sub(r"\s+", "", s.strip())


# ---------------------------------------------------------------- corridors
CORRIDORS = {
    "ولی‌عصر (خیابان)":            ["ولیعصر"],
    "ولی‌عصر شمالی (خیابان)":      ["ولیعصر"],
    "چمران (بزرگراه)":             ["چمران"],
    "مدرس (بزرگراه)":              ["مدرس"],
    "همت (بزرگراه)":               ["همت"],
    "رسالت–شهید سلیمانی (بزرگراه)": ["رسالت", "سلیمانی"],
    "نیایش (بزرگراه)":             ["نیایش"],
    "صدر–بابایی (بزرگراه)":        ["صدر", "بابایی"],
    "امام علی (بزرگراه)":          ["امامعلی"],
    "آزادگان (بزرگراه)":           ["آزادگان"],
    "شریعتی (خیابان)":             ["شریعتی"],
    "انقلاب (خیابان)":             ["انقلاب"],
    "آزادی (خیابان)":              ["آزادی"],
    "کاشانی (بزرگراه)":            ["کاشانی"],
    "جلال آل‌احمد (بزرگراه)":       ["جلالآلاحمد"],
    "اشرفی اصفهانی (بزرگراه)":     ["اشرفیاصفهانی"],
}
SMOOTH_WINDOW_M = 250.0   # centered smoothing window for corridor profiles
ARTEFACT_GRADE = 30.0     # |grade|% above this = suspect DSM artefact


def longest_spine(G):
    """Find the two farthest degree-1 nodes and return the path between them."""
    ends = [n for n, d in G.degree() if d == 1]
    if len(ends) < 2:
        ends = list(G.nodes())
    best_len, best_path = -1, None
    for a in ends[:150]:                      # cap for speed
        dist, paths = nx.single_source_dijkstra(G, a, weight="length")
        for b, d in dist.items():
            if d > best_len and b in paths:
                best_len, best_path = d, paths[b]
    return best_path


def binned_profile(ce, node_pos, node_elev, axis="lat", bin_m=150.0,
                   box=None, smooth_bins=5):
    """
    Robust along-corridor profile for fragmented / dual-carriageway corridors.
    Collect nodes of matched edges (optionally inside a lat/lon box), bin along
    `axis`, take median elevation per bin -> (s, H, smoothed grade).
    Works regardless of OSM way fragmentation.
    """
    uv = set()
    for r in ce.itertuples():
        uv.add(int(r.u)); uv.add(int(r.v))
    lat = np.array([node_pos[n][1] for n in uv])
    lon = np.array([node_pos[n][0] for n in uv])
    ele = np.array([node_elev.get(n, np.nan) for n in uv])
    if box is not None:
        keep = ((lat >= box[0]) & (lat <= box[1]) & (lon >= box[2]) & (lon <= box[3])
                & np.isfinite(ele))
        lat, lon, ele = lat[keep], lon[keep], ele[keep]
    p = lat if axis == "lat" else lon
    m_lat = np.nanmedian(lat)
    deg_len = 111320.0 if axis == "lat" else 111320.0 * np.cos(np.radians(m_lat))
    rng = p.max() - p.min()
    if rng <= 0:
        return None
    nbins = max(5, int(rng * deg_len / bin_m))
    edges_bins = np.linspace(p.min(), p.max(), nbins + 1)
    idx = np.digitize(p, edges_bins) - 1
    s_l, h_l = [], []
    for b in range(nbins):
        mm = idx == b
        if mm.sum() == 0 or not np.isfinite(ele[mm]).any():
            continue
        s_l.append(np.mean(p[mm]) * deg_len)
        h_l.append(np.median(ele[mm]))
    if len(s_l) < 5:
        return None
    s = np.array(s_l); s = s - s[0]
    h = np.array(h_l)
    prof = pd.DataFrame({"s_m": s, "elev_m": h})
    prof["grade_raw_pct"] = prof["elev_m"].diff() / prof["s_m"].diff() * 100
    prof["grade_pct"] = (prof["grade_raw_pct"]
                         .rolling(smooth_bins, center=True, min_periods=1).mean())
    return prof


# corridors whose OSM naming is fragmented -> robust binned profile
# disp -> (axis, box=(lat_min, lat_max, lon_min, lon_max))
BINNED_CORRIDORS = {
    "ولی‌عصر (خیابان)": ("lat", (35.645, 35.815, 51.36, 51.47)),   # میدان راه‌آهن تا تجریش
    "رسالت–شهید سلیمانی (بزرگراه)": ("lon", (35.725, 35.755, 51.40, 51.63)),
    "کاشانی (بزرگراه)": ("lon", (35.725, 35.785, 51.27, 51.46)),  # ستاری تا مدرس
}


def main():
    # ---------------- load network ----------------
    nodes = pd.read_csv(os.path.join(OUT, "tehran_network_nodes.csv"))
    edges = pd.read_csv(os.path.join(OUT, "tehran_network_edges.csv"),
                        dtype={"name": str}, keep_default_na=False, na_values=[""])
    print(f"nodes={len(nodes):,}  edges={len(edges):,}")

    # ---------------- sample DEM at nodes ----------------
    print("Sampling DEM at network nodes ...")
    lons = nodes["lon"].to_numpy()
    lats = nodes["lat"].to_numpy()
    elev = sample_elevation(lons, lats)
    nodes["elev_m"] = elev
    ok = np.isfinite(elev)
    print(f"  elevation sampled for {ok.sum():,}/{len(nodes):,} nodes")

    # ---------------- DEM overall statistics ----------------
    dem_stats = {
        "source": "Copernicus DEM GLO-30 (DSM, 30 m) — AWS open bucket",
        "tiles": ["N35_00_E051_00", "N35_00_E050_00"],
        "node_elev_m": {
            "min": round(float(np.nanmin(elev)), 1),
            "p5": round(float(np.nanpercentile(elev, 5)), 1),
            "median": round(float(np.nanmedian(elev)), 1),
            "p95": round(float(np.nanpercentile(elev, 95)), 1),
            "max": round(float(np.nanmax(elev)), 1),
        },
    }

    # ---------------- edge grades ----------------
    emap = dict(zip(nodes["node_id"], nodes["elev_m"]))
    edges["elev_u"] = edges["u"].map(emap)
    edges["elev_v"] = edges["v"].map(emap)
    edges["grade_pct"] = (edges["elev_v"] - edges["elev_u"]) / edges["length_m"] * 100.0
    n_grad = int(edges["grade_pct"].notna().sum())
    print(f"  grades computed for {n_grad:,}/{len(edges):,} edges")

    # robust network-level statistics (clip artefact-suspect extremes for stats only)
    g = edges["grade_pct"].dropna()
    g_ok = g[g.abs() <= ARTEFACT_GRADE]
    w = edges.loc[g_ok.index, "length_m"]
    dem_stats["network_grade_pct"] = {
        "edges_with_grade": n_grad,
        "artefact_suspect_edges": int((g.abs() > ARTEFACT_GRADE).sum()),
        "length_weighted_mean_abs_grade": round(
            float((g_ok.abs() * w).sum() / w.sum()), 2),
        "p50_abs_grade": round(float(np.percentile(g_ok.abs(), 50)), 2),
        "p90_abs_grade": round(float(np.percentile(g_ok.abs(), 90)), 2),
        "p99_abs_grade": round(float(np.percentile(g_ok.abs(), 99)), 2),
        "max_abs_grade_kept": round(float(g_ok.abs().max()), 2),
        "share_len_grade_gt_4pct": round(
            float(w[g_ok.abs() > 4].sum() / w.sum() * 100), 2),
        "share_len_grade_gt_6pct": round(
            float(w[g_ok.abs() > 6].sum() / w.sum() * 100), 2),
    }

    # by highway class
    by_cls = {}
    sub = edges.dropna(subset=["grade_pct"])
    sub = sub[sub["grade_pct"].abs() <= ARTEFACT_GRADE]
    for cls, grp in sub.groupby("highway"):
        by_cls[cls] = {
            "length_km": round(float(grp["length_m"].sum() / 1000), 1),
            "mean_abs_grade_pct": round(
                float((grp["grade_pct"].abs() * grp["length_m"]).sum()
                      / grp["length_m"].sum()), 2),
            "p90_abs_grade_pct": round(float(np.percentile(grp["grade_pct"].abs(), 90)), 2),
        }
    dem_stats["grade_by_highway_class"] = dict(
        sorted(by_cls.items(), key=lambda kv: -kv[1]["length_km"]))

    edges.to_csv(os.path.join(OUT, "edges_with_grade.csv"), index=False,
                 encoding="utf-8-sig")
    print("Wrote edges_with_grade.csv")

    # ---------------- corridors ----------------
    print("\nExtracting corridors ...")
    node_pos = {int(r.node_id): (float(r.lon), float(r.lat))
                for r in nodes[["node_id", "lon", "lat"]].itertuples()}
    node_elev = dict(zip(nodes["node_id"], nodes["elev_m"]))

    edges["name_norm"] = edges["name"].map(norm_name)
    corr_stats = {}
    shomali_flag = edges["name"].astype("string").str.contains("شمالی").fillna(False).astype(bool)
    for disp, aliases in CORRIDORS.items():
        is_shomali = "شمالی" in disp
        name_match = edges["name_norm"].apply(
            lambda s: any(a in s for a in aliases)) & (shomali_flag == is_shomali)
        if disp in BINNED_CORRIDORS:
            # geographic box does the filtering; keep بلوار variants (e.g.
            # «بلوار رسالت» is the western section of the Resalat corridor)
            base = name_match
        else:
            base = name_match \
                & ~edges["name"].str.startswith(("میدان", "کنارگذر", "بلوار"), na=False)
        ce = edges[base]
        ce = ce.dropna(subset=["grade_pct"])
        if len(ce) < 5:
            corr_stats[disp] = {"status": "NOT FOUND / too few segments",
                                "segments": int(len(ce))}
            print(f"  {disp:32s} — not found")
            continue

        G = nx.Graph()
        for r in ce.itertuples():
            G.add_edge(int(r.u), int(r.v), length=float(r.length_m),
                       grade=float(r.grade_pct))

        # fragmented corridors (ولی‌عصر: 220+ OSM components, etc.)
        # -> robust binned profile instead of path chaining
        if disp in BINNED_CORRIDORS:
            axis, box = BINNED_CORRIDORS[disp]
            prof = binned_profile(ce, node_pos, node_elev, axis=axis,
                                  bin_m=150.0, box=box)
            method = f"{axis}_binned"
            if prof is None:
                corr_stats[disp] = {"status": "binned profile failed"}
                continue
            s_acc = float(prof["s_m"].iloc[-1])
            south_first = True
        else:
            path = longest_spine(G)
            if path is None or len(path) < 5:
                corr_stats[disp] = {"status": "chain failed", "segments": int(len(ce))}
                continue
            method = "spine_chain"

            # walk path -> s, elevation, smoothed grade
            s_vals, h_vals = [], []
            s_acc = 0.0
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                seg_len = G[a][b]["length"]
                if i == 0:
                    s_vals.append(0.0)
                    h_vals.append(node_elev.get(a, np.nan))
                s_acc += seg_len
                s_vals.append(s_acc)
                h_vals.append(node_elev.get(b, np.nan))

            prof = pd.DataFrame({"s_m": s_vals, "elev_m": h_vals})
            prof["grade_raw_pct"] = prof["elev_m"].diff() / prof["s_m"].diff() * 100
            # centered smoothing over ~SMOOTH_WINDOW_M
            prof["grade_pct"] = prof["grade_raw_pct"].rolling(
                window=max(3, int(SMOOTH_WINDOW_M / max(np.mean(np.diff(s_vals)), 1))),
                center=True, min_periods=1).mean()
            south_first = node_pos[path[0]][1] <= node_pos[path[-1]][1]

        slug = re.sub(r"[^\w\u0600-\u06FF]+", "_", disp).strip("_")
        prof.to_csv(os.path.join(PROF, f"{slug}_profile.csv"), index=False)

        gsm = prof["grade_pct"].dropna()
        corr_stats[disp] = {
            "status": "OK",
            "profile_method": method,
            "segments": int(len(ce)),
            "chain_length_km": round(s_acc / 1000, 2),
            "chain_direction": "S->N" if south_first else "N->S",
            "elev_start_m": round(float(prof["elev_m"].iloc[0]), 1),
            "elev_end_m": round(float(prof["elev_m"].iloc[-1]), 1),
            "elev_min_m": round(float(prof["elev_m"].min()), 1),
            "elev_max_m": round(float(prof["elev_m"].max()), 1),
            "mean_abs_grade_pct": round(float(gsm.abs().mean()), 2),
            "max_grade_pct": round(float(gsm.max()), 2),
            "min_grade_pct": round(float(gsm.min()), 2),
            "p90_abs_grade_pct": round(float(np.percentile(gsm.abs(), 90)), 2),
        }
        print(f"  {disp:32s} — {corr_stats[disp]['chain_length_km']:6.2f} km, "
              f"ΔH={corr_stats[disp]['elev_end_m'] - corr_stats[disp]['elev_start_m']:+7.1f} m, "
              f"max|θ|={max(abs(corr_stats[disp]['max_grade_pct']), abs(corr_stats[disp]['min_grade_pct'])):5.2f}%")

    with open(os.path.join(OUT, "corridor_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(corr_stats, fh, ensure_ascii=False, indent=2)

    dem_stats["random_seed"] = SEED
    dem_stats["smoothing_window_m"] = SMOOTH_WINDOW_M
    dem_stats["artefact_threshold_pct"] = ARTEFACT_GRADE
    with open(os.path.join(OUT, "dem_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dem_stats, fh, ensure_ascii=False, indent=2)

    print("\n===== DEM / GRADE SUMMARY =====")
    print(json.dumps(dem_stats, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
