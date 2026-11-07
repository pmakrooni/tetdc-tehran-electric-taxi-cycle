#!/usr/bin/env python3.13
# -*- coding: utf-8 -*-
"""
TETDC Stage 1 — Build Tehran road network from OSM XML extracts
================================================================
Input : data/raw/tehran_major_roads.osm, tehran_minor_S1/S2/S3.osm
        (Overpass mirror maps.mail.ru, OSM base 2026-09-11, bbox 35.45-35.90N / 51.00-51.75E)
Output: data/processed/tehran_network_nodes.csv
        data/processed/tehran_network_edges.csv
        data/processed/network_summary.json

Deterministic (no randomness). RANDOM_SEED=42 set for project consistency.
Edge length: haversine on WGS84 (projection error < 0.3% at Tehran latitude).
"""
import xml.etree.ElementTree as ET
from math import radians, sin, cos, asin, sqrt
import json
import csv
import os
import sys

RANDOM_SEED = 42  # project-wide fixed seed (master prompt §28)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO_ROOT, "data", "raw")
OUT = os.path.join(REPO_ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)

FILES = [
    "tehran_major_roads.osm",
    "tehran_minor_S1.osm",
    "tehran_minor_S2.osm",
    "tehran_minor_S3.osm",
]

# highway classes that are drivable for taxis (exclude footway/cycleway/etc.
# which were never queried anyway; this is a safety filter)
DRIVABLE = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "residential", "unclassified", "living_street",
}

R_EARTH = 6371008.8  # mean Earth radius [m], IUGG


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters."""
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlmb = radians(lon2 - lon1)
    a = sin(dphi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(dlmb / 2.0) ** 2
    return 2.0 * R_EARTH * asin(sqrt(a))


def parse_maxspeed(v):
    """Normalize OSM maxspeed tag to km/h (float or None)."""
    if not v:
        return None
    v = str(v).strip().lower()
    if "mph" in v:
        digits = "".join(ch for ch in v if ch.isdigit() or ch == ".")
        try:
            return round(float(digits) * 1.609344, 1)
        except ValueError:
            return None
    digits = "".join(ch for ch in v if ch.isdigit())
    if digits:
        try:
            return float(digits)
        except ValueError:
            return None
    return None  # e.g. "RU:urban", "walk"


def parse_osm_file(path, nodes, ways, stats):
    """Stream-parse one .osm XML file; collect nodes + highway ways."""
    n_nodes = n_ways = 0
    for event, elem in ET.iterparse(path, events=("end",)):
        tag = elem.tag
        if tag == "node":
            nid = int(elem.get("id"))
            nodes[nid] = (float(elem.get("lat")), float(elem.get("lon")))
            n_nodes += 1
            elem.clear()
        elif tag == "way":
            hw = None
            tags = {}
            for t in elem.findall("tag"):
                k, v = t.get("k"), t.get("v")
                tags[k] = v
                if k == "highway":
                    hw = v
            if hw in DRIVABLE:
                nds = [int(nd.get("ref")) for nd in elem.findall("nd")]
                ways.append((int(elem.get("id")), hw, tags, nds))
                n_ways += 1
            elem.clear()
    stats["parsed"].append(os.path.basename(path))
    print(f"  {os.path.basename(path)}: {n_nodes:,} nodes, {n_ways:,} highway ways")


def main():
    nodes = {}      # id -> (lat, lon)
    ways = []       # (way_id, highway, tags, [node refs])
    stats = {"parsed": []}

    print("Parsing OSM files ...")
    for f in FILES:
        parse_osm_file(os.path.join(RAW, f), nodes, ways, stats)

    print(f"\nTotal: {len(nodes):,} nodes, {len(ways):,} drivable ways")

    # ---------- build edges ----------
    edges = []
    missing_node = 0
    for wid, hw, tags, nds in ways:
        name = tags.get("name", "")
        maxspeed = parse_maxspeed(tags.get("maxspeed"))
        lanes_raw = tags.get("lanes")
        try:
            lanes = int(str(lanes_raw).split(";")[0]) if lanes_raw else None
        except ValueError:
            lanes = None
        oneway_raw = tags.get("oneway", "")
        implied_oneway = 1 if hw in ("motorway", "motorway_link") else 0
        oneway = 1 if (oneway_raw in ("yes", "1", "true") or implied_oneway) else 0
        bridge = 1 if tags.get("bridge") not in (None, "no") else 0
        tunnel = 1 if tags.get("tunnel") not in (None, "no") else 0

        pts = []
        for nid in nds:
            if nid in nodes:
                pts.append((nid,) + nodes[nid])
            else:
                missing_node += 1
        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            (u, ulat, ulon) = pts[i]
            (v, vlat, vlon) = pts[i + 1]
            seg_len = haversine(ulat, ulon, vlat, vlon)
            if seg_len <= 0:
                continue
            edges.append({
                "way_id": wid, "u": u, "v": v,
                "lat_u": round(ulat, 7), "lon_u": round(ulon, 7),
                "lat_v": round(vlat, 7), "lon_v": round(vlon, 7),
                "length_m": round(seg_len, 2),
                "highway": hw,
                "maxspeed_kmh": maxspeed,
                "lanes": lanes,
                "oneway": oneway,
                "bridge": bridge,
                "tunnel": tunnel,
                "name": name,
            })

    print(f"Edges built: {len(edges):,} (missing node refs: {missing_node:,})")

    # ---------- write edges CSV ----------
    edge_path = os.path.join(OUT, "tehran_network_edges.csv")
    fields = list(edges[0].keys())
    with open(edge_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(edges)
    print(f"Wrote {edge_path}")

    # ---------- write nodes CSV (only nodes referenced by drivable ways) ----------
    used = set()
    for e in edges:
        used.add(e["u"]); used.add(e["v"])
    node_path = os.path.join(OUT, "tehran_network_nodes.csv")
    with open(node_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["node_id", "lat", "lon"])
        for nid in sorted(used):
            lat, lon = nodes[nid]
            w.writerow([nid, round(lat, 7), round(lon, 7)])
    print(f"Wrote {node_path} ({len(used):,} nodes)")

    # ---------- summary statistics ----------
    total_len_km = sum(e["length_m"] for e in edges) / 1000.0
    by_class = {}
    for e in edges:
        c = by_class.setdefault(e["highway"], {"edges": 0, "length_km": 0.0})
        c["edges"] += 1
        c["length_km"] += e["length_m"] / 1000.0
    for c in by_class.values():
        c["length_km"] = round(c["length_km"], 1)

    oneway_len = sum(e["length_m"] for e in edges if e["oneway"] == 1) / 1000.0
    named_len = sum(e["length_m"] for e in edges if e["name"]) / 1000.0
    speed_vals = [e["maxspeed_kmh"] for e in edges if e["maxspeed_kmh"]]
    bridge_len = sum(e["length_m"] for e in edges if e["bridge"] == 1) / 1000.0
    tunnel_len = sum(e["length_m"] for e in edges if e["tunnel"] == 1) / 1000.0

    summary = {
        "source": "OpenStreetMap via Overpass mirror maps.mail.ru (OSM base 2026-09-11)",
        "bbox": {"south": 35.45, "west": 51.00, "north": 35.90, "east": 51.75},
        "files_parsed": stats["parsed"],
        "total_ways_drivable": len(ways),
        "total_nodes_used": len(used),
        "total_edges": len(edges),
        "total_length_km": round(total_len_km, 1),
        "by_highway_class": dict(sorted(by_class.items(), key=lambda kv: -kv[1]["length_km"])),
        "oneway_length_km": round(oneway_len, 1),
        "oneway_fraction_of_length": round(oneway_len / total_len_km, 3),
        "named_length_km": round(named_len, 1),
        "named_fraction_of_length": round(named_len / total_len_km, 3),
        "edges_with_maxspeed": len(speed_vals),
        "maxspeed_kmh": {
            "min": min(speed_vals), "max": max(speed_vals),
            "mean": round(sum(speed_vals) / len(speed_vals), 1),
        } if speed_vals else None,
        "bridge_length_km": round(bridge_len, 1),
        "tunnel_length_km": round(tunnel_len, 1),
        "random_seed": RANDOM_SEED,
        "edge_length_method": "haversine (WGS84, R=6371008.8 m)",
    }
    with open(os.path.join(OUT, "network_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\n===== SUMMARY =====")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_highway_class"},
                     ensure_ascii=False, indent=2))
    print("Top classes by length:")
    for hw, c in list(summary["by_highway_class"].items())[:10]:
        print(f"  {hw:16s} {c['length_km']:9.1f} km  ({c['edges']:,} edges)")


if __name__ == "__main__":
    main()
