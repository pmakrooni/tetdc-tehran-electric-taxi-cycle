"""List the raw 1 Hz trip files of the Hull (2024) dataset: files that are UTF-8 CSV with the 12 raw
columns (Date, Time, Speed, Latitude, Longitude, ...). Processed files have extra columns and are skipped;
one file named .csv is actually an Excel workbook and is skipped.
Usage: python hull_split.py <hull_dir> <out_dir>"""
import sys, os, json, glob
import pandas as pd
HULL, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
raw, proc, bad = [], [], []
for fn in sorted(glob.glob(os.path.join(HULL, "raw", "*.csv"))):
    rel = os.path.relpath(fn, HULL)
    with open(fn, "rb") as f:
        if f.read(2) == b"PK": bad.append(rel); continue            # xlsx disguised as csv
    cols = pd.read_csv(fn, nrows=0, encoding="utf-8-sig").columns
    (raw if len(cols) == 12 else proc).append(rel)
json.dump({"raw": raw, "processed": proc, "not_csv": bad}, open(os.path.join(OUT, "hull_split.json"), "w"), indent=1)
print(f"raw {len(raw)}  processed {len(proc)}  not_csv {len(bad)}")
