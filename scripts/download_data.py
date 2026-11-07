"""Download the public raw datasets that are too large for the repository and verify them.

  python scripts/download_data.py

Downloads into data/raw/ and checks SHA-256 against the values used for the published results:
  * Porto taxi trajectories, ECML/PKDD 2015 (UCI Machine Learning Repository, dataset 339)
  * Copernicus DEM GLO-30 tiles N35E050 and N35E051 (official AWS open-data bucket)
The OpenStreetMap extracts (Overpass API, OSM base 2026-09-11) are included in the repository as
data/raw/osm_tehran_2026-09-11.zip (a fresh query would return newer map data); this script checks
and unpacks them. The OSM charging-station query is included as data/raw/overpass_charging_tehran.json.
"""
import hashlib, os, sys, urllib.request, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
FILES = [
    ("porto_taxi.zip",
     "https://archive.ics.uci.edu/static/public/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015.zip",
     "a33e2a5e145607ae2bad0db5d21b7548c88b7e0f9db1ce15839f24c4c61f8c76"),
    ("dem_N35_E050.tif",
     "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N35_00_E050_00_DEM/Copernicus_DSM_COG_10_N35_00_E050_00_DEM.tif",
     "c29d43f1af37b9341ee61c53ff5a0bf8d47fc74e17c97e47eec5511e1102ba25"),
    ("dem_N35_E051.tif",
     "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N35_00_E051_00_DEM/Copernicus_DSM_COG_10_N35_00_E051_00_DEM.tif",
     "7184e4da86a163201ea034ec2c4cb5ba85293d9df0bf2f40b9ee5fcb8303b3dd"),
]
OSM_ZIP = ("osm_tehran_2026-09-11.zip", "ef29d0404c7205586e46d40b85a42ccd5a8a6eb012e65e7f61b451067c88b6d3")
TRAIN_SHA = "2891d03ea99ce0fc203daedd434848a1a8baa7a8fb6da10f8d7ff26559efff6d"

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def main():
    os.makedirs(RAW, exist_ok=True)
    ok = True
    for name, url, expected in FILES:
        path = os.path.join(RAW, name)
        if not os.path.exists(path):
            print(f"downloading {name} ...", flush=True)
            urllib.request.urlretrieve(url, path)
        got = sha256(path)
        print(f"{name}: {'OK' if got == expected else 'HASH MISMATCH ' + got}")
        ok &= got == expected
    train = os.path.join(RAW, "train.csv")
    if not os.path.exists(train):
        print("extracting train.csv ...", flush=True)
        with zipfile.ZipFile(os.path.join(RAW, "porto_taxi.zip")) as z:
            z.extract("train.csv.zip", RAW)
        inner = os.path.join(RAW, "train.csv.zip")
        with zipfile.ZipFile(inner) as z:
            z.extract("train.csv", RAW)
        os.remove(inner)
    got = sha256(train)
    print(f"train.csv: {'OK' if got == TRAIN_SHA else 'HASH MISMATCH ' + got}")
    ok &= got == TRAIN_SHA
    name, expected = OSM_ZIP
    got = sha256(os.path.join(RAW, name))
    print(f"{name}: {'OK' if got == expected else 'HASH MISMATCH ' + got}")
    ok &= got == expected
    with zipfile.ZipFile(os.path.join(RAW, name)) as z:
        for member in z.namelist():
            if not os.path.exists(os.path.join(RAW, member)):
                z.extract(member, RAW)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
