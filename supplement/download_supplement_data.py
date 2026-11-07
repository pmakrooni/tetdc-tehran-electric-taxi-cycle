"""Download the open data used by the supplementary analyses and verify every file.
  1. Hull, C. (2024) 1Hz GPS Tracking Data, Mendeley Data V3, doi:10.17632/xt69cnwh56.3 (CC BY 4.0)
     -> data/supplement/hull/raw/NNN_<name>.csv ; SHA-256 checked against the Mendeley API.
  2. Azadi et al. (2024) Mendeley Data V1, doi:10.17632/f7c75syg2f.1 (CC BY 4.0)
     -> data/supplement/tehran19/OutputData.xlsx ; SHA-256 checked against the Mendeley API.
  3. US EPA FTP-75 and UDDS speed traces (public domain)
     -> data/supplement/epa/ftpcol.txt, uddscol.txt ; SHA-256 recorded in the manifest.
Usage: python download_supplement_data.py [target_dir]   (default: data/supplement)"""
import sys, os, json, hashlib, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "supplement")
UA = {"User-Agent": "TETDC-supplement/1.0"}
def sha(fn): return hashlib.sha256(open(fn, "rb").read()).hexdigest()
def fetch(url, fn, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(fn, "wb") as f:
                f.write(r.read()); return
        except Exception:
            time.sleep(2 ** k)
    raise RuntimeError("download failed: " + url)
def mendeley_files(ds, version):
    """File list of a Mendeley dataset (latest version); stops if the version differs from the cited one."""
    d = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"https://data.mendeley.com/public-api/datasets/{ds}", headers=UA), timeout=60).read())
    if d.get("version") != version:
        sys.exit(f"{ds}: latest version is {d.get('version')}, the paper cites version {version}; check before use")
    return d["files"]
def get_verified(f, fn):
    want = f["content_details"].get("sha256_hash", "")
    if not (os.path.exists(fn) and want and sha(fn) == want):
        fetch(f["content_details"]["download_url"], fn)
    got = sha(fn)
    return {"file": fn, "bytes": os.path.getsize(fn), "sha256": got, "sha256_expected": want, "ok": (got == want) if want else None}
manifest = []
# 1. Hull 1 Hz GPS data (version 3)
hd = os.path.join(ROOT, "hull", "raw"); os.makedirs(hd, exist_ok=True)
hfiles = mendeley_files("xt69cnwh56", 3)
with ThreadPoolExecutor(8) as ex:
    manifest += list(ex.map(lambda i_f: get_verified(i_f[1], os.path.join(hd, f"{i_f[0]:03d}_{i_f[1]['filename']}")), enumerate(hfiles)))
# 2. Tehran cycles output data
td = os.path.join(ROOT, "tehran19"); os.makedirs(td, exist_ok=True)
for f in mendeley_files("f7c75syg2f", 1):
    if f["filename"] == "OutputData.xlsx":
        manifest.append(get_verified(f, os.path.join(td, "OutputData.xlsx")))
# 3. EPA traces
ed = os.path.join(ROOT, "epa"); os.makedirs(ed, exist_ok=True)
for name in ("ftpcol.txt", "uddscol.txt"):
    fn = os.path.join(ed, name)
    if not os.path.exists(fn): fetch("https://www.epa.gov/sites/default/files/2015-10/" + name, fn)
    manifest.append({"file": fn, "bytes": os.path.getsize(fn), "sha256": sha(fn), "sha256_expected": "", "ok": None})
json.dump(manifest, open(os.path.join(ROOT, "manifest.json"), "w"), indent=1)
bad = [m for m in manifest if m["ok"] is False]
print(f"files {len(manifest)} | verified {sum(m['ok'] is True for m in manifest)} | mismatches {len(bad)}")
sys.exit(1 if bad else 0)
