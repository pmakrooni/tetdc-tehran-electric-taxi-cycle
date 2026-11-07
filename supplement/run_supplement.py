"""Run all supplementary analyses (interpolation bias and Tehran-cycle comparison).
Usage: python supplement/run_supplement.py [data_dir] [out_dir]
Defaults: data_dir = data/supplement (filled by download_supplement_data.py), out_dir = results/supplement
After the run, every file in results_reference/supplement is compared with the fresh output."""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
DATA = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(REPO, "data", "supplement")
OUT = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(REPO, "results", "supplement")
os.makedirs(OUT, exist_ok=True)
py = [sys.executable, "-I"]
steps = [["hull_split.py", os.path.join(DATA, "hull"), OUT],
         ["interp_bias.py", REPO, os.path.join(DATA, "hull"), OUT],
         ["interp_control.py", REPO, os.path.join(DATA, "hull"), OUT],
         ["tehran19_compare.py", os.path.join(DATA, "epa"), os.path.join(DATA, "tehran19", "OutputData.xlsx"),
          os.path.join(REPO, "results_reference", "TETDC_1.csv"), OUT],
         ["bias_factor.py", OUT],
         ["dynamics_energy.py", OUT]]
for s in steps:
    print("==", s[0]); subprocess.run(py + [os.path.join(HERE, s[0])] + s[1:], check=True, stdout=subprocess.DEVNULL)
print("outputs in", OUT)

# compare every reference output with the fresh run (same tolerance as verify_outputs.py)
sys.path.insert(0, REPO)
from verify_outputs import same, load
REF = os.path.join(REPO, "results_reference", "supplement"); failures = 0
for name in sorted(os.listdir(REF)):
    new = os.path.join(DATA, "manifest.json") if name == "data_manifest.json" else os.path.join(OUT, name)
    if not os.path.exists(new):
        print(f"MISSING   {name}"); failures += 1; continue
    ref, got = load(os.path.join(REF, name)), load(new)
    if name == "data_manifest.json":                       # file paths are stored relative to the download root
        got = [dict(m, file=m["file"].split("supplement/", 1)[-1]) for m in got]
    ok = same(ref, got)
    print(f"{'OK       ' if ok else 'DIFFERENT'} {name}"); failures += not ok
print(f"\n{failures} problem(s)")
sys.exit(1 if failures else 0)
