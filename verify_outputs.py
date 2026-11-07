"""Compare regenerated outputs in data/processed/ with the verified copies in results_reference/.

Numbers are compared with a relative tolerance of 1e-9, so harmless floating-point differences
between computers do not count as failures. Exit code 0 means every reference output was reproduced.
"""
import csv, json, math, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REF, NEW = os.path.join(ROOT, "results_reference"), os.path.join(ROOT, "data", "processed")
TOL = 1e-9

def same(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    for conv in (float,):
        try:
            x, y = conv(a), conv(b)
            if isinstance(a, bool) or isinstance(b, bool):
                break
            return math.isclose(x, y, rel_tol=TOL, abs_tol=1e-12) or (math.isnan(x) and math.isnan(y))
        except (TypeError, ValueError):
            pass
    return a == b

def load(path):
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))

def main():
    failures = 0
    for name in sorted(os.listdir(REF)):
        if os.path.isdir(os.path.join(REF, name)):
            continue                      # results_reference/supplement is compared at the end of supplement/run_supplement.py
        new = os.path.join(NEW, name)
        if not os.path.exists(new):
            print(f"MISSING   {name}"); failures += 1; continue
        ok = same(load(os.path.join(REF, name)), load(new))
        print(f"{'OK       ' if ok else 'DIFFERENT'} {name}")
        failures += not ok
    print(f"\n{failures} problem(s)")
    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()
