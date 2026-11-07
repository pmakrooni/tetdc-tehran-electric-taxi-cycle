"""Run the complete TETDC pipeline in order, from public raw data to all published results.

  python scripts/download_data.py   # once: fetch and verify raw data (~0.6 GB download)
  python run_all.py                 # run every stage (about 10 minutes on one CPU core)
  python verify_outputs.py          # compare the regenerated outputs with results_reference/

Run a subset with:  python run_all.py energy baselines
"""
import os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
STAGES = [
    ("network",      "preprocessing/build_network.py"),         # OSM -> Tehran drivable network
    ("topography",   "preprocessing/extract_topography.py"),    # Copernicus GLO-30 -> edge grades, corridor profiles
    ("porto",        "preprocessing/porto_preprocess.py"),      # Porto GPS cleaning + 3% trace sample
    ("charging",     "preprocessing/build_charging_layer.py"),  # OSM charging stations in Tehran
    ("chain",        "analysis/make_chain_params.py"),          # Tehran trip-chain parameters
    ("microtrips",   "drive_cycle/extract_microtrips.py"),      # micro-trip bank
    ("clustering",   "drive_cycle/pca_cluster.py"),             # PCA + K-means
    ("cycle",        "drive_cycle/build_markov.py"),            # TETDC-1 construction
    ("energy",       "digital_twin/simulate_cycle.py"),         # energy on TETDC-1, grade scenarios
    ("ev_validation","digital_twin/validate_ev_model.py"),      # vehicle-model checks
    ("baselines",    "experiments/run_baselines.py"),           # TETDC vs official WLTC / NEDC
    ("fleet",        "experiments/run_fleet_sim.py"),           # 24 h fleet simulation
    ("growth",       "experiments/run_growth_scenarios.py"),    # fleet-growth scenarios
    ("sensitivity",  "analysis/sensitivity_analysis.py"),
    ("statistics",   "analysis/statistical_analysis.py"),
    ("validation",   "analysis/deep_validation.py"),
    ("uq_sample",    "uncertainty/uq.py"),                       # Monte-Carlo + Sobol' (new analysis)
    ("uq_analyze",   "uncertainty/analyze.py"),
    ("terrain",      "analysis/terrain_scenarios.py"),          # central grade, bounds, 13 corridor profiles
    ("cycle_checks", "analysis/cycle_checks.py"),               # bank size, implausible accelerations, energy, seeds
    ("bias_sens",    "uncertainty/bias_sensitivity.py"),        # range gap under four bias assumptions
    ("uq_ensemble",  "uncertainty/uq_ensemble.py"),             # 11 construction runs x 10,000 paired draws
    ("figures",      "figures/make_figures.py"),                 # article figures -> figures/output/
    ("uq_figures",   "figures/make_uq_figures.py"),              # Fig. 6 (with bias correction) and Fig. 8
]

def main():
    wanted = sys.argv[1:] or [s for s, _ in STAGES]
    os.makedirs(os.path.join(ROOT, "data", "processed"), exist_ok=True)
    for name, script in STAGES:
        if name not in wanted:
            continue
        t0 = time.time()
        print(f"=== {name}: {script}", flush=True)
        r = subprocess.run([sys.executable, os.path.join(ROOT, script)], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"stage '{name}' failed (exit {r.returncode})")
        print(f"=== {name} done in {time.time() - t0:.0f} s", flush=True)

if __name__ == "__main__":
    main()
