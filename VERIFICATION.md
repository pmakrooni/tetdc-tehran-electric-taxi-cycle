# Verification record

On 2026-10-03 the complete pipeline was re-executed from public raw data in a clean environment
(Python 3.12.3, package versions in `requirements.txt`) and every output was compared with the
original run. The raw data were checked first:

| Raw data | Check |
|---|---|
| UCI Porto archive and `train.csv` | SHA-256 identical to the values in `scripts/download_data.py` |
| Copernicus DEM tiles | SHA-256 identical to the official files in the AWS open-data bucket |
| OpenStreetMap extracts | Overpass header `osm_base = 2026-09-11T20:59-21:01Z` |

Results of the re-execution:

| Stage | Script | Output | Result |
|---|---|---|---|
| Network | preprocessing/build_network.py | network_summary.json (21,049.1 km; 497,863 directed edges) | byte-identical |
| Topography | preprocessing/extract_topography.py | edges_with_grade.csv; dem_summary.json; corridor_stats.json; 16 corridor profiles | byte-identical |
| Porto cleaning | preprocessing/porto_preprocess.py | trip_pattern_stats.json (1,710,670 trips -> 1,594,900 kept; 440 taxis); 48,415-trip trace sample | byte-identical / content-identical |
| Charging layer | preprocessing/build_charging_layer.py | charging_stations.csv; charging_layer_stats.json | byte-identical |
| Trip chains | analysis/make_chain_params.py | tehran_chain_params.json | byte-identical |
| Micro-trips | drive_cycle/extract_microtrips.py | microtrip_bank_stats.json; microtrips_features.parquet (200,365 micro-trips) | identical (max difference 1.4e-17) |
| Clustering | drive_cycle/pca_cluster.py | microtrip_pca_kmeans.json | byte-identical (cluster table differs only at 1e-12) |
| Cycle | drive_cycle/build_markov.py | TETDC_1.csv; drive_cycle_stats.json | byte-identical |
| Energy | digital_twin/simulate_cycle.py | cycle_energy_results.json | byte-identical |
| Baselines | experiments/run_baselines.py | baseline_comparison.json (official WLTC/NEDC from `wltp`) | byte-identical |
| Fleet | experiments/run_fleet_sim.py | fleet_sim_results.json | byte-identical |
| Growth | experiments/run_growth_scenarios.py | growth_scenarios.json | byte-identical |
| Sensitivity | analysis/sensitivity_analysis.py | sensitivity_results.json | byte-identical |
| Statistics | analysis/statistical_analysis.py | statistical_analysis.json; route_energy_dataset.csv | byte-identical |
| Validation | analysis/deep_validation.py | validation_results.json | identical except two p-values at the 13th significant digit |
| Uncertainty (new) | uncertainty/uq.py, uncertainty/analyze.py | uq_results.json | new analysis (15 inputs incl. battery U(62,80) kWh); vectorised model matches digital_twin/ev_model.py to 1.6e-16 |

`verify_outputs.py` repeats this comparison automatically against `results_reference/`.

## Re-run of 7 November 2026 (version 1.1.0)

The working environment was rebuilt from the archived repository; the raw data were downloaded again (all SHA-256 checksums OK) and `run_all.py` was executed in a fresh Python environment (numpy 2.4.4, pandas 3.0.2, pyarrow 25.0.1, scipy 1.17.1, scikit-learn 1.8.0, rasterio 1.5.2, networkx 3.6.1, SALib 1.6.0, wltp 0.1.2a0, matplotlib 3.10). `verify_outputs.py` reported 0 differences for every file in `results_reference/`. The new output `uq_ensemble.json` was then added to `results_reference/`, and `dynamics_energy.json` to `results_reference/supplement/`; `uq_ensemble.py` asserts that the rebuilt seed-42 cycle is identical to the published TETDC-1.

## Re-run of 9 October 2026 (version 1.0.0)

The raw data were downloaded again (`scripts/download_data.py`; all SHA-256 checksums OK) and `run_all.py` was executed in a fresh Python environment (numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0, rasterio 1.5.2, networkx 3.6.1, SALib 1.6.0, wltp 0.1.2a0). `verify_outputs.py` reported 0 differences for every file in `results_reference/`, including the bias-corrected uncertainty analysis. The new robustness checks (`terrain_scenarios.json`, `cycle_checks.json`, `uq_bias_sensitivity.json`) were then added to `results_reference/`; a second `verify_outputs.py` run compared them with a fresh execution.
