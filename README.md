# TETDC — Tehran Electric Taxi Driving Cycle

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24039688.svg)](https://doi.org/10.5281/zenodo.24039688)

Archived at Zenodo: all versions [10.5281/zenodo.24039688](https://doi.org/10.5281/zenodo.24039688); version 1.2.0, used for the article, [10.5281/zenodo.24039689](https://doi.org/10.5281/zenodo.24039689).

Code, data and verified results for the article
**"Decomposing the Electric-Taxi Range Gap in Tehran with an Open-Data Digital Twin"** (Pouria Makrooni et al., submitted to *Transportation Research Part D: Transport and Environment*).

TETDC (Tehran Electric Taxi Driving Cycle) is the name of the code and of the driving cycle TETDC-1; the article title is given above.

Every number in the article can be regenerated from public raw data with the scripts in this repository.
`VERIFICATION.md` documents an independent end-to-end re-execution; `verify_outputs.py` repeats the check on your computer.

## What the pipeline does

1. **Road network and topography.** Tehran's drivable network from OpenStreetMap (21,049 km, 497,863 directed edges) is combined with the Copernicus GLO-30 elevation model to obtain road grades and elevation profiles of 16 major corridors.
2. **Driving cycle.** Taxi trips from the public Porto ECML/PKDD 2015 dataset (1,710,670 trips; 1,594,900 kept after cleaning) are split into 200,365 micro-trips, clustered (PCA + K-means), and recombined into the 1,920 s TETDC-1 cycle.
3. **Vehicle model.** A longitudinal electric-vehicle model of the GAC Aion V Plus class computes energy use on TETDC-1 and on the official WLTC Class 3b and NEDC traces, with and without Tehran road grades, for summer, winter and spring/autumn auxiliary loads.
4. **Fleet simulation (not used in the article).** A 24 h simulation of 470 electric taxis using the 12 public charging stations (24 plugs) found in OpenStreetMap; it is kept for completeness, because the OpenStreetMap charging inventory is incomplete.
5. **Uncertainty analysis.** 10,000 paired Monte-Carlo draws and Sobol' indices (17,408 samples) per season over 15 uncertain inputs, including battery capacity U(62, 80) kWh; a bias-corrected analysis adds the 15 s reconstruction bias as a 16th input; the Monte-Carlo analysis is repeated for eleven construction runs of the cycle (ensemble).

## Reproducing the results

Requirements: Python 3.12 and about 4 GB of free disk space.

```bash
pip install -r requirements.txt
python scripts/download_data.py   # fetches Porto + DEM data (~0.6 GB) and checks SHA-256 hashes
python run_all.py                 # runs all stages in order (~10 min on one CPU core before the
                                  # ensemble stage was added; uq_ensemble adds 11 cycles x 6 cases x 10,000 draws)
python verify_outputs.py          # compares data/processed/ with results_reference/
```

A single stage can be re-run, e.g. `python run_all.py baselines`.

## Repository layout

| Folder / file | Content |
|---|---|
| `preprocessing/` | OSM network, DEM topography, Porto GPS cleaning, charging-station layer |
| `drive_cycle/` | micro-trip extraction, PCA/K-means clustering, cycle construction |
| `digital_twin/` | vehicle model (`ev_model.py`), parameters with provenance tags (`ev_params.json`), energy simulation |
| `experiments/` | comparison with official cycles, fleet simulation, growth scenarios |
| `analysis/` | trip-chain parameters, sensitivity, statistics, validation |
| `uncertainty/` | Monte-Carlo and Sobol' analysis |
| `figures/` | script that draws every article figure from the outputs (`figures/output/`) |
| `data/raw/` | OpenStreetMap extracts used for the article, zipped (unpacked and checked by `scripts/download_data.py`) |
| `results_reference/` | verified outputs used in the article |
| `VERIFICATION.md` | record of the independent re-execution |
| `DATA_LICENSES.md` | licences of all data sources |

## Supplementary analyses

`supplement/` contains two additional checks: the bias of rebuilding 1 Hz speed from 15 s GPS positions, tested on open 1 Hz data, and a comparison of TETDC-1 with 19 real Tehran driving cycles. See `supplement/README.md`.

The measured bias feeds a second, bias-corrected uncertainty analysis: `uncertainty/uq.py` modes `mc_bc` and `sobol_bc` add the bias as a 16th input (U(0, 2b), b from `supplement/bias_factor.py`), and `uncertainty/analyze.py` writes `uq_results_bc.json` next to the original `uq_results.json`. `figures/make_uq_figures.py` draws the Monte-Carlo figure with both analyses and the reconstruction-bias figure.

## Robustness checks (version 1.0.0, 9 October 2026)

| Script | Question | Output |
|---|---|---|
| `analysis/terrain_scenarios.py` | Energy use at the central grade (2.57 %) and the grade bounds; out-and-back drives over 13 corridor profiles | `terrain_scenarios.json` |
| `analysis/cycle_checks.py` | Size of the recombination bank (37,542 micro-trips from 9,000 traces); micro-trips with accelerations above 4 m/s²; energy of TETDC-1 compared with 8,174 source traces; construction repeated with ten further seeds | `cycle_checks.json` |
| `uncertainty/bias_sensitivity.py` | Range gap under four treatments of the reconstruction bias (none, U(0,b), fixed b, U(0,2b)) | `uq_bias_sensitivity.json` |

`drive_cycle/build_markov.py` accepts the environment variables `TETDC_SEED` and `TETDC_OUTDIR`, which are used only by the seed check; without them it behaves exactly as before. All three scripts run as part of `run_all.py`, and `verify_outputs.py` checks their outputs.

## Ensemble and driving-dynamics analyses (version 1.1.0, 7 November 2026)

| Script | Question | Output |
|---|---|---|
| `uncertainty/uq_ensemble.py` | Paired Monte-Carlo analysis (modes `mc` and `mc_bc`) repeated for all eleven construction runs of the cycle (seed 42 = TETDC-1, seeds 1-10) on identical input draws; pooled over all eleven cycles (110,000 draws per season) and over the eight cycles with composite error below 10 % (80,000 draws) | `uq_ensemble.json` |
| `supplement/dynamics_energy.py` | Does more dynamic driving raise energy use? OLS of 1 Hz energy use on RMS acceleration and mean speed over the 54 Hull trips, 5,000 bootstrap resamples | `results_reference/supplement/dynamics_energy.json` |

`uq_ensemble.py` stores the eleven cycles and their Monte-Carlo outputs in `data/processed/ensemble/seed<k>/` (about 30 MB) and checks that the seed-42 rebuild equals the published `TETDC_1.csv`. It runs in `run_all.py` before the figures; `figures/make_uq_figures.py` adds the pooled bias-corrected distribution to the Monte-Carlo figure as a dashed line, and its legend gives the mean and the share of draws with a positive gap, e.g. "Original: 4.8% (94% >0)", instead of "P>0". `uq_ensemble.json` also splits the pooled variance into between-cycle and within-cycle parts (law of total variance). `supplement/run_supplement.py` now compares its outputs with `results_reference/supplement/` at the end of the run.

Figure numbers in the article: Fig1 workflow, Fig2 speed profiles, Fig3 terrain (`Fig4_terrain`), Fig4 SAFD (`Fig3_SAFD`), Fig5 reconstruction bias (`Fig8_reconstruction_bias`), Fig6 seasonal energy (`Fig5_seasonal_energy`), Fig7 Monte-Carlo (`Fig6_MC_overestimate`), Fig8 Sobol' (`Fig7_Sobol`, two panels).

## Known limitations

These are discussed in the article and should be considered when reusing the results:

* The driving cycle is built from **Porto** taxi data and conditioned on Tehran's network; no Tehran taxi driving data were available.
* The Porto data are sampled every **15 s**; speed and acceleration at 1 Hz are reconstructed by interpolation.
* Copernicus GLO-30 is a **surface** model (buildings and vegetation included), so network-wide grades are likely overestimated; the uncertainty analysis brackets grade between 1.86 % and 3.28 %.
* The Tehran grade scenario is stylised (symmetric uphill/downhill, zero net elevation).
* Vehicle parameters are class-typical values with documented ranges. `digital_twin/ev_params.json` contains a 93 kWh battery, which corresponds to the GAC Aion LX Plus rather than the Aion V Plus used as Tehran's electric taxi; the deterministic pipeline is kept unchanged for traceability, and the uncertainty analysis replaces it with the published Aion V Plus range, U(62, 80) kWh. Energy results in kWh/100 km do not depend on battery size.
* The 24 h fleet simulation (`experiments/run_fleet_sim.py`) relies on an incomplete OpenStreetMap charging inventory (18 mapped elements versus about 126 announced stations) and is not used in the article.
* The charging-station inventory relies on OpenStreetMap and may be incomplete.

## Development note

Parts of this code were developed with generative-AI assistance. All results were independently
re-executed from public raw data and verified as documented in `VERIFICATION.md`; the authors take
full responsibility for the code and results.

## Licence and citation

Code: MIT (see `LICENSE`). Data: see `DATA_LICENSES.md` (OpenStreetMap-derived data are ODbL).
Please cite the article and this repository (`CITATION.cff`; archived version: [ZENODO DOI]).
