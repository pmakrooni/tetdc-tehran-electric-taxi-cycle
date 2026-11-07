# Change log

## 1.2.0 — 7 November 2026

* `uncertainty/uq_ensemble.py`: adds a between-/within-cycle variance split (law of total variance) for each pool; all earlier keys of `uq_ensemble.json` are unchanged.
* `supplement/run_supplement.py`: now compares every file in `results_reference/supplement/` with the fresh run (including the data manifest) and exits with an error if any differs; comment in `verify_outputs.py` corrected.
* `figures/make_figures.py`: workflow figure shows the 11 construction runs. `figures/make_uq_figures.py`: larger legend in the Monte-Carlo figure.
* README: description of the Monte-Carlo legend corrected. Verified again: `verify_outputs.py` 0 problems; `run_supplement.py` 0 problems.

## 1.1.0 — 7 November 2026

* New `uncertainty/uq_ensemble.py`: paired Monte-Carlo analysis (original and bias-corrected) repeated for all eleven construction runs of the cycle on identical input draws; pooled results in `results_reference/uq_ensemble.json`. Added to `run_all.py` before the figures.
* New `supplement/dynamics_energy.py`: regression of 1 Hz energy use on RMS acceleration and mean speed over the 54 Hull trips, with bootstrap interval; output `results_reference/supplement/dynamics_energy.json`. Added to `supplement/run_supplement.py`.
* `figures/make_uq_figures.py`: legend reads "share >0" instead of "P>0"; pooled bias-corrected ensemble distribution added as a dashed line; x-axis range widened to −20…30 % so that the ensemble tails are visible.
* `CITATION.cff`: second author and affiliations added; version 1.1.0.
* README, VERIFICATION.md and supplement/README.md updated. All earlier reference outputs were reproduced exactly from freshly downloaded raw data (`verify_outputs.py`: 0 problems).

## 1.0.0 — 9 October 2026

* Robustness checks `analysis/terrain_scenarios.py`, `analysis/cycle_checks.py`, `uncertainty/bias_sensitivity.py`; bias-corrected uncertainty analysis (`uq.py` modes `mc_bc`, `sobol_bc`); supplementary 1 Hz and Tehran-cycle checks.
