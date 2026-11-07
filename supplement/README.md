# Supplementary analyses

Two checks that are not part of the main pipeline but support the discussion of data limitations.

| Script | Question | Inputs | Output |
|---|---|---|---|
| `download_supplement_data.py` | fetch and hash-check the open data | internet | `data/supplement/`, `manifest.json` |
| `hull_split.py` | which Hull files are raw 1 Hz trips | Hull data | `hull_split.json` |
| `interp_bias.py` | how much does rebuilding 1 Hz speed from 15 s positions (the exact `trip_1hz_speed` of the pipeline) change kinematics and energy? | Hull 1 Hz trips | `interp_bias_per_trip.csv`, `interp_bias_summary.json` |
| `interp_control.py` | same reconstruction from 1, 5, 10 and 15 s positions; bias by driving dynamics | Hull 1 Hz trips | `interp_control_per_trip.csv`, `interp_control_summary.json` |
| `bias_factor.py` | aggregate energy bias b per season, with bootstrap interval; input to the bias-corrected uncertainty analysis | `interp_bias_per_trip.csv` | `bias_factor.json` |
| `dynamics_energy.py` | does more dynamic driving raise energy use at constant mean speed? (OLS on RMS acceleration and mean speed, bootstrap interval) | `interp_bias_per_trip.csv` | `dynamics_energy.json` |
| `tehran19_compare.py` | where does TETDC-1 sit among the 19 real Tehran cycles of Bagheri et al. (2025)? | `OutputData.xlsx`, EPA traces, `TETDC_1.csv` | `tehran19_comparison.json` |

Run:

```bash
python supplement/download_supplement_data.py      # about 30 MB
python supplement/run_supplement.py                # about 20 s
```

Reference outputs are in `results_reference/supplement/`; a run on 2026-10-08 reproduced them exactly.

Limits: the 1 Hz ground truth comes from minibus taxis in South Africa, not from Porto or Tehran taxis, so the
bias is indicative. The metric definitions of Bagheri et al. (2025) were inferred by calibration on FTP-75/UDDS
(mean relative error 17 %), so the Tehran comparison is approximate.

Verified run: Python 3.13, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, openpyxl 3.1.5 (the main pipeline
versions in `requirements.txt` also work; only numpy, scipy, pandas and openpyxl are needed here).
