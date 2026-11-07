# Data sources and licences

The MIT licence in `LICENSE` covers the code only. Data keep their original licences:

| Data | Where it lives | Licence / terms |
|---|---|---|
| Porto taxi trajectories, ECML/PKDD 2015 (UCI ML Repository, dataset 339, doi:10.24432/C55W25) | downloaded by `scripts/download_data.py` (not redistributed) | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Copernicus DEM GLO-30, tiles N35E050 and N35E051 (AWS open-data bucket `copernicus-dem-30m`) | downloaded by `scripts/download_data.py` (not redistributed) | Copernicus DEM licence, free for the general public; see https://registry.opendata.aws/copernicus-dem/ |
| OpenStreetMap road network and charging stations for Tehran (Overpass API, OSM base 2026-09-11) | `data/raw/osm_tehran_2026-09-11.zip`, `data/raw/overpass_charging_tehran.json` | © OpenStreetMap contributors, Open Database License (ODbL) 1.0 |
| Outputs derived from OpenStreetMap (network, grades, corridor profiles, charging layer) | `results_reference/`, regenerated in `data/processed/` | ODbL 1.0 (share-alike), © OpenStreetMap contributors |
| Official WLTC Class 3b and NEDC speed traces | from the `wltp` Python package | see the `wltp` package licence |

### Supplementary analyses (`supplement/`)

| Data | Where it lives | Licence / terms |
|---|---|---|
| Hull, C. (2024) 1Hz GPS Tracking Data, Mendeley Data V3, doi:10.17632/xt69cnwh56.3 (minibus-taxi trips, South Africa) | downloaded by `supplement/download_supplement_data.py` (not redistributed) | CC BY 4.0 |
| Azadi, Bagheri, Masih-Tehrani, Moosavian (2024) Mendeley Data V1, doi:10.17632/f7c75syg2f.1, file `OutputData.xlsx` | downloaded by `supplement/download_supplement_data.py` (not redistributed) | CC BY 4.0 |
| US EPA FTP-75 and UDDS speed traces (`ftpcol.txt`, `uddscol.txt`) | downloaded by `supplement/download_supplement_data.py` | US Government work, public domain |
| Derived summaries | `results_reference/supplement/` | follow the licences of the inputs above |
