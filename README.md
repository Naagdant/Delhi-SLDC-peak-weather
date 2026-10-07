# Weather sensitivity of Delhi's daily peak electricity demand

Data and code for the paper **"Weather Sensitivity of the Daily Peak Electricity Demand of Delhi: A Season-Resolved Analysis of Load Despatch Data"** by K. D. Bodha, V. Arun and A. Awasthi.

The study relates 972 days of daily peak demand published by the State Load Despatch Centre (SLDC), Delhi (13 September 2023 to 6 October 2026) to daily ERA5 reanalysis weather. It estimates a balance-temperature (hinge) model, season-wise regressions with Newey–West errors, and an out-of-sample random forest with permutation importance and partial dependence.

## Main results

| Quantity | Value |
|---|---|
| Balance temperature (mean temperature) | 20.9 °C (95% block bootstrap 20.0–22.0 °C) |
| Cooling / heating slope of the peak | +251 / +170 MW per °C away from the balance point |
| Sensitivity by season (MW/°C) | Monsoon 434, Pre-monsoon 392, Post-monsoon 150, Winter −115, March not significant |
| Pre-monsoon days with Tmax ≥ 40 °C | peak 23% higher |
| Random forest on unseen period (weather only) | MAPE 5.5%, R² 0.84 |

## Repository layout

```
data/
  delhi_peak_2023_2026.csv        daily peak (MW) and time of peak, Delhi SLDC
  delhi_weather_openmeteo.csv     daily ERA5 weather from the Open-Meteo archive API
  delhi_peak_weather_merged.csv   cleaned and merged data set (972 days)
prepare_data.py                   raw files -> merged data set
analysis.py                       all models, statistics, tables and figures
results/
  tables.json                     every number quoted in the paper
  season_regression.csv           season-wise regression coefficients
figs/                             figures 1-9 of the paper
```

## Reproduce

```
pip install -r requirements.txt
python prepare_data.py
python analysis.py
```

`analysis.py` takes about 30 s and rewrites `results/` and `figs/`.

## Data sources and notes

- **Demand:** State Load Despatch Centre, Delhi, "Monthly peak load details", https://www.delhisldc.org/monthlypeak.aspx (accessed 7 October 2026). The archive had no data before September 2023 or from late November 2025 to early April 2026. Four days recorded as 0 MW were removed as telemetry errors.
- **Weather:** ERA5 reanalysis (Hersbach et al., 2020) through the Open-Meteo historical weather API (Zippenfenig, 2023, doi:10.5281/zenodo.7970649), grid point 28.61° N, 77.21° E, daily values in Asia/Kolkata time.
- The demand data belong to Delhi SLDC and the weather data to ECMWF/Copernicus and Open-Meteo, under their own terms. The MIT licence covers the code in this repository.

## Citation

K. D. Bodha, V. Arun and A. Awasthi, "Weather Sensitivity of the Daily Peak Electricity Demand of Delhi: A Season-Resolved Analysis of Load Despatch Data," submitted to IIPESS 2027.
