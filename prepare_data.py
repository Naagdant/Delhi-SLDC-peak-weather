"""Build data/delhi_peak_weather_merged.csv from the two raw files.
Raw inputs (both public):
  data/delhi_peak_2023_2026.csv   Delhi SLDC monthly peak-load page, https://www.delhisldc.org/monthlypeak.aspx (accessed 7 Oct 2026)
  data/delhi_weather_openmeteo.csv  Open-Meteo ERA5 archive, 28.61 N 77.21 E, daily, timezone Asia/Kolkata
Cleaning: SLDC records with peak 0 MW (telemetry errors) are dropped; days without weather are dropped.
"""
import io, pandas as pd
p = pd.read_csv('data/delhi_peak_2023_2026.csv', parse_dates=['date'])
raw = open('data/delhi_weather_openmeteo.csv').read().split('\n\n', 1)[1]
w = pd.read_csv(io.StringIO(raw))
w.columns = ['date', 'tmax', 'tmin', 'tmean', 'app_tmax', 'rh', 'dew', 'rain', 'wind', 'rad']
w['date'] = pd.to_datetime(w['date'])
n_raw = len(p); zero = int((p.peak_mw < 1500).sum())
p = p[p.peak_mw >= 1500]
m = p.merge(w, on='date'); m['month'] = m.date.dt.month
print(f'SLDC days: {n_raw}, zero-MW records removed: {zero}, days without weather: {len(p) - len(m)}, final: {len(m)}')
m.to_csv('data/delhi_peak_weather_merged.csv', index=False)
