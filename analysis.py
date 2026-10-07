"""Delhi SLDC daily peak demand vs weather. Regenerates every number (tables.json) and figure used in the paper.
Run: python3 analysis.py
"""
import json, numpy as np, pandas as pd, statsmodels.api as sm, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error

N = {}
def fp(p): return '<0.001' if p < 0.001 else f'{p:.3f}'
def put(k, v): N[k] = v

m = pd.read_csv('data/delhi_peak_weather_merged.csv', parse_dates=['date']).sort_values('date').reset_index(drop=True)
assert (m.peak_mw > 1500).all()
m['mo'] = m.date.dt.month
m['wkend'] = (m.date.dt.dayofweek >= 5).astype(int)
m['t_yr'] = (m.date - m.date.min()).dt.days / 365.25          # linear trend (years)
h = m.peak_time.str.slice(0, 2).astype(int) + m.peak_time.str.slice(3, 5).astype(int) / 60
m['hour'] = h
m['night'] = ((h >= 20) | (h < 3)).astype(int)                  # 20:00-03:00
m['afternoon'] = ((h >= 12) & (h < 18)).astype(int)
m['morning'] = ((h >= 6) & (h < 12)).astype(int)
m['evening'] = ((h >= 18) & (h < 20)).astype(int)
SEAS = [('Winter', [12, 1, 2]), ('March', [3]), ('Pre-monsoon', [4, 5, 6]), ('Monsoon', [7, 8, 9]), ('Post-monsoon', [10, 11])]
m['season'] = m.mo.map({mo: s for s, mos in SEAS for mo in mos})
ORDER = [s for s, _ in SEAS]

# ---------------- data summary
put('n_days', len(m)); put('d0', m.date.min().strftime('%-d %B %Y')); put('d1', m.date.max().strftime('%-d %B %Y'))
put('peak_max', f"{m.peak_mw.max():.0f}"); put('peak_max_date', m.loc[m.peak_mw.idxmax(), 'date'].strftime('%-d %B %Y'))
put('peak_min', f"{m.peak_mw.min():.0f}"); put('peak_mean', f"{m.peak_mw.mean():.0f}")
for s in ORDER: put(f'n_{s}', int((m.season == s).sum()))
put('tmax_max', f"{m.tmax.max():.1f}"); put('tmin_min', f"{m.tmin.min():.1f}")

# ---------------- whole-period correlations
for v in ['tmax', 'tmin', 'tmean', 'app_tmax', 'rh', 'dew', 'wind', 'rain', 'rad']:
    put(f'rho_{v}', f"{spearmanr(m[v], m.peak_mw)[0]:.2f}")

# ---------------- hinge (balance-point) model on tmean with trend + weekend, HAC SE
def hinge_fit(Tb, d):
    X = pd.DataFrame({'heat': np.maximum(Tb - d.tmean, 0), 'cool': np.maximum(d.tmean - Tb, 0),
                      't_yr': d.t_yr, 'wkend': d.wkend})
    return sm.OLS(d.peak_mw, sm.add_constant(X)).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
grid = np.arange(14, 28.01, 0.1)
sse = [hinge_fit(Tb, m).ssr for Tb in grid]
Tb = grid[int(np.argmin(sse))]
hf = hinge_fit(Tb, m)
put('Tb', f"{Tb:.1f}")
# bootstrap (moving blocks of 14 days) for Tb CI
rng = np.random.default_rng(1); B = 300; L = 14; tbs = []
idx_starts = np.arange(len(m) - L)
for _ in range(B):
    st = rng.choice(idx_starts, size=len(m) // L + 1); ii = np.concatenate([np.arange(s, s + L) for s in st])[:len(m)]
    d = m.iloc[ii].reset_index(drop=True)
    tbs.append(grid[int(np.argmin([hinge_fit(t, d).ssr for t in grid[::5]]) * 5)] if False else grid[::5][int(np.argmin([hinge_fit(t, d).ssr for t in grid[::5]]))])
put('Tb_lo', f"{np.percentile(tbs, 2.5):.1f}"); put('Tb_hi', f"{np.percentile(tbs, 97.5):.1f}")
for k, nm in [('heat', 'heat'), ('cool', 'cool'), ('t_yr', 'trend'), ('wkend', 'wk')]:
    put(f'h_{nm}', f"{hf.params[k]:.0f}"); put(f'h_{nm}_lo', f"{hf.conf_int().loc[k, 0]:.0f}"); put(f'h_{nm}_hi', f"{hf.conf_int().loc[k, 1]:.0f}")
    put(f'h_{nm}_p', hf.pvalues[k])
put('h_R2', f"{hf.rsquared:.2f}")
put('trend_pct', f"{100 * hf.params['t_yr'] / m.peak_mw.mean():.1f}")

# ---------------- robustness: hinge model with alternative temperature metrics
def hinge_any(var, d):
    g = np.arange(np.percentile(d[var], 5), np.percentile(d[var], 95), 0.1)
    def fit(t):
        X = pd.DataFrame({'heat': np.maximum(t - d[var], 0), 'cool': np.maximum(d[var] - t, 0), 't_yr': d.t_yr, 'wkend': d.wkend})
        return sm.OLS(d.peak_mw, sm.add_constant(X)).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
    t = g[int(np.argmin([fit(x).ssr for x in g]))]; r = fit(t)
    return t, r
tab4 = []
for var, nm in [('tmean', 'Mean temperature'), ('tmax', 'Maximum temperature'), ('tmin', 'Minimum temperature'), ('app_tmax', 'Apparent max. temperature')]:
    t, r = hinge_any(var, m)
    tab4.append([nm, f"{t:.1f}", f"{r.params['heat']:.0f}", f"{r.params['cool']:.0f}", f"{r.rsquared:.2f}", f"{np.sqrt(r.mse_resid):.0f}"])
    put(f'rb_Tb_{var}', f"{t:.1f}"); put(f'rb_R2_{var}', f"{r.rsquared:.2f}"); put(f'rb_cool_{var}', f"{r.params['cool']:.0f}"); put(f'rb_heat_{var}', f"{r.params['heat']:.0f}")
    put(f'rb_rmse_{var}', f"{np.sqrt(r.mse_resid):.0f}")
# monsoon vs pre-monsoon at equal temperature: residual of hinge model
m['hres'] = hf.resid
for s_ in ['Pre-monsoon', 'Monsoon']:
    put(f'hres_{s_}', f"{m.loc[m.season == s_, 'hres'].mean():.0f}")
put('hres_p', fp(mannwhitneyu(m.loc[m.season == 'Monsoon', 'hres'], m.loc[m.season == 'Pre-monsoon', 'hres']).pvalue))
# ---------------- season-wise regression with trend + HAC
rows = []
for s in ORDER:
    g = m[m.season == s]
    cols = ['tmean', 'rh', 'wind', 'rain', 'wkend', 't_yr']
    r = sm.OLS(g.peak_mw, sm.add_constant(g[cols])).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
    ci = r.conf_int()
    rho = spearmanr(g.tmean, g.peak_mw)[0]
    rows.append(dict(season=s, n=len(g), b=r.params['tmean'], lo=ci.loc['tmean', 0], hi=ci.loc['tmean', 1], p=r.pvalues['tmean'],
                     rh=r.params['rh'], rh_p=r.pvalues['rh'], wind=r.params['wind'], wind_p=r.pvalues['wind'],
                     rain=r.params['rain'], rain_p=r.pvalues['rain'], wk=r.params['wkend'], wk_p=r.pvalues['wkend'], R2=r.rsquared, rho=rho,
                     mean=g.peak_mw.mean()))
S = pd.DataFrame(rows).set_index('season')
S.to_csv('results/season_regression.csv')
tab2 = []
for s in ORDER:
    r = S.loc[s]
    tab2.append([s, str(int(r.n)), f"{r.b:.0f} [{r.lo:.0f}, {r.hi:.0f}]", fp(r.p), f"{r.rh:.1f}", fp(r.rh_p), f"{r.wk:.0f}", f"{r.R2:.2f}"])
    put(f'b_{s}', f"{r.b:.0f}"); put(f'lo_{s}', f"{r.lo:.0f}"); put(f'hi_{s}', f"{r.hi:.0f}"); put(f'p_{s}', fp(r.p))
    put(f'R2_{s}', f"{r.R2:.2f}"); put(f'rho_{s}', f"{r.rho:.2f}"); put(f'wk_{s}', f"{r.wk:.0f}"); put(f'rh_{s}', f"{r.rh:.1f}"); put(f'rhp_{s}', fp(r.rh_p))
    put(f'wind_{s}', f"{r.wind:.1f}"); put(f'windp_{s}', fp(r.wind_p)); put(f'rain_{s}', f"{r.rain:.1f}"); put(f'rainp_{s}', fp(r.rain_p))
    put(f'mean_{s}', f"{r['mean']:.0f}")

# ---------------- heat days and humid nights
pm = m[m.season == 'Pre-monsoon']
hot = pm[pm.tmax >= 40]; cool = pm[pm.tmax < 40]
put('n_hot', len(hot)); put('hot_mean', f"{hot.peak_mw.mean():.0f}"); put('nothot_mean', f"{cool.peak_mw.mean():.0f}")
put('hot_pct', f"{100 * (hot.peak_mw.mean() / cool.peak_mw.mean() - 1):.0f}"); put('hot_p', fp(mannwhitneyu(hot.peak_mw, cool.peak_mw).pvalue))
warm = m[m.season.isin(['Pre-monsoon', 'Monsoon'])]
put('night_share_warm', f"{100 * warm.night.mean():.0f}")
for s in ORDER:
    g = m[m.season == s]
    put(f'night_{s}', f"{100 * g.night.mean():.0f}"); put(f'aft_{s}', f"{100 * g.afternoon.mean():.0f}")
    put(f'morn_{s}', f"{100 * g.morning.mean():.0f}"); put(f'eve_{s}', f"{100 * g.evening.mean():.0f}")
    put(f'medh_{s}', f"{g.hour.median():.1f}")
nd, ad = warm[warm.night == 1], warm[warm.night == 0]
put('dew_night', f"{nd.dew.mean():.1f}"); put('dew_day', f"{ad.dew.mean():.1f}"); put('dew_p', fp(mannwhitneyu(nd.dew, ad.dew).pvalue))
put('tmin_night', f"{nd.tmin.mean():.1f}"); put('tmin_day', f"{ad.tmin.mean():.1f}"); put('tmin_p', fp(mannwhitneyu(nd.tmin, ad.tmin).pvalue))
put('rad_night', f"{nd.rad.mean():.1f}"); put('rad_day', f"{ad.rad.mean():.1f}"); put('rad_p', fp(mannwhitneyu(nd.rad, ad.rad).pvalue))
put('pk_night', f"{nd.peak_mw.mean():.0f}"); put('pk_day', f"{ad.peak_mw.mean():.0f}")
put('n_night_warm', len(nd)); put('n_warm', len(warm))

# ---------------- out-of-sample Random Forest (time-ordered split)
split = pd.Timestamp('2025-07-01')
tr, te = m[m.date < split], m[m.date >= split]
put('n_tr', len(tr)); put('n_te', len(te))
W = ['tmax', 'tmin', 'rh', 'dew', 'wind', 'rain', 'rad', 'wkend']
res = {}
def evalm(name, cols, model):
    model.fit(tr[cols], tr.peak_mw); p = model.predict(te[cols])
    res[name] = (r2_score(te.peak_mw, p), 100 * mean_absolute_percentage_error(te.peak_mw, p), mean_absolute_error(te.peak_mw, p))
    return p
class OLSm:
    def fit(self, X, y): self.r = sm.OLS(y, sm.add_constant(X)).fit(); return self
    def predict(self, X): return self.r.predict(sm.add_constant(X, has_constant='add'))
rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=0)
p_rf = evalm('RF', W, rf)
evalm('OLS', W, OLSm())
for k in res: put(f'r2_{k}', f"{res[k][0]:.2f}"); put(f'mape_{k}', f"{res[k][1]:.1f}"); put(f'mae_{k}', f"{res[k][2]:.0f}")
# seasonal-naive benchmark: same calendar day one year earlier is not available everywhere; use 7-day persistence
pers = m.set_index('date').peak_mw.shift(7).reindex(te.date).values; ok = ~np.isnan(pers)
put('mape_pers7', f"{100 * mean_absolute_percentage_error(te.peak_mw.values[ok], pers[ok]):.1f}")
res['Persistence (7 days)'] = (r2_score(te.peak_mw.values[ok], pers[ok]), 100 * mean_absolute_percentage_error(te.peak_mw.values[ok], pers[ok]), mean_absolute_error(te.peak_mw.values[ok], pers[ok]))
put('n_pers', int(ok.sum()))
q90 = te.peak_mw.quantile(0.9); put('top10_bias', f"{(p_rf - te.peak_mw.values)[te.peak_mw.values > q90].mean():.0f}")
put('train_max', f"{tr.peak_mw.max():.0f}"); put('pred_max', f"{p_rf.max():.0f}"); put('n_above_train', int((te.peak_mw > tr.peak_mw.max()).sum()))
pi = permutation_importance(rf, te[W], te.peak_mw, n_repeats=30, random_state=0, scoring='r2')
imp = pd.Series(pi.importances_mean, W).sort_values(ascending=False); ims = pd.Series(pi.importances_std, W)
for k in W: put(f'imp_{k}', f"{imp[k]:.2f}")
# RF without tmin (robustness: does dew/tmax take over?)
rf2 = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=0)
evalm('RF_no_tmin', [c for c in W if c != 'tmin'], rf2)
put('r2_RF_no_tmin', f"{res['RF_no_tmin'][0]:.2f}"); put('mape_RF_no_tmin', f"{res['RF_no_tmin'][1]:.1f}")

# ---------------- Table I: descriptive statistics
tab1 = []
for v, nm, u in [('peak_mw', 'Daily peak demand', 'MW'), ('tmax', 'Maximum temperature', '°C'), ('tmin', 'Minimum temperature', '°C'),
                 ('tmean', 'Mean temperature', '°C'), ('rh', 'Relative humidity', '%'), ('dew', 'Dew point', '°C'),
                 ('wind', 'Maximum wind speed', 'km/h'), ('rain', 'Precipitation', 'mm'), ('rad', 'Shortwave radiation', 'MJ/m²')]:
    x = m[v]; f = (lambda z: f'{z:.0f}') if v == 'peak_mw' else (lambda z: f'{z:.1f}')
    tab1.append([nm, u, f(x.mean()), f(x.std()), f(x.min()), f(x.max()), '1.00' if v == 'peak_mw' else f"{spearmanr(x, m.peak_mw)[0]:.2f}"])

json.dump({'N': {k: (str(v) if not isinstance(v, (int, float)) else v) for k, v in N.items()}, 'tab1': tab1, 'tab2': tab2,
           'tab3': [[k, f"{res[k][0]:.2f}", f"{res[k][1]:.1f}", f"{res[k][2]:.0f}"] for k in ['Persistence (7 days)', 'OLS', 'RF_no_tmin', 'RF']]},
          open('results/tables.json', 'w'), indent=1)

# ======================= figures (IEEE style)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 8, 'savefig.dpi': 400,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'legend.frameon': False})
W1, W2 = 3.42, 7.10
COL = {'Winter': '#2b6cb0', 'March': '#38a169', 'Pre-monsoon': '#c05621', 'Monsoon': '#6b46c1', 'Post-monsoon': '#718096'}

# Fig 1 time series
fig, ax = plt.subplots(figsize=(W1, 2.1))
s = m.set_index('date').asfreq('D')
ax.plot(s.index, s.peak_mw / 1000, lw=0.6, c='#1a202c', label='Daily peak demand')
ax.set_ylabel('Peak demand (GW)')
ax2 = ax.twinx(); ax2.plot(s.index, s.tmean, lw=0.5, c='#c05621', alpha=0.8, label='Mean temperature'); ax2.set_ylabel('Mean temperature (°C)', color='#c05621')
ax2.spines['right'].set_visible(True); ax2.tick_params(axis='y', colors='#c05621')
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); ax.legend(h1+h2,l1+l2,loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=2,fontsize=6.5)
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
fig.tight_layout(); fig.savefig('figs/fig1_series.png'); plt.close(fig)

# Fig 2 scatter + hinge
fig, ax = plt.subplots(figsize=(W1, 2.6))
for s_ in ORDER:
    g = m[m.season == s_]; ax.scatter(g.tmean, g.peak_mw / 1000, s=4, alpha=0.65, c=COL[s_], label=s_, lw=0)
tt = np.linspace(m.tmean.min(), m.tmean.max(), 200)
yy = hf.params['const'] + hf.params['heat'] * np.maximum(Tb - tt, 0) + hf.params['cool'] * np.maximum(tt - Tb, 0) + hf.params['t_yr'] * m.t_yr.mean() + hf.params['wkend'] * m.wkend.mean()
ax.plot(tt, yy / 1000, c='k', lw=1.0, label=f'Hinge fit ($T_b$ = {Tb:.1f} °C)')
ax.set_xlabel('Daily mean temperature (°C)'); ax.set_ylabel('Daily peak demand (GW)'); ax.legend(fontsize=6, loc='upper left', markerscale=2)
fig.tight_layout(); fig.savefig('figs/fig2_scatter.png'); plt.close(fig)

# Fig 3 seasonal sensitivity with 95% CI
fig, ax = plt.subplots(figsize=(W1, 2.2))
y = np.arange(len(ORDER))
for i, s_ in enumerate(ORDER):
    r = S.loc[s_]; ax.errorbar(r.b, i, xerr=[[r.b - r.lo], [r.hi - r.b]], fmt='o', c=COL[s_], ms=4, capsize=2, lw=1)
ax.axvline(0, c='grey', lw=0.6, ls='--'); ax.set_yticks(y); ax.set_yticklabels(ORDER); ax.invert_yaxis()
ax.set_xlabel('Sensitivity of peak demand (MW/°C)')
fig.tight_layout(); fig.savefig('figs/fig3_sensitivity.png'); plt.close(fig)

# Fig 4 time of daily peak by season
fig, ax = plt.subplots(figsize=(W1, 2.3))
bins = np.arange(0, 24.01, 1)
for s_ in ['Winter', 'Pre-monsoon', 'Monsoon', 'Post-monsoon']:
    g = m[m.season == s_]; hh, _ = np.histogram(g.hour, bins=bins); ax.plot(bins[:-1] + 0.5, 100 * hh / len(g), marker='o', ms=2.5, lw=0.9, c=COL[s_], label=s_)
ax.set_xlabel('Hour of daily peak (IST)'); ax.set_ylabel('Share of days (%)'); ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24]); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig('figs/fig4_peakhour.png'); plt.close(fig)

# Fig 5 heat days and humid nights (two panels)
fig, ax = plt.subplots(1, 2, figsize=(W1, 2.3))
bp = ax[0].boxplot(flierprops=dict(markersize=2), x=[cool.peak_mw / 1000, hot.peak_mw / 1000], widths=0.5, patch_artist=True, medianprops=dict(color='k'))
for b_, c_ in zip(bp['boxes'], ['#cbd5e0', '#c05621']): b_.set_facecolor(c_)
ax[0].set_xticks([1, 2]); ax[0].set_xticklabels(['< 40 °C', '≥ 40 °C'], fontsize=7); ax[0].set_xlabel('Daily maximum temperature', fontsize=7); ax[0].set_ylabel('Peak demand (GW)'); ax[0].set_title('Apr–Jun', fontsize=7)
bp = ax[1].boxplot(flierprops=dict(markersize=2), x=[ad.dew, nd.dew], widths=0.5, patch_artist=True, medianprops=dict(color='k'))
for b_, c_ in zip(bp['boxes'], ['#fbd38d', '#4c51bf']): b_.set_facecolor(c_)
ax[1].set_xticks([1, 2]); ax[1].set_xticklabels(['Day', 'Night'], fontsize=7); ax[1].set_xlabel('Time of daily peak', fontsize=7); ax[1].set_ylabel('Dew point (°C)'); ax[1].set_title('Apr–Sep', fontsize=7)
fig.tight_layout(); fig.savefig('figs/fig5_heat_night.png'); plt.close(fig)

# Fig 6 out-of-sample RF ; Fig 7 importance
import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(W1, 2.1))
tt_ = te.assign(pred=p_rf).set_index('date').asfreq('D')
ax.plot(tt_.index, tt_.peak_mw / 1000, lw=0.7, c='#1a202c', label='Observed')
ax.plot(tt_.index, tt_.pred / 1000, lw=0.7, c='#c05621', label=f"Random forest (MAPE {res['RF'][1]:.1f}%)")
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10])); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
ax.set_ylabel('Peak demand (GW)'); ax.legend(fontsize=6.5, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout(); fig.savefig('figs/fig6_rf.png'); plt.close(fig)
fig, ax = plt.subplots(figsize=(W1, 1.9))
lab = {'tmax': 'Max temperature', 'tmin': 'Min temperature', 'rh': 'Relative humidity', 'dew': 'Dew point', 'wind': 'Wind speed', 'rain': 'Precipitation', 'rad': 'Solar radiation', 'wkend': 'Weekend'}
o = imp.sort_values()
ax.barh([lab[k] for k in o.index], o.values, xerr=ims[o.index].values, color='#c05621', error_kw=dict(lw=0.6, capsize=1.5), height=0.6)
ax.set_xlabel('Permutation importance (drop in $R^2$)')
fig.tight_layout(); fig.savefig('figs/fig7_importance.png'); plt.close(fig)
from sklearn.inspection import partial_dependence
fig, ax = plt.subplots(1, 2, figsize=(W1, 1.9), sharey=True)
for k_, (v_, lab_) in enumerate([('tmin', 'Minimum temperature (°C)'), ('dew', 'Dew point (°C)')]):
    pdp = partial_dependence(rf, m[W], [v_], grid_resolution=50, percentiles=(0.02, 0.98))
    xs = pdp['grid_values'][0]; ys = pdp['average'][0]
    ax[k_].plot(xs, ys / 1000, c='#c05621', lw=1.2); ax[k_].set_xlabel(lab_)
    ax[k_].plot(m[v_], np.full(len(m), 4.0), '|', ms=3, c='grey', alpha=0.3)
    if v_ == 'tmin': put('pdp_tmin_lo', f"{ys.min():.0f}"); put('pdp_tmin_hi', f"{ys.max():.0f}")
    if v_ == 'dew': put('pdp_dew_lo', f"{ys.min():.0f}"); put('pdp_dew_hi', f"{ys.max():.0f}")
ax[0].set_ylabel('Partial dependence (GW)')
fig.tight_layout(); fig.savefig('figs/fig8_pdp.png'); plt.close(fig)
json.dump({'N': {k: (str(v) if not isinstance(v, (int, float)) else v) for k, v in N.items()}, 'tab1': tab1, 'tab2': tab2, 'tab4': tab4,
           'tab3': [[k, f"{res[k][0]:.2f}", f"{res[k][1]:.1f}", f"{res[k][2]:.0f}"] for k in ['Persistence (7 days)', 'OLS', 'RF_no_tmin', 'RF']]},
          open('results/tables.json', 'w'), indent=1)
# Fig 9 monthly maximum peak by year
mm = m.assign(yr=m.date.dt.year).groupby(['yr', 'mo']).agg(pk=('peak_mw', 'max'), n=('peak_mw', 'size')).reset_index()
mm = mm[mm.n >= 20]
fig, ax = plt.subplots(figsize=(W1, 2.0))
for yr, c_ in zip([2023, 2024, 2025, 2026], ['#a0aec0', '#2b6cb0', '#38a169', '#c05621']):
    g = mm[mm.yr == yr]; ax.plot(g.mo, g.pk / 1000, marker='o', ms=3, lw=1, c=c_, label=str(yr))
ax.set_xticks(range(1, 13)); ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
ax.set_xlabel('Month'); ax.set_ylabel('Monthly maximum (GW)'); ax.legend(fontsize=6.5, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(); fig.savefig('figs/fig9_monthly.png'); plt.close(fig)
for yr in [2024, 2025, 2026]:
    for mo, nm in [(5, 'May'), (6, 'Jun'), (7, 'Jul')]:
        r = mm[(mm.yr == yr) & (mm.mo == mo)]
        if len(r): put(f'mx_{nm}{yr}', f"{r.pk.iloc[0]:.0f}")
pre_t = {yr: m[(m.date.dt.year == yr) & m.mo.isin([5, 6])].tmean.mean() for yr in [2024, 2025, 2026]}
for yr, v in pre_t.items(): put(f'tMJ_{yr}', f"{v:.1f}")
_t = json.load(open('results/tables.json')); _t['N'] = {k: (str(v) if not isinstance(v, (int, float)) else v) for k, v in N.items()}; json.dump(_t, open('results/tables.json', 'w'), indent=1)
print(json.dumps(N, indent=0, default=str)[:4000])
