import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset.csv').dropna(subset=['datetime']).reset_index(drop=True).iloc[:2000]
df['datetime_dt'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
expiry = pd.to_datetime('2026-01-27')
df['dte'] = (expiry - df['datetime_dt']).dt.total_seconds() / (24*3600*365.25)

option_cols = [c for c in df.columns if c.startswith('NIFTY') and (c.endswith('CE') or c.endswith('PE'))]
ce_cols = sorted([c for c in option_cols if c.endswith('CE')])
pe_cols = sorted([c for c in option_cols if c.endswith('PE')])

strikes_ce = np.array([float(c.replace('NIFTY27JAN26','').replace('CE','')) for c in ce_cols])
strikes_pe = np.array([float(c.replace('NIFTY27JAN26','').replace('PE','')) for c in pe_cols])

def get_mask(df, option_cols, seed=42):
    rng = np.random.RandomState(seed)
    masked = df.copy()
    truth = []
    for c in option_cols:
        obs = df.index[df[c].notna()].tolist()
        k = max(1, int(len(obs) * 0.15))
        hide = rng.choice(obs, size=k, replace=False)
        for ix in hide:
            truth.append((ix, c, df.at[ix, c]))
            masked.at[ix, c] = np.nan
    return masked, truth

masked, truth = get_mask(df, option_cols)

def eval_linear_var(masked, truth):
    filled = masked[option_cols].copy()
    for ix, row in masked.iterrows():
        S = row['underlying_price']
        tau = max(1e-6, row['dte'])
        # CE
        v_s = [strikes_ce[j] for j, c in enumerate(ce_cols) if pd.notna(row[c])]
        v_iv = [row[c] for j, c in enumerate(ce_cols) if pd.notna(row[c])]
        if len(v_s) >= 2:
            var = np.array(v_iv)**2 * tau
            f = interp1d(v_s, var, kind='linear', fill_value='extrapolate')
            for j, c in enumerate(ce_cols):
                if pd.isna(row[c]): 
                    v = float(f(strikes_ce[j]))
                    filled.at[ix, c] = np.sqrt(max(0, v) / tau)
        elif len(v_s) == 1:
            for c in ce_cols:
                if pd.isna(row[c]): filled.at[ix, c] = v_iv[0]
        # PE
        v_s = [strikes_pe[j] for j, c in enumerate(pe_cols) if pd.notna(row[c])]
        v_iv = [row[c] for j, c in enumerate(pe_cols) if pd.notna(row[c])]
        if len(v_s) >= 2:
            var = np.array(v_iv)**2 * tau
            f = interp1d(v_s, var, kind='linear', fill_value='extrapolate')
            for j, c in enumerate(pe_cols):
                if pd.isna(row[c]): 
                    v = float(f(strikes_pe[j]))
                    filled.at[ix, c] = np.sqrt(max(0, v) / tau)
        elif len(v_s) == 1:
            for c in pe_cols:
                if pd.isna(row[c]): filled.at[ix, c] = v_iv[0]

    y_t, y_p = [], []
    for ix, c, tv in truth:
        pv = filled.at[ix, c]
        if pd.notna(pv): y_t.append(tv); y_p.append(np.clip(pv, 0.01, 6.0))
    return mean_squared_error(y_t, y_p)

print("Linear Variance MSE:", eval_linear_var(masked, truth))
