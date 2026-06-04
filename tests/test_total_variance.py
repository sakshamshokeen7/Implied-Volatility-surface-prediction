import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset.csv').dropna(subset=['datetime']).reset_index(drop=True).iloc[:2000]

df['datetime_dt'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
expiry = pd.to_datetime('2026-01-27 15:30')
df['dte'] = (expiry - df['datetime_dt']).dt.total_seconds() / (24*3600)
df['dte'] = df['dte'].clip(lower=0.001)

option_cols = [c for c in df.columns if c.startswith('NIFTY') and (c.endswith('CE') or c.endswith('PE'))]

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

def eval_total_variance(masked, truth):
    filled = masked[option_cols].copy()
    ce_cols = [c for c in option_cols if 'CE' in c]
    ce_s = [float(c.replace('NIFTY27JAN26','').replace('CE','')) for c in ce_cols]
    pe_cols = [c for c in option_cols if 'PE' in c]
    pe_s = [float(c.replace('NIFTY27JAN26','').replace('PE','')) for c in pe_cols]

    for ix, row in masked.iterrows():
        dte = row['dte']
        
        # CE
        v_s = [s for j, s in enumerate(ce_s) if pd.notna(row[ce_cols[j]])]
        v_iv = [row[ce_cols[j]] for j, s in enumerate(ce_s) if pd.notna(row[ce_cols[j]])]
        if len(v_s) >= 2:
            # Transform IV to Total Variance
            v_w = [ (iv**2)*dte for iv in v_iv ]
            f = interp1d(v_s, v_w, kind='linear', fill_value='extrapolate')
            for j, c in enumerate(ce_cols):
                if pd.isna(row[c]):
                    w_pred = f(ce_s[j])
                    iv_pred = np.sqrt(max(0.0001, w_pred / dte))
                    filled.at[ix, c] = iv_pred
                    
        # PE
        v_s = [s for j, s in enumerate(pe_s) if pd.notna(row[pe_cols[j]])]
        v_iv = [row[pe_cols[j]] for j, s in enumerate(pe_s) if pd.notna(row[pe_cols[j]])]
        if len(v_s) >= 2:
            v_w = [ (iv**2)*dte for iv in v_iv ]
            f = interp1d(v_s, v_w, kind='linear', fill_value='extrapolate')
            for j, c in enumerate(pe_cols):
                if pd.isna(row[c]):
                    w_pred = f(pe_s[j])
                    iv_pred = np.sqrt(max(0.0001, w_pred / dte))
                    filled.at[ix, c] = iv_pred

    y_t, y_p = [], []
    for ix, c, tv in truth:
        pv = filled.at[ix, c]
        if pd.notna(pv): y_t.append(tv); y_p.append(np.clip(pv, 0.01, 6.0))
    return mean_squared_error(y_t, y_p)

print("Linear Total Variance MSE:", eval_total_variance(masked, truth))
