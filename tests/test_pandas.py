import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset.csv').dropna(subset=['datetime']).reset_index(drop=True).iloc[:2000]

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

def eval_pandas_interp(masked, truth):
    filled = masked.copy()
    filled[ce_cols] = filled[ce_cols].interpolate(method='linear', axis=1, limit_direction='both')
    filled[pe_cols] = filled[pe_cols].interpolate(method='linear', axis=1, limit_direction='both')
    y_t, y_p = [], []
    for ix, c, tv in truth:
        pv = filled.at[ix, c]
        if pd.notna(pv): y_t.append(tv); y_p.append(pv)
    return mean_squared_error(y_t, y_p)

def eval_pandas_index(masked, truth):
    filled = masked.copy()
    # Need to set columns as numeric for index interp
    filled_ce = filled[ce_cols].copy()
    filled_ce.columns = strikes_ce
    filled_ce = filled_ce.interpolate(method='index', axis=1, limit_direction='both')
    filled_ce.columns = ce_cols
    filled[ce_cols] = filled_ce
    
    filled_pe = filled[pe_cols].copy()
    filled_pe.columns = strikes_pe
    filled_pe = filled_pe.interpolate(method='index', axis=1, limit_direction='both')
    filled_pe.columns = pe_cols
    filled[pe_cols] = filled_pe

    y_t, y_p = [], []
    for ix, c, tv in truth:
        pv = filled.at[ix, c]
        if pd.notna(pv): y_t.append(tv); y_p.append(pv)
    return mean_squared_error(y_t, y_p)

print("Pandas Linear (Equal Space, Flat Wings) MSE:", eval_pandas_interp(masked, truth))
print("Pandas Index (Strike Space, Flat Wings) MSE:", eval_pandas_index(masked, truth))
