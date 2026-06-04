import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset.csv').dropna(subset=['datetime']).reset_index(drop=True).iloc[:2000]

option_cols = [c for c in df.columns if c.startswith('NIFTY') and (c.endswith('CE') or c.endswith('PE'))]
ce_cols = sorted([c for c in option_cols if c.endswith('CE')])
pe_cols = sorted([c for c in option_cols if c.endswith('PE')])
strikes_ce = np.array([float(c.replace('NIFTY27JAN26','').replace('CE','')) for c in ce_cols])
strikes_pe = np.array([float(c.replace('NIFTY27JAN26','').replace('PE','')) for c in pe_cols])

rng = np.random.RandomState(42)
masked = df.copy()
truth = []
for c in option_cols:
    obs = df.index[df[c].notna()].tolist()
    k = max(1, int(len(obs) * 0.15))
    hide = rng.choice(obs, size=k, replace=False)
    for ix in hide:
        truth.append((ix, c, df.at[ix, c]))
        masked.at[ix, c] = np.nan

def eval_pandas(method, order=None):
    filled = masked.copy()
    
    # CE
    filled_ce = filled[ce_cols].copy()
    filled_ce.columns = strikes_ce
    if order:
        filled_ce = filled_ce.interpolate(method=method, order=order, axis=1, limit_direction='both')
    else:
        filled_ce = filled_ce.interpolate(method=method, axis=1, limit_direction='both')
    filled_ce.columns = ce_cols
    filled[ce_cols] = filled_ce
    
    # PE
    filled_pe = filled[pe_cols].copy()
    filled_pe.columns = strikes_pe
    if order:
        filled_pe = filled_pe.interpolate(method=method, order=order, axis=1, limit_direction='both')
    else:
        filled_pe = filled_pe.interpolate(method=method, axis=1, limit_direction='both')
    filled_pe.columns = pe_cols
    filled[pe_cols] = filled_pe

    y_t, y_p = [], []
    for ix, c, tv in truth:
        pv = filled.at[ix, c]
        if pd.notna(pv): y_t.append(tv); y_p.append(pv)
    return mean_squared_error(y_t, y_p)

print("Pandas index (Linear flat wings):", eval_pandas('index'))
print("Pandas slinear:", eval_pandas('slinear'))
print("Pandas quadratic:", eval_pandas('quadratic'))
print("Pandas cubic:", eval_pandas('cubic'))
print("Pandas polynomial 2:", eval_pandas('polynomial', 2))
print("Pandas polynomial 3:", eval_pandas('polynomial', 3))
print("Pandas spline 2:", eval_pandas('spline', 2))
print("Pandas spline 3:", eval_pandas('spline', 3))
