import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset.csv').dropna(subset=['datetime']).reset_index(drop=True).iloc[:2000]
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

def eval_spline(masked, truth):
    filled = masked.copy()
    for ix, row in masked.iterrows():
        for opt_type in ['CE', 'PE']:
            cols = [c for c in option_cols if opt_type in c]
            strikes = [float(c.replace('NIFTY27JAN26','').replace(opt_type,'')) for c in cols]
            
            v_s = [s for j, s in enumerate(strikes) if pd.notna(row[cols[j]])]
            v_iv = [row[cols[j]] for j, s in enumerate(strikes) if pd.notna(row[cols[j]])]
            
            if len(v_s) >= 4:
                try:
                    f = UnivariateSpline(v_s, v_iv, k=3, s=0.001) # Small smoothing
                    for j, c in enumerate(cols):
                        if pd.isna(row[c]):
                            filled.at[ix, c] = float(f(strikes[j]))
                except Exception:
                    pass
    
    y_t, y_p = [], []
    for ix, c, tv in truth:
        pv = filled.at[ix, c]
        if pd.notna(pv): y_t.append(tv); y_p.append(np.clip(pv, 0.01, 6.0))
    return mean_squared_error(y_t, y_p)

print("Spline MSE:", eval_spline(masked, truth))
