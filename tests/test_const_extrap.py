import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
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

def eval_constant_extrapolate(masked, truth):
    filled = masked.copy()
    for ix, row in masked.iterrows():
        # CE
        ce_cols = [c for c in option_cols if 'CE' in c]
        ce_s = [float(c.replace('NIFTY27JAN26','').replace('CE','')) for c in ce_cols]
        v_s = [s for j, s in enumerate(ce_s) if pd.notna(row[ce_cols[j]])]
        v_iv = [row[ce_cols[j]] for j, s in enumerate(ce_s) if pd.notna(row[ce_cols[j]])]
        if len(v_s) >= 2:
            f = interp1d(v_s, v_iv, kind='linear', bounds_error=False, fill_value=(v_iv[0], v_iv[-1]))
            for j, c in enumerate(ce_cols):
                if pd.isna(row[c]): filled.at[ix, c] = float(f(ce_s[j]))
        
        # PE
        pe_cols = [c for c in option_cols if 'PE' in c]
        pe_s = [float(c.replace('NIFTY27JAN26','').replace('PE','')) for c in pe_cols]
        v_s = [s for j, s in enumerate(pe_s) if pd.notna(row[pe_cols[j]])]
        v_iv = [row[pe_cols[j]] for j, s in enumerate(pe_s) if pd.notna(row[pe_cols[j]])]
        if len(v_s) >= 2:
            f = interp1d(v_s, v_iv, kind='linear', bounds_error=False, fill_value=(v_iv[0], v_iv[-1]))
            for j, c in enumerate(pe_cols):
                if pd.isna(row[c]): filled.at[ix, c] = float(f(pe_s[j]))
                
    y_t, y_p = [], []
    for ix, c, tv in truth:
        pv = filled.at[ix, c]
        if pd.notna(pv): y_t.append(tv); y_p.append(np.clip(pv, 0.01, 6.0))
    return mean_squared_error(y_t, y_p)

print("Constant Extrapolate MSE:", eval_constant_extrapolate(masked, truth))
