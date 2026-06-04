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

def eval_transform(masked, truth, transform_func, inv_transform_func):
    filled = masked[option_cols].copy()
    for ix, row in masked.iterrows():
        # CE
        v_s = [strikes_ce[j] for j, c in enumerate(ce_cols) if pd.notna(row[c])]
        v_iv = [row[c] for j, c in enumerate(ce_cols) if pd.notna(row[c])]
        if len(v_s) >= 2:
            t_iv = transform_func(np.array(v_iv))
            f = interp1d(v_s, t_iv, kind='linear', fill_value='extrapolate')
            for j, c in enumerate(ce_cols):
                if pd.isna(row[c]):
                    pred_t = float(f(strikes_ce[j]))
                    filled.at[ix, c] = inv_transform_func(pred_t)
                    
        # PE
        v_s = [strikes_pe[j] for j, c in enumerate(pe_cols) if pd.notna(row[c])]
        v_iv = [row[c] for j, c in enumerate(pe_cols) if pd.notna(row[c])]
        if len(v_s) >= 2:
            t_iv = transform_func(np.array(v_iv))
            f = interp1d(v_s, t_iv, kind='linear', fill_value='extrapolate')
            for j, c in enumerate(pe_cols):
                if pd.isna(row[c]):
                    pred_t = float(f(strikes_pe[j]))
                    filled.at[ix, c] = inv_transform_func(pred_t)

    y_t, y_p = [], []
    for ix, c, tv in truth:
        pv = filled.at[ix, c]
        if pd.notna(pv): y_t.append(tv); y_p.append(np.clip(pv, 0.01, 6.0))
    return mean_squared_error(y_t, y_p)

print("Linear log(IV) MSE:", eval_transform(masked, truth, lambda x: np.log(x), lambda x: np.exp(x)))
print("Linear 1/IV MSE:", eval_transform(masked, truth, lambda x: 1.0/x, lambda x: 1.0/x))
print("Linear sqrt(IV) MSE:", eval_transform(masked, truth, lambda x: np.sqrt(x), lambda x: x**2))
