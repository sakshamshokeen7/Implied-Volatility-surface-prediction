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

def eval_poly(masked, truth, deg=2):
    filled = masked[option_cols].copy()
    for ix, row in masked.iterrows():
        # CE
        v_s = [strikes_ce[j] for j, c in enumerate(ce_cols) if pd.notna(row[c])]
        v_iv = [row[c] for j, c in enumerate(ce_cols) if pd.notna(row[c])]
        if len(v_s) >= deg + 1:
            coefs = np.polyfit(v_s, v_iv, deg)
            p = np.poly1d(coefs)
            for j, c in enumerate(ce_cols):
                if pd.isna(row[c]): filled.at[ix, c] = p(strikes_ce[j])
                
        # PE
        v_s = [strikes_pe[j] for j, c in enumerate(pe_cols) if pd.notna(row[c])]
        v_iv = [row[c] for j, c in enumerate(pe_cols) if pd.notna(row[c])]
        if len(v_s) >= deg + 1:
            coefs = np.polyfit(v_s, v_iv, deg)
            p = np.poly1d(coefs)
            for j, c in enumerate(pe_cols):
                if pd.isna(row[c]): filled.at[ix, c] = p(strikes_pe[j])

    y_t, y_p = [], []
    for ix, c, tv in truth:
        pv = filled.at[ix, c]
        if pd.notna(pv): y_t.append(tv); y_p.append(np.clip(pv, 0.01, 6.0))
    return mean_squared_error(y_t, y_p)

print("Poly Deg 2 MSE:", eval_poly(masked, truth, 2))
print("Poly Deg 3 MSE:", eval_poly(masked, truth, 3))
print("Poly Deg 4 MSE:", eval_poly(masked, truth, 4))
