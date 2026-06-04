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

def eval_constrained(masked, truth):
    filled = masked[option_cols].copy()
    for ix, row in masked.iterrows():
        # CE
        v_s = [strikes_ce[j] for j, c in enumerate(ce_cols) if pd.notna(row[c])]
        v_iv = [row[c] for j, c in enumerate(ce_cols) if pd.notna(row[c])]
        if len(v_s) >= 2:
            f_interp = interp1d(v_s, v_iv, kind='linear')
            for j, c in enumerate(ce_cols):
                if pd.isna(row[c]): 
                    st = strikes_ce[j]
                    if st >= v_s[0] and st <= v_s[-1]:
                        filled.at[ix, c] = float(f_interp(st))
                    elif st < v_s[0]:
                        # Left wing: force slope <= 0 (IV should drop as strike increases)
                        raw_slope = (v_iv[1] - v_iv[0]) / (v_s[1] - v_s[0])
                        slope = min(0.0, raw_slope)
                        filled.at[ix, c] = v_iv[0] + slope * (st - v_s[0])
                    elif st > v_s[-1]:
                        # Right wing: force slope >= 0 (IV should rise as strike increases)
                        raw_slope = (v_iv[-1] - v_iv[-2]) / (v_s[-1] - v_s[-2])
                        slope = max(0.0, raw_slope)
                        filled.at[ix, c] = v_iv[-1] + slope * (st - v_s[-1])
                        
        # PE (Do exactly the same, PE smile is similar shape)
        v_s_pe = [strikes_pe[j] for j, c in enumerate(pe_cols) if pd.notna(row[c])]
        v_iv_pe = [row[c] for j, c in enumerate(pe_cols) if pd.notna(row[c])]
        if len(v_s_pe) >= 2:
            f_interp = interp1d(v_s_pe, v_iv_pe, kind='linear')
            for j, c in enumerate(pe_cols):
                if pd.isna(row[c]): 
                    st = strikes_pe[j]
                    if st >= v_s_pe[0] and st <= v_s_pe[-1]:
                        filled.at[ix, c] = float(f_interp(st))
                    elif st < v_s_pe[0]:
                        raw_slope = (v_iv_pe[1] - v_iv_pe[0]) / (v_s_pe[1] - v_s_pe[0])
                        slope = min(0.0, raw_slope)
                        filled.at[ix, c] = v_iv_pe[0] + slope * (st - v_s_pe[0])
                    elif st > v_s_pe[-1]:
                        raw_slope = (v_iv_pe[-1] - v_iv_pe[-2]) / (v_s_pe[-1] - v_s_pe[-2])
                        slope = max(0.0, raw_slope)
                        filled.at[ix, c] = v_iv_pe[-1] + slope * (st - v_s_pe[-1])

    y_t, y_p = [], []
    for ix, c, tv in truth:
        pv = filled.at[ix, c]
        if pd.notna(pv): y_t.append(tv); y_p.append(np.clip(pv, 0.01, 6.0))
    return mean_squared_error(y_t, y_p)

print("Constrained Extrapolation MSE:", eval_constrained(masked, truth))
