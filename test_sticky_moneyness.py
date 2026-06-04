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

def eval_sticky_moneyness(masked, truth):
    filled = masked.copy()
    
    ce_cols = [c for c in option_cols if 'CE' in c]
    ce_s = [float(c.replace('NIFTY27JAN26','').replace('CE','')) for c in ce_cols]
    pe_cols = [c for c in option_cols if 'PE' in c]
    pe_s = [float(c.replace('NIFTY27JAN26','').replace('PE','')) for c in pe_cols]

    # Precompute Moneyness curves for previous row
    prev_ce_f = None
    prev_pe_f = None
    
    for ix, row in masked.iterrows():
        S = row['underlying_price']
        
        # CE
        v_s = [s for j, s in enumerate(ce_s) if pd.notna(row[ce_cols[j]])]
        v_iv = [row[ce_cols[j]] for j, s in enumerate(ce_s) if pd.notna(row[ce_cols[j]])]
        v_m = [s / S for s in v_s]
        
        if len(v_s) >= 2:
            f_ce = interp1d(v_s, v_iv, kind='linear', fill_value='extrapolate')
            f_ce_m = interp1d(v_m, v_iv, kind='linear', fill_value='extrapolate')
            for j, c in enumerate(ce_cols):
                if pd.isna(row[c]):
                    m = ce_s[j] / S
                    pred = None
                    if prev_ce_f is not None:
                        pred = float(prev_ce_f(m))
                    else:
                        pred = float(f_ce(ce_s[j]))
                    filled.at[ix, c] = pred
            prev_ce_f = f_ce_m
        else:
            if prev_ce_f is not None:
                for j, c in enumerate(ce_cols):
                    if pd.isna(row[c]):
                        m = ce_s[j] / S
                        filled.at[ix, c] = float(prev_ce_f(m))

        # PE
        v_s = [s for j, s in enumerate(pe_s) if pd.notna(row[pe_cols[j]])]
        v_iv = [row[pe_cols[j]] for j, s in enumerate(pe_s) if pd.notna(row[pe_cols[j]])]
        v_m = [s / S for s in v_s]
        
        if len(v_s) >= 2:
            f_pe = interp1d(v_s, v_iv, kind='linear', fill_value='extrapolate')
            f_pe_m = interp1d(v_m, v_iv, kind='linear', fill_value='extrapolate')
            for j, c in enumerate(pe_cols):
                if pd.isna(row[c]):
                    m = pe_s[j] / S
                    pred = None
                    if prev_pe_f is not None:
                        pred = float(prev_pe_f(m))
                    else:
                        pred = float(f_pe(pe_s[j]))
                    filled.at[ix, c] = pred
            prev_pe_f = f_pe_m
        else:
            if prev_pe_f is not None:
                for j, c in enumerate(pe_cols):
                    if pd.isna(row[c]):
                        m = pe_s[j] / S
                        filled.at[ix, c] = float(prev_pe_f(m))

    y_t, y_p = [], []
    for ix, c, tv in truth:
        pv = filled.at[ix, c]
        if pd.notna(pv): y_t.append(tv); y_p.append(np.clip(pv, 0.01, 6.0))
    return mean_squared_error(y_t, y_p)

print("Sticky Moneyness MSE:", eval_sticky_moneyness(masked, truth))
