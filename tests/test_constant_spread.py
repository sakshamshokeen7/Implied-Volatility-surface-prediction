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

def eval_constant_spread(masked, truth):
    filled = masked[option_cols].copy()
    for ix, row in masked.iterrows():
        ce_valid_s = [strikes_ce[j] for j, c in enumerate(ce_cols) if pd.notna(row[c])]
        ce_valid_iv = [row[c] for j, c in enumerate(ce_cols) if pd.notna(row[c])]
        pe_valid_s = [strikes_pe[j] for j, c in enumerate(pe_cols) if pd.notna(row[c])]
        pe_valid_iv = [row[c] for j, c in enumerate(pe_cols) if pd.notna(row[c])]
        
        # Calculate spread CE - PE on overlapping strikes
        overlap_s = []
        spreads = []
        for s in ce_valid_s:
            if s in pe_valid_s:
                overlap_s.append(s)
                c_iv = ce_valid_iv[ce_valid_s.index(s)]
                p_iv = pe_valid_iv[pe_valid_s.index(s)]
                spreads.append(c_iv - p_iv)
                
        def get_spread(k):
            if len(overlap_s) == 0: return 0.0
            if k <= overlap_s[0]: return spreads[0] # Constant left
            if k >= overlap_s[-1]: return spreads[-1] # Constant right
            # Linear interior
            idx = np.searchsorted(overlap_s, k)
            s1, s2 = overlap_s[idx-1], overlap_s[idx]
            sp1, sp2 = spreads[idx-1], spreads[idx]
            return sp1 + (sp2 - sp1) * (k - s1) / (s2 - s1)
            
        f_ce = interp1d(ce_valid_s, ce_valid_iv, kind='linear', fill_value='extrapolate') if len(ce_valid_s) >= 2 else None
        f_pe = interp1d(pe_valid_s, pe_valid_iv, kind='linear', fill_value='extrapolate') if len(pe_valid_s) >= 2 else None
        
        for j, c in enumerate(ce_cols):
            if pd.isna(row[c]):
                s = strikes_ce[j]
                # If s is outside CE known range, but INSIDE PE known range!
                if ce_valid_s and (s < ce_valid_s[0] or s > ce_valid_s[-1]) and pe_valid_s and (pe_valid_s[0] <= s <= pe_valid_s[-1]):
                    p_iv = pe_valid_iv[pe_valid_s.index(s)]
                    filled.at[ix, c] = p_iv + get_spread(s)
                else:
                    if f_ce: filled.at[ix, c] = float(f_ce(s))
                    
        for j, c in enumerate(pe_cols):
            if pd.isna(row[c]):
                s = strikes_pe[j]
                if pe_valid_s and (s < pe_valid_s[0] or s > pe_valid_s[-1]) and ce_valid_s and (ce_valid_s[0] <= s <= ce_valid_s[-1]):
                    c_iv = ce_valid_iv[ce_valid_s.index(s)]
                    filled.at[ix, c] = c_iv - get_spread(s)
                else:
                    if f_pe: filled.at[ix, c] = float(f_pe(s))

    y_t, y_p = [], []
    for ix, c, tv in truth:
        pv = filled.at[ix, c]
        if pd.notna(pv): y_t.append(tv); y_p.append(np.clip(pv, 0.01, 6.0))
    return mean_squared_error(y_t, y_p)

print("Constant Spread Extrapolation MSE:", eval_constant_spread(masked, truth))
