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

def eval_smoothed_slope(masked, truth):
    filled = masked.copy()
    
    ce_cols = [c for c in option_cols if 'CE' in c]
    ce_s = [float(c.replace('NIFTY27JAN26','').replace('CE','')) for c in ce_cols]
    pe_cols = [c for c in option_cols if 'PE' in c]
    pe_s = [float(c.replace('NIFTY27JAN26','').replace('PE','')) for c in pe_cols]

    ce_left_slope_hist = []
    ce_right_slope_hist = []
    pe_left_slope_hist = []
    pe_right_slope_hist = []
    
    for ix, row in masked.iterrows():
        # CE
        v_s = [s for j, s in enumerate(ce_s) if pd.notna(row[ce_cols[j]])]
        v_iv = [row[ce_cols[j]] for j, s in enumerate(ce_s) if pd.notna(row[ce_cols[j]])]
        
        if len(v_s) >= 2:
            left_s = (v_iv[1] - v_iv[0]) / (v_s[1] - v_s[0])
            right_s = (v_iv[-1] - v_iv[-2]) / (v_s[-1] - v_s[-2])
            
            ce_left_slope_hist.append(left_s)
            ce_right_slope_hist.append(right_s)
            
            s_left = np.mean(ce_left_slope_hist[-10:])
            s_right = np.mean(ce_right_slope_hist[-10:])
            
            f_in = interp1d(v_s, v_iv, kind='linear')
            for j, c in enumerate(ce_cols):
                if pd.isna(row[c]):
                    s = ce_s[j]
                    if v_s[0] <= s <= v_s[-1]:
                        filled.at[ix, c] = float(f_in(s))
                    elif s < v_s[0]:
                        filled.at[ix, c] = v_iv[0] + s_left * (s - v_s[0])
                    else:
                        filled.at[ix, c] = v_iv[-1] + s_right * (s - v_s[-1])
                        
        # PE
        v_s = [s for j, s in enumerate(pe_s) if pd.notna(row[pe_cols[j]])]
        v_iv = [row[pe_cols[j]] for j, s in enumerate(pe_s) if pd.notna(row[pe_cols[j]])]
        
        if len(v_s) >= 2:
            left_s = (v_iv[1] - v_iv[0]) / (v_s[1] - v_s[0])
            right_s = (v_iv[-1] - v_iv[-2]) / (v_s[-1] - v_s[-2])
            
            pe_left_slope_hist.append(left_s)
            pe_right_slope_hist.append(right_s)
            
            s_left = np.mean(pe_left_slope_hist[-10:])
            s_right = np.mean(pe_right_slope_hist[-10:])
            
            f_in = interp1d(v_s, v_iv, kind='linear')
            for j, c in enumerate(pe_cols):
                if pd.isna(row[c]):
                    s = pe_s[j]
                    if v_s[0] <= s <= v_s[-1]:
                        filled.at[ix, c] = float(f_in(s))
                    elif s < v_s[0]:
                        filled.at[ix, c] = v_iv[0] + s_left * (s - v_s[0])
                    else:
                        filled.at[ix, c] = v_iv[-1] + s_right * (s - v_s[-1])

    y_t, y_p = [], []
    for ix, c, tv in truth:
        pv = filled.at[ix, c]
        if pd.notna(pv): y_t.append(tv); y_p.append(np.clip(pv, 0.01, 6.0))
    return mean_squared_error(y_t, y_p)

print("Smoothed Slope Extrapolate MSE:", eval_smoothed_slope(masked, truth))
