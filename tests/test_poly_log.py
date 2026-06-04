import pandas as pd
import numpy as np
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

def eval_poly_log_m(masked, truth, deg=2):
    filled = masked.copy()
    
    ce_cols = [c for c in option_cols if 'CE' in c]
    ce_s = [float(c.replace('NIFTY27JAN26','').replace('CE','')) for c in ce_cols]
    pe_cols = [c for c in option_cols if 'PE' in c]
    pe_s = [float(c.replace('NIFTY27JAN26','').replace('PE','')) for c in pe_cols]
    
    for ix, row in masked.iterrows():
        S = row['underlying_price']
        
        # CE
        v_s = [s for j, s in enumerate(ce_s) if pd.notna(row[ce_cols[j]])]
        v_iv = [row[ce_cols[j]] for j, s in enumerate(ce_s) if pd.notna(row[ce_cols[j]])]
        if len(v_s) > deg:
            v_k = np.log(np.array(v_s) / S)
            poly = np.poly1d(np.polyfit(v_k, v_iv, deg))
            for j, c in enumerate(ce_cols):
                if pd.isna(row[c]):
                    k = np.log(ce_s[j] / S)
                    filled.at[ix, c] = float(poly(k))
        elif len(v_s) >= 1:
            for j, c in enumerate(ce_cols):
                if pd.isna(row[c]): filled.at[ix, c] = v_iv[0]
                
        # PE
        v_s = [s for j, s in enumerate(pe_s) if pd.notna(row[pe_cols[j]])]
        v_iv = [row[pe_cols[j]] for j, s in enumerate(pe_s) if pd.notna(row[pe_cols[j]])]
        if len(v_s) > deg:
            v_k = np.log(np.array(v_s) / S)
            poly = np.poly1d(np.polyfit(v_k, v_iv, deg))
            for j, c in enumerate(pe_cols):
                if pd.isna(row[c]):
                    k = np.log(pe_s[j] / S)
                    filled.at[ix, c] = float(poly(k))
        elif len(v_s) >= 1:
            for j, c in enumerate(pe_cols):
                if pd.isna(row[c]): filled.at[ix, c] = v_iv[0]

    y_t, y_p = [], []
    for ix, c, tv in truth:
        pv = filled.at[ix, c]
        if pd.notna(pv): y_t.append(tv); y_p.append(np.clip(pv, 0.01, 6.0))
    return mean_squared_error(y_t, y_p)

print("Quadratic Log-Moneyness MSE:", eval_poly_log_m(masked, truth, 2))
print("Cubic Log-Moneyness MSE:", eval_poly_log_m(masked, truth, 3))
