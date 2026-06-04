import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset.csv').dropna(subset=['datetime']).reset_index(drop=True).iloc[:2000]
df['datetime_dt'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
expiry = pd.to_datetime('2026-01-27 15:30')
df['dte'] = (expiry - df['datetime_dt']).dt.total_seconds() / (24*3600)
df['dte'] = df['dte'].clip(lower=0.001)

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

def eval_delta_interp(masked, truth):
    filled = masked.copy()
    
    ce_cols = [c for c in option_cols if 'CE' in c]
    ce_s = [float(c.replace('NIFTY27JAN26','').replace('CE','')) for c in ce_cols]
    pe_cols = [c for c in option_cols if 'PE' in c]
    pe_s = [float(c.replace('NIFTY27JAN26','').replace('PE','')) for c in pe_cols]

    for ix, row in masked.iterrows():
        S = row['underlying_price']
        dte = row['dte']
        
        # Approximate ATM IV
        atm_iv = 0.15 # Fallback
        v_s_all = []
        v_iv_all = []
        for j, c in enumerate(ce_cols):
            if pd.notna(row[c]):
                v_s_all.append(ce_s[j])
                v_iv_all.append(row[c])
        if len(v_s_all) > 0:
            idx = np.argmin(np.abs(np.array(v_s_all) - S))
            atm_iv = v_iv_all[idx]
            
        # CE
        v_s = [s for j, s in enumerate(ce_s) if pd.notna(row[ce_cols[j]])]
        v_iv = [row[ce_cols[j]] for j, s in enumerate(ce_s) if pd.notna(row[ce_cols[j]])]
        if len(v_s) >= 2:
            v_d1 = np.log(np.array(v_s) / S) / (atm_iv * np.sqrt(dte))
            f_ce = interp1d(v_d1, v_iv, kind='linear', fill_value='extrapolate')
            for j, c in enumerate(ce_cols):
                if pd.isna(row[c]):
                    d1 = np.log(ce_s[j] / S) / (atm_iv * np.sqrt(dte))
                    filled.at[ix, c] = float(f_ce(d1))
        elif len(v_s) >= 1:
            for j, c in enumerate(ce_cols):
                if pd.isna(row[c]): filled.at[ix, c] = v_iv[0]
                
        # PE
        v_s = [s for j, s in enumerate(pe_s) if pd.notna(row[pe_cols[j]])]
        v_iv = [row[pe_cols[j]] for j, s in enumerate(pe_s) if pd.notna(row[pe_cols[j]])]
        if len(v_s) >= 2:
            v_d1 = np.log(np.array(v_s) / S) / (atm_iv * np.sqrt(dte))
            f_pe = interp1d(v_d1, v_iv, kind='linear', fill_value='extrapolate')
            for j, c in enumerate(pe_cols):
                if pd.isna(row[c]):
                    d1 = np.log(pe_s[j] / S) / (atm_iv * np.sqrt(dte))
                    filled.at[ix, c] = float(f_pe(d1))
        elif len(v_s) >= 1:
            for j, c in enumerate(pe_cols):
                if pd.isna(row[c]): filled.at[ix, c] = v_iv[0]

    y_t, y_p = [], []
    for ix, c, tv in truth:
        pv = filled.at[ix, c]
        if pd.notna(pv): y_t.append(tv); y_p.append(np.clip(pv, 0.01, 6.0))
    return mean_squared_error(y_t, y_p)

print("Delta Interpolation MSE:", eval_delta_interp(masked, truth))
