import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
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

def eval_loo_boosting(masked, truth):
    # 1. Collect LOO residuals from training (masked) data
    X_train = []
    y_train = []
    
    for ix, row in masked.iterrows():
        S = row['underlying_price']
        
        # CE
        v_s = [strikes_ce[j] for j, c in enumerate(ce_cols) if pd.notna(row[c])]
        v_iv = [row[c] for j, c in enumerate(ce_cols) if pd.notna(row[c])]
        if len(v_s) >= 3:
            for k in range(len(v_s)):
                loo_s = v_s[:k] + v_s[k+1:]
                loo_iv = v_iv[:k] + v_iv[k+1:]
                if len(loo_s) >= 2:
                    f = interp1d(loo_s, loo_iv, kind='linear', fill_value='extrapolate')
                    pred = float(f(v_s[k]))
                    res = v_iv[k] - pred
                    X_train.append([v_s[k]/S, 1])
                    y_train.append(res)
                    
        # PE
        v_s = [strikes_pe[j] for j, c in enumerate(pe_cols) if pd.notna(row[c])]
        v_iv = [row[c] for j, c in enumerate(pe_cols) if pd.notna(row[c])]
        if len(v_s) >= 3:
            for k in range(len(v_s)):
                loo_s = v_s[:k] + v_s[k+1:]
                loo_iv = v_iv[:k] + v_iv[k+1:]
                if len(loo_s) >= 2:
                    f = interp1d(loo_s, loo_iv, kind='linear', fill_value='extrapolate')
                    pred = float(f(v_s[k]))
                    res = v_iv[k] - pred
                    X_train.append([v_s[k]/S, 0])
                    y_train.append(res)
                    
    # 2. Train HGBR
    model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 3. Predict missing values
    filled = masked[option_cols].copy()
    for ix, row in masked.iterrows():
        S = row['underlying_price']
        # CE
        v_s = [strikes_ce[j] for j, c in enumerate(ce_cols) if pd.notna(row[c])]
        v_iv = [row[c] for j, c in enumerate(ce_cols) if pd.notna(row[c])]
        if len(v_s) >= 2:
            f = interp1d(v_s, v_iv, kind='linear', fill_value='extrapolate')
            for j, c in enumerate(ce_cols):
                if pd.isna(row[c]):
                    pred_lin = float(f(strikes_ce[j]))
                    pred_res = model.predict([[strikes_ce[j]/S, 1]])[0]
                    filled.at[ix, c] = pred_lin + pred_res
        # PE
        v_s = [strikes_pe[j] for j, c in enumerate(pe_cols) if pd.notna(row[c])]
        v_iv = [row[c] for j, c in enumerate(pe_cols) if pd.notna(row[c])]
        if len(v_s) >= 2:
            f = interp1d(v_s, v_iv, kind='linear', fill_value='extrapolate')
            for j, c in enumerate(pe_cols):
                if pd.isna(row[c]):
                    pred_lin = float(f(strikes_pe[j]))
                    pred_res = model.predict([[strikes_pe[j]/S, 0]])[0]
                    filled.at[ix, c] = pred_lin + pred_res

    y_t, y_p = [], []
    for ix, c, tv in truth:
        pv = filled.at[ix, c]
        if pd.notna(pv): y_t.append(tv); y_p.append(np.clip(pv, 0.01, 6.0))
    return mean_squared_error(y_t, y_p)

print("LOO Residual Boosting MSE:", eval_loo_boosting(masked, truth))
