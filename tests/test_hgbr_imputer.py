"""
Test IterativeImputer with HistGradientBoostingRegressor (non-linear).
"""
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

df_original = pd.read_csv('dataset.csv')
df = df_original.dropna(subset=['datetime']).reset_index(drop=True)

df['datetime_dt'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
df['time_of_day'] = df['datetime_dt'].dt.hour * 60 + df['datetime_dt'].dt.minute
expiry = pd.to_datetime('2026-01-27 15:30')
df['dte'] = (expiry - df['datetime_dt']).dt.total_seconds() / (24*3600)
df['dte'] = df['dte'].clip(lower=0.001)

option_cols = [c for c in df_original.columns if c.startswith('NIFTY')]
base_aux = ['underlying_price', 'time_of_day', 'dte']

def validate(df_in, n_trials=2):
    scores = []
    for trial in range(n_trials):
        rng = np.random.RandomState(trial * 101 + 13)
        masked = df_in.copy()
        truth = []
        for c in option_cols:
            obs = df_in.index[df_in[c].notna()].tolist()
            k = max(1, int(len(obs) * 0.15))
            hide = rng.choice(obs, size=k, replace=False)
            for ix in hide:
                truth.append((ix, c, df_in.at[ix, c]))
                masked.at[ix, c] = np.nan
        
        all_cols = base_aux + option_cols
        est = HistGradientBoostingRegressor(max_iter=50, random_state=42)
        imp = IterativeImputer(estimator=est, max_iter=15, tol=1e-5, random_state=42)
        
        filled_matrix = imp.fit_transform(masked[all_cols])
        filled = pd.DataFrame(filled_matrix, columns=all_cols)[option_cols].clip(lower=0.001)
        
        y_true, y_pred = [], []
        for ix, c, tv in truth:
            pv = filled.at[ix, c]
            if pd.notna(pv) and pd.notna(tv):
                y_true.append(tv)
                y_pred.append(pv)
        mse = mean_squared_error(y_true, y_pred)
        scores.append(mse)
        print(f"Trial {trial} MSE: {mse:.10f}")
    
    mu = np.mean(scores)
    print(f"HGBR Imputer Mean CV : {mu:.10f}")
    return mu

print("Testing IterativeImputer(HistGradientBoostingRegressor)...")
validate(df)
