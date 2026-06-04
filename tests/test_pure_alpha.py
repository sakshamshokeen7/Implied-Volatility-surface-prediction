"""
Tune alpha for pure IterativeImputer with Ridge (no lag).
"""
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge
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

def validate(alpha, n_trials=3):
    scores = []
    for trial in range(n_trials):
        rng = np.random.RandomState(trial * 101 + 13)
        masked = df.copy()
        truth = []
        for c in option_cols:
            obs = df.index[df[c].notna()].tolist()
            k = max(1, int(len(obs) * 0.15))
            hide = rng.choice(obs, size=k, replace=False)
            for ix in hide:
                truth.append((ix, c, df.at[ix, c]))
                masked.at[ix, c] = np.nan
        
        all_cols = base_aux + option_cols
        imp = IterativeImputer(estimator=Ridge(alpha=alpha), max_iter=20, tol=1e-5, random_state=42)
        
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
    
    mu = np.mean(scores)
    print(f"Alpha {alpha:8f} : {mu:.10f}")
    return mu

print("Tuning pure Ridge baseline...")
for a in [0.001, 0.005, 0.01, 0.015, 0.05, 0.1, 1.0]:
    validate(a)
