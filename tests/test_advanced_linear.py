"""
Test Lasso, ElasticNet, and Fine-Tuned Ridge for IterativeImputer
"""
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
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

def validate(estimator, name, n_trials=3):
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
        imp = IterativeImputer(estimator=estimator, max_iter=30, tol=1e-4, random_state=42)
        
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
    print(f"{name:20s} : {mu:.10f}")
    return mu

print("Evaluating Advanced Linear Estimators...")
validate(Ridge(alpha=0.03), "Ridge (alpha=0.03)")
validate(Ridge(alpha=0.04), "Ridge (alpha=0.04)")
validate(Ridge(alpha=0.05), "Ridge (alpha=0.05)")
validate(Ridge(alpha=0.06), "Ridge (alpha=0.06)")
validate(Ridge(alpha=0.07), "Ridge (alpha=0.07)")
validate(Lasso(alpha=0.0001), "Lasso (alpha=1e-4)")
validate(Lasso(alpha=0.001), "Lasso (alpha=1e-3)")
validate(ElasticNet(alpha=0.001, l1_ratio=0.5), "ElasticNet (a=1e-3)")
