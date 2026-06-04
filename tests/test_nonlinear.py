import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

df_original = pd.read_csv('dataset.csv')
df_original = df_original.dropna(subset=['datetime']).reset_index(drop=True)
df = df_original.copy()

df['datetime_dt'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
df['time_of_day'] = df['datetime_dt'].dt.hour * 60 + df['datetime_dt'].dt.minute
expiry = pd.to_datetime('2026-01-27')
df['dte'] = (expiry - df['datetime_dt']).dt.total_seconds() / (24*3600)
df['inv_underlying'] = 1.0 / df['underlying_price']
df['sq_underlying'] = df['underlying_price'] ** 2
df['log_underlying'] = np.log(df['underlying_price'])

scaler = StandardScaler()
df['underlying_scaled'] = scaler.fit_transform(df[['underlying_price']])
df['inv_underlying_scaled'] = scaler.fit_transform(df[['inv_underlying']])
df['sq_underlying_scaled'] = scaler.fit_transform(df[['sq_underlying']])
df['log_underlying_scaled'] = scaler.fit_transform(df[['log_underlying']])
df['time_scaled'] = scaler.fit_transform(df[['time_of_day']])
df['dte_scaled'] = scaler.fit_transform(df[['dte']])

feature_cols = [c for c in df_original.columns if c != 'datetime']
option_cols  = [c for c in feature_cols if c != 'underlying_price']

def validate_imputer(features, name, df_in, mask_frac=0.15, n_trials=3):
    scores = []
    for trial in range(n_trials):
        rng = np.random.RandomState(trial * 101 + 13)
        masked = df_in.copy()
        truth = []
        for c in option_cols:
            obs = df_in.index[df_in[c].notna()].tolist()
            k = max(1, int(len(obs) * mask_frac))
            hide = rng.choice(obs, size=k, replace=False)
            for ix in hide:
                truth.append((ix, c, df_in.at[ix, c]))
                masked.at[ix, c] = np.nan
        imp = IterativeImputer(estimator=Ridge(alpha=0.015), max_iter=30, tol=1e-5, random_state=42)
        filled_matrix = imp.fit_transform(masked[features + option_cols])
        filled = pd.DataFrame(filled_matrix, columns=features + option_cols)[option_cols].clip(lower=0.001)
        y_true, y_pred = [], []
        for ix, c, tv in truth:
            pv = filled.at[ix, c]
            if pd.notna(pv) and pd.notna(tv):
                y_true.append(tv)
                y_pred.append(pv)
        mse = mean_squared_error(y_true, y_pred)
        scores.append(mse)
    mu = np.mean(scores)
    print(f"{name:20s} : Mean MSE = {mu:.10f}")
    return mu

if __name__ == '__main__':
    validate_imputer(['underlying_scaled', 'inv_underlying_scaled', 'time_scaled', 'dte_scaled'], 'Inv', df.copy())
    validate_imputer(['underlying_scaled', 'inv_underlying_scaled', 'sq_underlying_scaled', 'time_scaled', 'dte_scaled'], 'Inv + Sq', df.copy())
    validate_imputer(['underlying_scaled', 'inv_underlying_scaled', 'log_underlying_scaled', 'time_scaled', 'dte_scaled'], 'Inv + Log', df.copy())
    validate_imputer(['underlying_scaled', 'inv_underlying_scaled', 'sq_underlying_scaled', 'log_underlying_scaled', 'time_scaled', 'dte_scaled'], 'All Underlyings', df.copy())
