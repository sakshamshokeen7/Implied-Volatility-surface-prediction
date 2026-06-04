import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge, BayesianRidge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# Load dataset
df_original = pd.read_csv('dataset.csv')
df_original = df_original.dropna(subset=['datetime']).reset_index(drop=True)
df = df_original.copy()

# Feature Engineering
df['datetime_dt'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
df['time_of_day'] = df['datetime_dt'].dt.hour * 60 + df['datetime_dt'].dt.minute
expiry = pd.to_datetime('2026-01-27')
df['dte'] = (expiry - df['datetime_dt']).dt.total_seconds() / (24*3600)

scaler = StandardScaler()
df['underlying_scaled'] = scaler.fit_transform(df[['underlying_price']])
df['time_scaled'] = scaler.fit_transform(df[['time_of_day']])
df['dte_scaled'] = scaler.fit_transform(df[['dte']])

feature_cols = [c for c in df_original.columns if c != 'datetime']
option_cols  = [c for c in feature_cols if c != 'underlying_price']
all_impute_cols = ['underlying_scaled', 'time_scaled', 'dte_scaled'] + option_cols

def validate_imputer(estimator, name, df_in, mask_frac=0.15, n_trials=3):
    print(f"\n{'-' * 72}")
    print(f"  VALIDATION - {name}")
    print(f"{'-' * 72}")

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

        imp = IterativeImputer(estimator=estimator, max_iter=20, tol=1e-4, random_state=42)
        filled_matrix = imp.fit_transform(masked[all_impute_cols])
        filled = pd.DataFrame(filled_matrix, columns=all_impute_cols)[option_cols].clip(lower=0.001)

        y_true, y_pred = [], []
        for ix, c, tv in truth:
            pv = filled.at[ix, c]
            if pd.notna(pv) and pd.notna(tv):
                y_true.append(tv)
                y_pred.append(pv)

        mse = mean_squared_error(y_true, y_pred)
        scores.append(mse)
        print(f"  Trial {trial+1}/{n_trials}  MSE = {mse:.10f}")

    mu = np.mean(scores)
    print(f"  Mean Validation MSE = {mu:.10f}")
    return mu

if __name__ == '__main__':
    models = {
        'Ridge(alpha=0.02)': Ridge(alpha=0.02),
        'Ridge(alpha=0.001)': Ridge(alpha=0.001),
        'BayesianRidge()': BayesianRidge(),
        'ElasticNet(alpha=0.001)': ElasticNet(alpha=0.001)
    }

    for name, est in models.items():
        validate_imputer(est, name, df.copy())
