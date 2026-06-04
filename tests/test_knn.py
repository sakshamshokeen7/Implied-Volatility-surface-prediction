import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
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

scaler = StandardScaler()
df['underlying_scaled'] = scaler.fit_transform(df[['underlying_price']])
df['time_scaled'] = scaler.fit_transform(df[['time_of_day']])
df['dte_scaled'] = scaler.fit_transform(df[['dte']])

feature_cols = [c for c in df_original.columns if c != 'datetime']
option_cols  = [c for c in feature_cols if c != 'underlying_price']
all_impute_cols = ['underlying_scaled', 'time_scaled', 'dte_scaled'] + option_cols

def validate_knnimputer(k, weights, df_in, mask_frac=0.15, n_trials=3):
    scores = []
    for trial in range(n_trials):
        rng = np.random.RandomState(trial * 101 + 13)
        masked = df_in.copy()
        truth = []
        for c in option_cols:
            obs = df_in.index[df_in[c].notna()].tolist()
            k_mask = max(1, int(len(obs) * mask_frac))
            hide = rng.choice(obs, size=k_mask, replace=False)
            for ix in hide:
                truth.append((ix, c, df_in.at[ix, c]))
                masked.at[ix, c] = np.nan
        imp = KNNImputer(n_neighbors=k, weights=weights)
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
    mu = np.mean(scores)
    print(f"KNN(k={k}, {weights}) : Mean MSE = {mu:.10f}")
    return mu

if __name__ == '__main__':
    for k in [3, 5, 7, 10, 15]:
        validate_knnimputer(k, 'distance', df.copy())
