import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

df_original = pd.read_csv('dataset.csv')
df_original = df_original.dropna(subset=['datetime']).reset_index(drop=True)
df = df_original.copy()

feature_cols = [c for c in df_original.columns if c != 'datetime']
option_cols  = [c for c in feature_cols if c != 'underlying_price']

def soft_impute(X_missing, lambda_=0.1, max_iter=100, tol=1e-5):
    # Initialize with column means
    col_means = np.nanmean(X_missing, axis=0)
    X = np.where(np.isnan(X_missing), col_means, X_missing)
    
    for i in range(max_iter):
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        S_thresh = np.maximum(S - lambda_, 0)
        X_new = U @ np.diag(S_thresh) @ Vt
        
        diff = np.linalg.norm(X_new[np.isnan(X_missing)] - X[np.isnan(X_missing)]) / (np.linalg.norm(X[np.isnan(X_missing)]) + 1e-9)
        
        X[np.isnan(X_missing)] = X_new[np.isnan(X_missing)]
        
        if diff < tol:
            break
            
    return X

def validate_softimpute(df_in, lambda_=0.1, mask_frac=0.15, n_trials=3):
    print(f"\n{'-' * 72}")
    print(f"  VALIDATION - SoftImpute(lambda={lambda_})")
    print(f"{'-' * 72}")
    scores = []
    
    # Scale features for SVD so that different magnitudes don't dominate inappropriately
    scaler = StandardScaler()
    X_orig = scaler.fit_transform(df_in[option_cols].values)
    
    for trial in range(n_trials):
        rng = np.random.RandomState(trial * 101 + 13)
        X_missing = X_orig.copy()
        truth = []
        
        # Mask
        for j, c in enumerate(option_cols):
            obs_indices = np.where(~np.isnan(X_orig[:, j]))[0]
            k = max(1, int(len(obs_indices) * mask_frac))
            hide = rng.choice(obs_indices, size=k, replace=False)
            for idx in hide:
                truth.append((idx, j, X_orig[idx, j]))
                X_missing[idx, j] = np.nan
                
        # Impute
        X_filled = soft_impute(X_missing, lambda_=lambda_)
        
        # Unscale for MSE calculation
        X_filled_unscaled = scaler.inverse_transform(X_filled)
        X_orig_unscaled = scaler.inverse_transform(X_orig)
        
        y_true, y_pred = [], []
        for idx, j, tv_scaled in truth:
            y_true.append(X_orig_unscaled[idx, j])
            y_pred.append(X_filled_unscaled[idx, j])
            
        mse = mean_squared_error(y_true, np.maximum(y_pred, 0.001))
        scores.append(mse)
        print(f"  Trial {trial+1}/{n_trials}  MSE = {mse:.10f}")
        
    mu = np.mean(scores)
    print(f"  Mean Validation MSE = {mu:.10f}")
    return mu

if __name__ == '__main__':
    for lam in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0]:
        validate_softimpute(df.copy(), lambda_=lam)
