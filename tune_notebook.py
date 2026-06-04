"""
Fast tuning script using dumped OOF data.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("Loading cached data...")
try:
    oof_df = pd.read_csv('oof_data.csv')
    base_preds = pd.read_csv('base_preds.csv')
except FileNotFoundError:
    print("Files not found. Run dump_features.py first.")
    exit(1)

features = ['time_of_day', 'dte', 'moneyness', 'is_ce', 'S', 'inv_S', 'sq_S', 'log_S']
X = oof_df[features]
y = oof_df['residual']

print(f"Loaded {len(X)} rows for training.")

def evaluate_model(params):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = HistGradientBoostingRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        scores.append(mean_squared_error(y_val, preds))
    return np.mean(scores)

print("\n--- Tuning HistGradientBoostingRegressor ---")

baseline_params = {'max_iter': 300, 'learning_rate': 0.03, 'max_leaf_nodes': 31}
base_score = evaluate_model(baseline_params)
print(f"Baseline (notebook.py): {base_score:.10f}")

grid = [
    {'max_iter': 100, 'learning_rate': 0.1, 'max_leaf_nodes': 31},
    {'max_iter': 500, 'learning_rate': 0.03, 'max_leaf_nodes': 31},
    {'max_iter': 300, 'learning_rate': 0.03, 'max_leaf_nodes': 15},
    {'max_iter': 300, 'learning_rate': 0.03, 'max_leaf_nodes': 63},
    {'max_iter': 500, 'learning_rate': 0.01, 'max_leaf_nodes': 31},
    {'max_iter': 200, 'learning_rate': 0.05, 'max_leaf_nodes': 31},
    {'max_iter': 300, 'learning_rate': 0.03, 'max_leaf_nodes': 31, 'l2_regularization': 0.1},
    {'max_iter': 300, 'learning_rate': 0.03, 'max_leaf_nodes': 31, 'l2_regularization': 1.0},
]

for p in grid:
    score = evaluate_model(p)
    print(f"Params: {p} -> Score: {score:.10f}")

print("\nDone!")
