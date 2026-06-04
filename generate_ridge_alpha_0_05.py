"""
Generate final submission using pure IterativeImputer with tuned Ridge.
Fix ID format to use ||
"""
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
df_original = pd.read_csv('dataset.csv')
df = df_original.copy()

df['datetime_dt'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
df['time_of_day'] = df['datetime_dt'].dt.hour * 60 + df['datetime_dt'].dt.minute
expiry = pd.to_datetime('2026-01-27 15:30')
df['dte'] = (expiry - df['datetime_dt']).dt.total_seconds() / (24*3600)
df['dte'] = df['dte'].clip(lower=0.001)

option_cols = [c for c in df_original.columns if c.startswith('NIFTY')]
base_aux = ['underlying_price', 'time_of_day', 'dte']

print("Running IterativeImputer with Ridge(alpha=0.05)...")
all_cols = base_aux + option_cols
imp = IterativeImputer(estimator=Ridge(alpha=0.05), max_iter=30, tol=1e-5, random_state=42)

filled_matrix = imp.fit_transform(df[all_cols])
filled = pd.DataFrame(filled_matrix, columns=all_cols)[option_cols].clip(lower=0.001)

print("Formatting submission.csv...")
sub_cols = ['id', 'implied_volatility']
sub_data = []
for ix, row in df_original.iterrows():
    if pd.isna(row['datetime']):
        continue
    for c in option_cols:
        if pd.isna(row[c]):
            sub_id = f"{row['datetime']}||{c}"
            sub_data.append((sub_id, filled.at[ix, c]))

sub_df = pd.DataFrame(sub_data, columns=sub_cols)
sub_df.to_csv('submission_ridge_alpha_0_05.csv', index=False)
print(f"Saved submission_ridge_alpha_0_05.csv with {len(sub_df)} rows.")
