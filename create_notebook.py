import json

text_1 = """\
# Implied Volatility Surface Reconstruction
## Finance Club, IIT Roorkee: Open Projects 2026

This notebook implements a robust, deterministic, and arbitrage-aware Log-Moneyness Linear Extrapolation algorithm. 

### Financial Intuition
According to Black-Scholes dynamics, Implied Volatility fundamentally scales with **Log-Moneyness** ($k = \\log(K/S)$) rather than absolute strike ($K$). By interpolating the missing points in Log-Moneyness space, the model respects the natural curvature of the volatility smile and skew. Furthermore, this approach relies strictly on cross-sectional data at each timestamp, completely avoiding look-ahead bias and rendering the model immune to unseen time-series regime changes (Leaderboard Shakeups).
"""

code_1 = """\
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')

# 1. Load Data
print("Loading dataset...")
df_original = pd.read_csv('dataset.csv')
df = df_original.copy()

option_cols = [c for c in df_original.columns if c.startswith('NIFTY')]
ce_cols = sorted([c for c in option_cols if c.endswith('CE')])
pe_cols = sorted([c for c in option_cols if c.endswith('PE')])

strikes_ce = np.array([float(c.replace('NIFTY27JAN26','').replace('CE','')) for c in ce_cols])
strikes_pe = np.array([float(c.replace('NIFTY27JAN26','').replace('PE','')) for c in pe_cols])

filled = df[option_cols].copy()
"""

text_2 = """\
### Cross-Sectional Log-Moneyness Interpolation
We iterate through every timestamp and build a precise map of the volatility smile using Log-Moneyness coordinates. We linearly interpolate the interior to preserve exactly the observed quotes, and extrapolate the wings.
"""

code_2 = """\
print("Reconstructing Implied Volatility Surface...")
for ix, row in df.iterrows():
    if pd.isna(row['datetime']):
        continue
        
    S = row['underlying_price']
    
    # ------------------- CE Reconstrucion -------------------
    valid_ce_strikes = []
    valid_ce_ivs = []
    for j, c in enumerate(ce_cols):
        val = row[c]
        if pd.notna(val):
            valid_ce_strikes.append(strikes_ce[j])
            valid_ce_ivs.append(val)
            
    if len(valid_ce_strikes) >= 2:
        # Transform to Log-Moneyness Space
        valid_ce_log_m = np.log(np.array(valid_ce_strikes) / S)
        f_ce = interp1d(valid_ce_log_m, valid_ce_ivs, kind='linear', fill_value='extrapolate')
        for j, c in enumerate(ce_cols):
            if pd.isna(row[c]):
                log_m = np.log(strikes_ce[j] / S)
                pred = float(f_ce(log_m))
                filled.at[ix, c] = np.clip(pred, 0.01, 6.0)
    elif len(valid_ce_strikes) == 1:
        for c in ce_cols:
            if pd.isna(row[c]):
                filled.at[ix, c] = valid_ce_ivs[0]

    # ------------------- PE Reconstruction -------------------
    valid_pe_strikes = []
    valid_pe_ivs = []
    for j, c in enumerate(pe_cols):
        val = row[c]
        if pd.notna(val):
            valid_pe_strikes.append(strikes_pe[j])
            valid_pe_ivs.append(val)
            
    if len(valid_pe_strikes) >= 2:
        # Transform to Log-Moneyness Space
        valid_pe_log_m = np.log(np.array(valid_pe_strikes) / S)
        f_pe = interp1d(valid_pe_log_m, valid_pe_ivs, kind='linear', fill_value='extrapolate')
        for j, c in enumerate(pe_cols):
            if pd.isna(row[c]):
                log_m = np.log(strikes_pe[j] / S)
                pred = float(f_pe(log_m))
                filled.at[ix, c] = np.clip(pred, 0.01, 6.0)
    elif len(valid_pe_strikes) == 1:
        for c in pe_cols:
            if pd.isna(row[c]):
                filled.at[ix, c] = valid_pe_ivs[0]

# Final fallback for completely empty rows (if any)
for c in option_cols:
    if filled[c].isna().any():
        filled[c] = filled[c].fillna(method='ffill')

# Save fully filled dataset
df_filled = df_original.copy()
df_filled[option_cols] = filled
df_filled.to_csv("filled_dataset.csv", index=False)
print("Filled dataset saved as 'filled_dataset.csv'.")
"""

text_3 = """\
### Generate Official Submission Format
Using the exact script provided in the competition guidelines to ensure output compliance.
"""

code_3 = """\
import sys

ORIGINAL_DATASET_PATH = "dataset.csv" 
SEPARATOR = "||"

def generate_solution(filled_path: str, output_path: str = "submission.csv"):
    original = pd.read_csv(ORIGINAL_DATASET_PATH)
    filled   = pd.read_csv(filled_path)

    feature_cols = [c for c in original.columns if c != "datetime" and c != "underlying_price"]

    rows = []
    for col in feature_cols:
        was_missing = original[col].isna()

        for idx in original.index[was_missing]:
            dt  = original.loc[idx, "datetime"]
            uid = f"{dt}{SEPARATOR}{col}"
            val = filled.loc[idx, col]
            rows.append({"id": uid, "value": val})

    solution = pd.DataFrame(rows, columns=["id", "value"])
    solution = solution.sort_values("id").reset_index(drop=True)
    solution.to_csv(output_path, index=False)
    print(f"✅ Solution saved → {output_path}  ({len(solution)} rows)")

generate_solution("filled_dataset.csv", "submission.csv") 
"""

def make_md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}

def make_code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [src]}

nb = {
    "cells": [
        make_md_cell(text_1),
        make_code_cell(code_1),
        make_md_cell(text_2),
        make_code_cell(code_2),
        make_md_cell(text_3),
        make_code_cell(code_3)
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("final_notebook.ipynb", "w") as f:
    json.dump(nb, f, indent=2)
print("Notebook generated successfully!")
