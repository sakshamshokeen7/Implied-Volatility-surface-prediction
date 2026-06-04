import pandas as pd
import numpy as np
from scipy.interpolate import Akima1DInterpolator, KroghInterpolator, interp1d
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset.csv').dropna(subset=['datetime']).reset_index(drop=True).iloc[:2000]

option_cols = [c for c in df.columns if c.startswith('NIFTY') and (c.endswith('CE') or c.endswith('PE'))]
ce_cols = sorted([c for c in option_cols if c.endswith('CE')])
pe_cols = sorted([c for c in option_cols if c.endswith('PE')])
strikes_ce = np.array([float(c.replace('NIFTY27JAN26','').replace('CE','')) for c in ce_cols])
strikes_pe = np.array([float(c.replace('NIFTY27JAN26','').replace('PE','')) for c in pe_cols])

def get_mask(df, option_cols, seed=42):
    rng = np.random.RandomState(seed)
    masked = df.copy()
    truth = []
    for c in option_cols:
        obs = df.index[df[c].notna()].tolist()
        k = max(1, int(len(obs) * 0.15))
        hide = rng.choice(obs, size=k, replace=False)
        for ix in hide:
            truth.append((ix, c, df.at[ix, c]))
            masked.at[ix, c] = np.nan
    return masked, truth

masked, truth = get_mask(df, option_cols)

def eval_interpolator(masked, truth, interp_class):
    filled = masked[option_cols].copy()
    for ix, row in masked.iterrows():
        # CE
        v_s = [strikes_ce[j] for j, c in enumerate(ce_cols) if pd.notna(row[c])]
        v_iv = [row[c] for j, c in enumerate(ce_cols) if pd.notna(row[c])]
        if len(v_s) > 3:
            try:
                if interp_class == 'akima':
                    f = Akima1DInterpolator(v_s, v_iv)
                    lin = interp1d(v_s, v_iv, kind='linear', fill_value='extrapolate')
                elif interp_class == 'krogh':
                    f = KroghInterpolator(v_s, v_iv)
                    lin = interp1d(v_s, v_iv, kind='linear', fill_value='extrapolate')
                
                for j, c in enumerate(ce_cols):
                    if pd.isna(row[c]):
                        if strikes_ce[j] < v_s[0] or strikes_ce[j] > v_s[-1]:
                            filled.at[ix, c] = lin(strikes_ce[j]) # use linear for wings
                        else:
                            filled.at[ix, c] = f(strikes_ce[j])
            except:
                pass
                
        # PE
        v_s = [strikes_pe[j] for j, c in enumerate(pe_cols) if pd.notna(row[c])]
        v_iv = [row[c] for j, c in enumerate(pe_cols) if pd.notna(row[c])]
        if len(v_s) > 3:
            try:
                if interp_class == 'akima':
                    f = Akima1DInterpolator(v_s, v_iv)
                    lin = interp1d(v_s, v_iv, kind='linear', fill_value='extrapolate')
                elif interp_class == 'krogh':
                    f = KroghInterpolator(v_s, v_iv)
                    lin = interp1d(v_s, v_iv, kind='linear', fill_value='extrapolate')
                
                for j, c in enumerate(pe_cols):
                    if pd.isna(row[c]):
                        if strikes_pe[j] < v_s[0] or strikes_pe[j] > v_s[-1]:
                            filled.at[ix, c] = lin(strikes_pe[j]) # use linear for wings
                        else:
                            filled.at[ix, c] = f(strikes_pe[j])
            except:
                pass

    y_t, y_p = [], []
    for ix, c, tv in truth:
        pv = filled.at[ix, c]
        if pd.notna(pv): y_t.append(tv); y_p.append(np.clip(pv, 0.01, 6.0))
    return mean_squared_error(y_t, y_p)

print("Akima MSE:", eval_interpolator(masked, truth, 'akima'))
print("Krogh MSE:", eval_interpolator(masked, truth, 'krogh'))
