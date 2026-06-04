import pandas as pd
df = pd.read_csv('dataset.csv')
c_cols = sorted([c for c in df.columns if 'CE' in c and c.startswith('NIFTY')])
p_cols = sorted([c for c in df.columns if 'PE' in c and c.startswith('NIFTY')])

count = 0
for c, p in zip(c_cols, p_cols):
    count += ((df[c].isna()) & (df[p].notna())).sum()
    count += ((df[c].notna()) & (df[p].isna())).sum()
print("Total mismatched missing values:", count)

# Let's also check if the IVs are exactly the same when both are present
same_count = 0
diff_count = 0
for c, p in zip(c_cols, p_cols):
    mask = df[c].notna() & df[p].notna()
    diffs = abs(df.loc[mask, c] - df.loc[mask, p])
    same_count += (diffs < 1e-4).sum()
    diff_count += (diffs >= 1e-4).sum()
print("Same IVs:", same_count)
print("Different IVs:", diff_count)
