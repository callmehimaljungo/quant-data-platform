
import pandas as pd
from pathlib import Path

# Target the latest file explicitly
path = Path(r"e:\GitHub\quant-trade\quant-data-platform\data\gold\low_beta_quality_lakehouse\data_v0_20260104_111248.parquet")
print(f"Reading: {path}")

if not path.exists():
    print("FILE NOT FOUND!")
    exit(1)

df = pd.read_parquet(path)
print(f"Shape: {df.shape}")
print(df[['ticker', 'sector']].head(20))

unknowns = df[df['sector'] == 'Unknown']
print(f"Unknown count: {len(unknowns)}")
if len(unknowns) > 0:
    print("Sample Unknowns:")
    print(unknowns.head())
