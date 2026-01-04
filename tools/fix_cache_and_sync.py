
import sys
import shutil
import glob
import os
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
GOLD_DIR = PROJECT_ROOT / 'data' / 'gold'
CACHE_DIR = GOLD_DIR / 'cache'

CACHE_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIES = [
    'low_beta_quality',
    'sector_rotation',
    'sentiment_allocation'
]

print("="*60)
print("REFRESHING GOLD CACHE FROM LAKEHOUSE")
print("="*60)

for strategy in STRATEGIES:
    lakehouse_dir = GOLD_DIR / f'{strategy}_lakehouse'
    if not lakehouse_dir.exists():
        print(f"[SKIP] Lakehouse not found: {lakehouse_dir}")
        continue
    
    # Find latest parquet
    files = sorted(lakehouse_dir.glob('*.parquet'), key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        print(f"[WARN] No parquet files in {lakehouse_dir}")
        continue
    
    latest_file = files[0]
    dest_file = CACHE_DIR / f'{strategy}_weights.parquet'
    
    print(f"Copying {latest_file.name} -> cache/{dest_file.name}")
    try:
        shutil.copy2(latest_file, dest_file)
        print(f"  [OK] Updated.")
    except Exception as e:
        print(f"  [FAIL] {e}")

print("\nCache refresh complete.")

# Now trigger R2 Upload
try:
    print("\nTriggering R2 Upload...")
    import subprocess
    cmd = [sys.executable, str(PROJECT_ROOT / 'utils' / 'r2_sync.py'), 'upload-gold']
    subprocess.run(cmd, check=True)
except Exception as e:
    print(f"Upload failed: {e}")
