#!/usr/bin/env python
"""
Manual Cache Sync Tool

Copies latest lakehouse files to cache folder and uploads to R2.
Use this for manual sync when not running full pipeline.

Usage:
    python tools/fix_cache_and_sync.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gold.utils import sync_lakehouse_to_cache

print("=" * 60)
print("REFRESHING GOLD CACHE FROM LAKEHOUSE")
print("=" * 60)

# Sync lakehouse to cache
results = sync_lakehouse_to_cache()

success_count = sum(1 for v in results.values() if v)
print(f"\nCache sync complete: {success_count}/{len(results)} strategies updated")

# Trigger R2 Upload
print("\nTriggering R2 Upload...")
try:
    import subprocess
    cmd = [sys.executable, str(PROJECT_ROOT / 'utils' / 'r2_sync.py'), 'upload-gold']
    subprocess.run(cmd, check=True)
    print("\n[DONE] Cache synced and uploaded to R2")
except Exception as e:
    print(f"[WARN] Upload failed: {e}")
