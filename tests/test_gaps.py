"""Quick test: Check Bronze gaps"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bronze.gap_detector import check_gaps

gaps = check_gaps()

if gaps:
    print(f"\n📊 Summary: {len(gaps)} gap(s) found")
    print("\nTop 5 largest gaps:")
    sorted_gaps = sorted(gaps, key=lambda x: (x[1] - x[0]).days, reverse=True)[:5]
    for start, end in sorted_gaps:
        days = (end - start).days + 1
        print(f"  {start.date()} → {end.date()} ({days} days)")
else:
    print("\n✅ No gaps found!")
