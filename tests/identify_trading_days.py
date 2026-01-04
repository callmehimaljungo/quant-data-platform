"""Identify trading days in gap period"""
import pandas as pd
from datetime import datetime, timedelta

# Gap period
start = datetime(2024, 12, 21)
end = datetime(2025, 1, 2)

# Generate all dates
all_dates = pd.date_range(start, end)

# US Market holidays 2024-2025
holidays = [
    '2024-12-25',  # Christmas
    '2025-01-01',  # New Year
]

# Trading days (exclude weekends and holidays)
trading_days = []
for date in all_dates:
    # Skip weekends
    if date.weekday() >= 5:  # Saturday=5, Sunday=6
        continue
    # Skip holidays
    if date.strftime('%Y-%m-%d') in holidays:
        continue
    trading_days.append(date)

print("📅 Trading Days in Gap (2024-12-21 to 2025-01-02):")
print(f"   Total: {len(trading_days)} days\n")

for i, day in enumerate(trading_days, 1):
    print(f"   {i}. {day.strftime('%Y-%m-%d')} ({day.strftime('%A')})")

print(f"\n✅ Need to collect data for {len(trading_days)} trading days")
