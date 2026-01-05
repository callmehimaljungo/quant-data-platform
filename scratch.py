"""
Scratch file for quick testing - DO NOT COMMIT important stuff here!
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Quick test of R2 connection
# from utils.r2_sync import list_r2_objects
# print(list_r2_objects('processed/'))

# Testing regex for ticker parsing
# import re
# text = "AAPL (Apple Inc.)"
# match = re.match(r"(\w+)", text)
# print(match.group(1))

# Checking parquet schema
# df = pd.read_parquet(r"data\gold\cache\risk_metrics.parquet")
# print(df.columns)
# print(df.head())
# print(df['volatility'].describe())
