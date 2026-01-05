
# Data Formatting Utilities
import streamlit as st
import pandas as pd

def format_dataframe(df):
    """
    Clean dataframe for display (round floats) but keep types for sorting.
    """
    if df is None: return None
    df_disp = df.copy()
    
    # Auto-detect float columns and round them
    float_cols = df_disp.select_dtypes(include=['float']).columns
    for col in float_cols:
        df_disp[col] = df_disp[col].round(4) # Keep some precision, format in UI
             
    return df_disp

def get_column_config():
    """Get common column configuration for st.dataframe"""
    return {
        "sharpe_ratio": st.column_config.NumberColumn("Sharpe Ratio", format="%.2f"),
        "volatility": st.column_config.NumberColumn("Biến động (Vol %)", format="%.1f%%"),
        "max_drawdown": st.column_config.NumberColumn("Max Drawdown (%)", format="%.1f%%"),
        "avg_daily_return": st.column_config.NumberColumn("Lợi nhuận TB", format="%.4f"),
        "beta": st.column_config.NumberColumn("Beta", format="%.2f"),
        "alpha": st.column_config.NumberColumn("Alpha", format="%.4f"),
        "win_rate": st.column_config.NumberColumn("Win Rate", format="%.1f%%"),
        "profit_factor": st.column_config.NumberColumn("Profit Factor", format="%.2f"),
        "return_stability": st.column_config.NumberColumn("Stability", format="%.2f"),
        "momentum": st.column_config.NumberColumn("Momentum", format="%.2f"),
        "turnover": st.column_config.NumberColumn("Turnover", format="%.2f"),
        "avg_volume": st.column_config.NumberColumn("Vol TB", format="%.2s"), # Use SI prefix (1M, 10K)
        "sector": "Ngành",
        "ticker": "Mã CP"
    }
