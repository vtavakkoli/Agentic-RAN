from __future__ import annotations
import pandas as pd

def rank_forecasting(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["r2","rmse","mae","wmape","peak_mae"], ascending=[False,True,True,True,True])

def rank_control(df: pd.DataFrame) -> pd.DataFrame:
    cols=[c for c in ["average_reward","cumulative_reward","safe_fallback_rate","action_switch_rate"] if c in df.columns]
    asc=[False,False,True,True][:len(cols)]
    return df.sort_values(cols, ascending=asc) if cols else df
