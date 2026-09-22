from __future__ import annotations
import pandas as pd
from config.settings import TIMEFRAMES
from regime.service import compute_regime_summary

def build_regime_mask(df: pd.DataFrame, asset: str, timeframe: str,
                      allowed_regimes: list[str]) -> pd.Series | None:
    if not allowed_regimes:
        return None
    ann_factor = TIMEFRAMES.get(timeframe, {}).get("ann_factor", 252)
    summary = compute_regime_summary(df, ann_factor=ann_factor)
    return summary.regime_series.isin(allowed_regimes).reindex(df.index).fillna(False)
