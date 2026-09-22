"""
strategies/rsi_mean_reversion.py
Estrategia de reversión a la media usando RSI.

Lógica:
- Long cuando RSI < oversold_level (zona de sobreventa)
- Short cuando RSI > overbought_level (zona de sobrecompra)
- Salida cuando RSI cruza la línea neutral (50)
- Stop loss basado en ATR
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy, Indicators


class RSIMeanReversion(BaseStrategy):
    """
    RSI Mean Reversion Strategy.

    Compra en sobreventa extrema y vende en sobrecompra extrema.
    Usa ATR para gestión de riesgo dinámica.
    """

    name = "RSI Mean Reversion"
    description = "Reversión a la media con RSI + gestión de riesgo ATR"
    version = "1.0.0"

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        use_trend_filter: bool = True,
        trend_ema: int = 200,
    ):
        super().__init__(
            rsi_period=rsi_period,
            oversold=oversold,
            overbought=overbought,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            use_trend_filter=use_trend_filter,
            trend_ema=trend_ema,
        )
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.use_trend_filter = use_trend_filter
        self.trend_ema = trend_ema

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Indicadores
        df["rsi"] = Indicators.rsi(df["close"], self.rsi_period)
        df["atr"] = Indicators.atr(df, self.atr_period)
        df["ema_trend"] = Indicators.ema(df["close"], self.trend_ema)

        # Condiciones base
        oversold_condition = df["rsi"] < self.oversold
        overbought_condition = df["rsi"] > self.overbought

        # Filtro de tendencia: solo operar en dirección de la tendencia principal
        if self.use_trend_filter:
            uptrend = df["close"] > df["ema_trend"]
            downtrend = df["close"] < df["ema_trend"]
            long_condition = oversold_condition & uptrend
            short_condition = overbought_condition & downtrend
        else:
            long_condition = oversold_condition
            short_condition = overbought_condition

        # Generación de señales
        df["signal"] = 0
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1

        # Stop loss y take profit basados en ATR
        df["stop_loss"] = np.where(
            df["signal"] == 1,
            df["close"] - self.atr_multiplier * df["atr"],
            np.where(
                df["signal"] == -1,
                df["close"] + self.atr_multiplier * df["atr"],
                np.nan,
            ),
        )
        df["take_profit"] = np.where(
            df["signal"] == 1,
            df["close"] + self.atr_multiplier * 1.5 * df["atr"],
            np.where(
                df["signal"] == -1,
                df["close"] - self.atr_multiplier * 1.5 * df["atr"],
                np.nan,
            ),
        )

        # Niveles RSI para visualización
        df["rsi_oversold"] = self.oversold
        df["rsi_overbought"] = self.overbought

        return df

    def get_param_space(self) -> dict:
        return {
            "rsi_period":      ("int",   5,    30),
            "oversold":        ("float", 20.0, 40.0),
            "overbought":      ("float", 60.0, 80.0),
            "atr_period":      ("int",   7,    21),
            "atr_multiplier":  ("float", 1.0,  4.0),
            "trend_ema":       ("int",   50,   300),
        }

    def get_indicator_columns(self) -> list:
        return ["rsi", "ema_trend"]
