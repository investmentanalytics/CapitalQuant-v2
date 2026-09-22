"""
strategies/ema_trend_following.py
Estrategia de seguimiento de tendencia con EMA cruzada.

Lógica:
- Long cuando EMA rápida cruza por encima de EMA lenta
- Short cuando EMA rápida cruza por debajo de EMA lenta
- Confirmación con volumen y ADX opcional
- Stop loss dinámico con ATR
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy, Indicators


class EMATrendFollowing(BaseStrategy):
    """
    EMA Crossover Trend Following Strategy.

    Sistema clásico de cruce de medias móviles exponenciales
    con gestión de riesgo basada en ATR.
    """

    name = "EMA Trend Following"
    description = "Cruce de EMAs con confirmación de volumen y stop ATR"
    version = "1.0.0"

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        ema_signal: int = 200,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        volume_filter: bool = True,
        volume_period: int = 20,
    ):
        super().__init__(
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            ema_signal=ema_signal,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            volume_filter=volume_filter,
            volume_period=volume_period,
        )
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_signal = ema_signal
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.volume_filter = volume_filter
        self.volume_period = volume_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # EMAs
        df["ema_fast"] = Indicators.ema(df["close"], self.ema_fast)
        df["ema_slow"] = Indicators.ema(df["close"], self.ema_slow)
        df["ema_signal"] = Indicators.ema(df["close"], self.ema_signal)
        df["atr"] = Indicators.atr(df, self.atr_period)

        # Cruce de EMAs (señal en el momento del cruce)
        df["cross_above"] = (
            (df["ema_fast"] > df["ema_slow"]) &
            (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
        )
        df["cross_below"] = (
            (df["ema_fast"] < df["ema_slow"]) &
            (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))
        )

        # Filtro de tendencia principal (EMA larga)
        in_uptrend = df["close"] > df["ema_signal"]
        in_downtrend = df["close"] < df["ema_signal"]

        # Filtro de volumen: el volumen debe superar su media
        if self.volume_filter and df["volume"].sum() > 0:
            vol_sma = Indicators.volume_sma(df["volume"], self.volume_period)
            vol_confirm = df["volume"] > vol_sma
        else:
            vol_confirm = pd.Series(True, index=df.index)

        # Condiciones finales
        long_condition = df["cross_above"] & in_uptrend & vol_confirm
        short_condition = df["cross_below"] & in_downtrend & vol_confirm

        # Señales
        df["signal"] = 0
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1

        # Stop loss y take profit
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
            df["close"] + self.atr_multiplier * 2.0 * df["atr"],
            np.where(
                df["signal"] == -1,
                df["close"] - self.atr_multiplier * 2.0 * df["atr"],
                np.nan,
            ),
        )

        return df

    def get_param_space(self) -> dict:
        return {
            "ema_fast":        ("int",   5,   50),
            "ema_slow":        ("int",   20,  200),
            "ema_signal":      ("int",   100, 400),
            "atr_period":      ("int",   7,   21),
            "atr_multiplier":  ("float", 1.0, 4.0),
        }

    def get_indicator_columns(self) -> list:
        return ["ema_fast", "ema_slow", "ema_signal"]
