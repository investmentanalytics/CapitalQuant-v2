"""
strategies/momentum.py
Estrategia de momentum basada en retornos pasados.

Lógica:
- Long cuando el momentum de N períodos es positivo y supera umbral
- Short cuando el momentum es negativo y supera umbral (en módulo)
- Confirmación con RSI para evitar extremos
- Trailing stop con ATR
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy, Indicators


class MomentumStrategy(BaseStrategy):
    """
    Price Momentum Strategy.

    Compra activos con fuerte momentum positivo reciente
    y vende activos con momentum negativo.
    Basada en el efecto momentum de Jegadeesh & Titman.
    """

    name = "Momentum"
    description = "Estrategia de momentum basada en retornos pasados con filtro RSI"
    version = "1.0.0"

    def __init__(
        self,
        mom_period: int = 20,
        mom_threshold: float = 0.02,
        rsi_period: int = 14,
        rsi_min: float = 40.0,
        rsi_max: float = 60.0,
        atr_period: int = 14,
        atr_multiplier: float = 2.5,
        smooth_period: int = 5,
    ):
        super().__init__(
            mom_period=mom_period,
            mom_threshold=mom_threshold,
            rsi_period=rsi_period,
            rsi_min=rsi_min,
            rsi_max=rsi_max,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            smooth_period=smooth_period,
        )
        self.mom_period = mom_period
        self.mom_threshold = mom_threshold
        self.rsi_period = rsi_period
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.smooth_period = smooth_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Momentum: retorno porcentual de N períodos
        df["momentum"] = Indicators.momentum(df["close"], self.mom_period)

        # Suavizado del momentum para evitar señales ruidosas
        df["momentum_smooth"] = df["momentum"].rolling(self.smooth_period).mean()

        # RSI para confirmar que no estamos en extremos peligrosos
        df["rsi"] = Indicators.rsi(df["close"], self.rsi_period)

        # ATR para stops
        df["atr"] = Indicators.atr(df, self.atr_period)

        # EMA para tendencia macro
        df["ema_50"] = Indicators.ema(df["close"], 50)
        df["ema_200"] = Indicators.ema(df["close"], 200)

        # Aceleración del momentum (derivada)
        df["mom_accel"] = df["momentum_smooth"] - df["momentum_smooth"].shift(1)

        # Condiciones Long:
        # - Momentum suavizado positivo y por encima del umbral
        # - RSI en rango válido (no sobrecomprado extremo)
        # - Tendencia macro alcista
        long_condition = (
            (df["momentum_smooth"] > self.mom_threshold) &
            (df["rsi"] < self.rsi_max) &
            (df["rsi"] > self.rsi_min) &
            (df["ema_50"] > df["ema_200"])
        )

        # Condiciones Short:
        short_condition = (
            (df["momentum_smooth"] < -self.mom_threshold) &
            (df["rsi"] > (100 - self.rsi_max)) &
            (df["rsi"] < (100 - self.rsi_min)) &
            (df["ema_50"] < df["ema_200"])
        )

        # Detectar solo el inicio de la señal (cruce)
        df["signal_raw"] = 0
        df.loc[long_condition, "signal_raw"] = 1
        df.loc[short_condition, "signal_raw"] = -1

        # Señal en cambio de estado (entrada, no mantener)
        df["signal"] = 0
        signal_change = df["signal_raw"] != df["signal_raw"].shift(1)
        df.loc[signal_change & (df["signal_raw"] != 0), "signal"] = \
            df.loc[signal_change & (df["signal_raw"] != 0), "signal_raw"]

        # Stops y targets
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

        return df

    def get_param_space(self) -> dict:
        return {
            "mom_period":      ("int",   5,    40),
            "mom_threshold":   ("float", 0.01, 0.05),
            "rsi_period":      ("int",   5,    21),
            "rsi_min":         ("float", 30.0, 50.0),
            "rsi_max":         ("float", 50.0, 70.0),
            "atr_period":      ("int",   7,    21),
            "atr_multiplier":  ("float", 1.5,  4.0),
            "smooth_period":   ("int",   2,    10),
        }

    def get_indicator_columns(self) -> list:
        return ["ema_50", "ema_200", "momentum_smooth", "rsi"]
