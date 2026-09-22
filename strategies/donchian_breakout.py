"""
strategies/donchian_breakout.py
Estrategia de ruptura de canales Donchian.

Lógica:
- Long cuando el precio supera el máximo de los últimos N períodos
- Short cuando el precio cae por debajo del mínimo de los últimos N períodos
- Salida cuando precio cruza la línea media del canal
- Gestión de riesgo con ATR
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy, Indicators


class DonchianBreakout(BaseStrategy):
    """
    Donchian Channel Breakout Strategy.

    Inspirada en el famoso sistema Turtle Trading de Richard Dennis.
    Opera rupturas de canal con confirmación y gestión ATR.
    """

    name = "Donchian Breakout"
    description = "Ruptura de canales Donchian (inspirado en Turtle Trading)"
    version = "1.0.0"

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        breakout_confirm: int = 1,
    ):
        super().__init__(
            entry_period=entry_period,
            exit_period=exit_period,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            breakout_confirm=breakout_confirm,
            )
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.breakout_confirm = breakout_confirm

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Canal Donchian de entrada (más largo)
        df["don_upper_entry"], df["don_lower_entry"], df["don_mid_entry"] = \
            Indicators.donchian(df, self.entry_period)

        # Canal Donchian de salida (más corto, para seguir la tendencia)
        df["don_upper_exit"], df["don_lower_exit"], df["don_mid_exit"] = \
            Indicators.donchian(df, self.exit_period)

        df["atr"] = Indicators.atr(df, self.atr_period)


        # Ruptura: precio supera el canal del período anterior
        upper_prev = df["don_upper_entry"].shift(
            self.breakout_confirm + 1
       )

        lower_prev = df["don_lower_entry"].shift(
            self.breakout_confirm + 1
       )

        long_break = df["close"] > upper_prev
        short_break = df["close"] < lower_prev

        # Señales
        df["signal"] = 0

        long_entry = long_break & (~long_break.shift(1).fillna(False).astype(bool))
        short_entry = short_break & (~short_break.shift(1).fillna(False).astype(bool))

        df.loc[long_entry, "signal"] = 1
        df.loc[short_entry, "signal"] = -1

        # Stop loss basado en canal de salida y ATR
        df["stop_loss"] = np.where(
            df["signal"] == 1,
            np.minimum(
                df["don_lower_exit"],
                df["close"] - self.atr_multiplier * df["atr"],
            ),
            np.where(
                df["signal"] == -1,
                np.maximum(
                    df["don_upper_exit"],
                    df["close"] + self.atr_multiplier * df["atr"],
                ),
                np.nan,
            ),
        )

        # Take profit: objetivo basado en amplitud del canal
        channel_width = df["don_upper_entry"] - df["don_lower_entry"]
        df["take_profit"] = np.where(
            df["signal"] == 1,
            df["close"] + channel_width,
            np.where(
                df["signal"] == -1,
                df["close"] - channel_width,
                np.nan,
            ),
        )

        return df

    def get_param_space(self) -> dict:
        return {
            "entry_period":    ("int",   10, 60),
            "exit_period":     ("int",   5,  30),
            "atr_period":      ("int",   7,  21),
            "atr_multiplier":  ("float", 1.0, 4.0),
            "breakout_confirm": ("int",  1,  3),
        }

    def get_indicator_columns(self) -> list:
        return ["don_upper_entry", "don_lower_entry",
                "don_upper_exit", "don_lower_exit"]
