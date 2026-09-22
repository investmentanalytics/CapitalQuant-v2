"""
strategies/range_reversion.py

Estrategia de reversión a la media para mercados laterales.

Compra cerca de la banda inferior de Bollinger.
Vende cerca de la banda superior.

Ideal para rangos.
"""

from .base import BaseStrategy, Indicators
import pandas as pd
import numpy as np


class RangeReversion(BaseStrategy):

    name = "Range Reversion"
    description = "Reversión a la media con Bollinger + RSI"
    version = "1.0.0"

    def __init__(
        self,
        bb_period=20,
        bb_std=2.0,
        rsi_period=14,
        rsi_buy=30,
        rsi_sell=70,
        atr_period=14,
        atr_multiplier=2.0,
    ):

        super().__init__(
            bb_period=bb_period,
            bb_std=bb_std,
            rsi_period=rsi_period,
            rsi_buy=rsi_buy,
            rsi_sell=rsi_sell,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
        )

        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def generate_signals(self, data: pd.DataFrame):

        df = data.copy()

        # ==================================================
        # INDICADORES
        # ==================================================

        bb_upper, bb_mid, bb_lower = Indicators.bollinger(
            df["close"],
            period=self.bb_period,
            std_dev=self.bb_std
        )
        df["bb_upper"] = bb_upper
        df["bb_mid"] = bb_mid
        df["bb_lower"] = bb_lower

        df["rsi"] = Indicators.rsi(
            df["close"],
            self.rsi_period
        )

        df["atr"] = Indicators.atr(
            df,
            self.atr_period
        )

        # ==================================================
        # ENTRADAS
        # ==================================================

        long_condition = (
            (df["close"] <= df["bb_lower"])
            &
            (df["rsi"] < self.rsi_buy)
        )

        short_condition = (
            (df["close"] >= df["bb_upper"])
            &
            (df["rsi"] > self.rsi_sell)
        )

        df["signal"] = 0

        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1

        # ==================================================
        # STOP LOSS
        # ==================================================

        df["stop_loss"] = np.where(
            df["signal"] == 1,
            df["close"] - (
                df["atr"] * self.atr_multiplier
            ),
            np.where(
                df["signal"] == -1,
                df["close"] + (
                    df["atr"] * self.atr_multiplier
                ),
                np.nan
            )
        )

        # ==================================================
        # TAKE PROFIT
        # ==================================================

        df["take_profit"] = np.where(
            df["signal"] == 1,
            df["bb_mid"],
            np.where(
                df["signal"] == -1,
                df["bb_mid"],
                np.nan
            )
        )

        return df

    def get_param_space(self):

        return {

            "bb_period":
                ("int", 10, 50),

            "bb_std":
                ("float", 1.0, 3.5),

            "rsi_period":
                ("int", 5, 30),

            "rsi_buy":
                ("int", 10, 40),

            "rsi_sell":
                ("int", 60, 90),

            "atr_period":
                ("int", 5, 30),

            "atr_multiplier":
                ("float", 1.0, 5.0),
        }

    def get_indicator_columns(self):

        return [
            "bb_upper",
            "bb_mid",
            "bb_lower",
            "rsi",
        ]