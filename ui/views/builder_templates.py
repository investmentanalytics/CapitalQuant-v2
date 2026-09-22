"""
ui/views/builder_templates.py
Plantillas de partida para el Constructor de Estrategias.

Todas siguen el mismo contrato que las estrategias precargadas: heredan
de BaseStrategy, usan Indicators para el cálculo, y devuelven un
DataFrame con la columna 'signal' (1/-1/0) y, opcionalmente,
'stop_loss' / 'take_profit' fijados EN LA MISMA FILA que la señal.
"""

BLANK = '''class MiEstrategia(BaseStrategy):
    name = "Mi Estrategia"
    description = "Describe aquí la lógica"

    def __init__(self, periodo=20, **kwargs):
        super().__init__(periodo=periodo, **kwargs)
        self.periodo = periodo

    def generate_signals(self, data):
        df = data.copy()
        df["mi_indicador"] = Indicators.sma(df["close"], self.periodo)

        df["signal"] = 0
        # df.loc[condicion_de_entrada_larga, "signal"] = 1
        # df.loc[condicion_de_entrada_corta, "signal"] = -1
        return df

    def get_param_space(self):
        return {"periodo": ("int", 5, 100)}

    def get_indicator_columns(self):
        return ["mi_indicador"]
'''

TREND_LONG_ONLY = '''class TendenciaSoloLargos(BaseStrategy):
    name = "Tendencia (solo largos)"
    description = "Cruce de EMAs a favor de la tendencia. Nunca abre cortos."

    def __init__(self, ema_fast=20, ema_slow=50, atr_period=14,
                 atr_sl_mult=2.0, atr_tp_mult=4.0, **kwargs):
        super().__init__(ema_fast=ema_fast, ema_slow=ema_slow, atr_period=atr_period,
                          atr_sl_mult=atr_sl_mult, atr_tp_mult=atr_tp_mult, **kwargs)
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_period = atr_period
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult

    def generate_signals(self, data):
        df = data.copy()
        df["ema_fast"] = Indicators.ema(df["close"], self.ema_fast)
        df["ema_slow"] = Indicators.ema(df["close"], self.ema_slow)
        df["atr"] = Indicators.atr(df, self.atr_period)

        cross_up = (df["ema_fast"] > df["ema_slow"]) & \\
                   (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))

        df["signal"] = 0
        df.loc[cross_up, "signal"] = 1     # solo largos — nunca se asigna -1

        # SL/TP basados en ATR, solo en la fila donde nace la señal
        df["stop_loss"] = np.where(
            df["signal"] == 1, df["close"] - self.atr_sl_mult * df["atr"], np.nan
        )
        df["take_profit"] = np.where(
            df["signal"] == 1, df["close"] + self.atr_tp_mult * df["atr"], np.nan
        )
        return df

    def get_param_space(self):
        return {
            "ema_fast": ("int", 5, 40),
            "ema_slow": ("int", 20, 150),
            "atr_sl_mult": ("float", 1.0, 4.0),
            "atr_tp_mult": ("float", 2.0, 8.0),
        }

    def get_indicator_columns(self):
        return ["ema_fast", "ema_slow"]
'''

REVERSION_SHORT_ONLY = '''class ReversionSoloCortos(BaseStrategy):
    name = "Reversion (solo cortos)"
    description = "Vende sobrecompra con RSI + banda superior de Bollinger. Nunca abre largos."

    def __init__(self, rsi_period=14, rsi_overbought=70,
                 bb_period=20, bb_std=2.0, **kwargs):
        super().__init__(rsi_period=rsi_period, rsi_overbought=rsi_overbought,
                          bb_period=bb_period, bb_std=bb_std, **kwargs)
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std

    def generate_signals(self, data):
        df = data.copy()
        df["rsi"] = Indicators.rsi(df["close"], self.rsi_period)
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = Indicators.bollinger(
            df["close"], self.bb_period, self.bb_std
        )

        short_cond = (df["close"] > df["bb_upper"]) & (df["rsi"] > self.rsi_overbought)

        df["signal"] = 0
        df.loc[short_cond, "signal"] = -1   # solo cortos — nunca se asigna 1

        # Salida objetivo: vuelta a la media móvil de las bandas
        df["take_profit"] = np.where(df["signal"] == -1, df["bb_mid"], np.nan)
        df["stop_loss"] = np.where(
            df["signal"] == -1, df["close"] * 1.02, np.nan  # 2% por encima de la entrada
        )
        return df

    def get_param_space(self):
        return {
            "rsi_period": ("int", 5, 30),
            "rsi_overbought": ("int", 60, 90),
            "bb_period": ("int", 10, 40),
        }

    def get_indicator_columns(self):
        return ["bb_upper", "bb_mid", "bb_lower"]
'''

MULTI_INDICATOR_CONFLUENCE = '''class ConfluenciaMultiIndicador(BaseStrategy):
    name = "Confluencia Multi-Indicador"
    description = "EMA (tendencia) + RSI (momentum) + MACD (confirmacion). Largos y cortos."

    def __init__(self, ema_period=50, rsi_period=14,
                 macd_fast=12, macd_slow=26, macd_signal=9, **kwargs):
        super().__init__(ema_period=ema_period, rsi_period=rsi_period,
                          macd_fast=macd_fast, macd_slow=macd_slow,
                          macd_signal=macd_signal, **kwargs)
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def generate_signals(self, data):
        df = data.copy()
        df["ema"] = Indicators.ema(df["close"], self.ema_period)
        df["rsi"] = Indicators.rsi(df["close"], self.rsi_period)
        df["macd"], df["macd_signal"], df["macd_hist"] = Indicators.macd(
            df["close"], self.macd_fast, self.macd_slow, self.macd_signal
        )

        # --- Las 3 condiciones deben alinearse (confluencia) ---
        trend_up   = df["close"] > df["ema"]
        trend_down = df["close"] < df["ema"]
        momentum_up   = df["rsi"] > 50
        momentum_down = df["rsi"] < 50
        macd_up   = df["macd"] > df["macd_signal"]
        macd_down = df["macd"] < df["macd_signal"]

        long_cond  = trend_up & momentum_up & macd_up
        short_cond = trend_down & momentum_down & macd_down

        df["signal"] = 0
        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1
        return df

    def get_param_space(self):
        return {
            "ema_period": ("int", 20, 100),
            "rsi_period": ("int", 5, 30),
            "macd_fast": ("int", 5, 20),
            "macd_slow": ("int", 15, 40),
        }

    def get_indicator_columns(self):
        return ["ema"]
'''

SUPERTREND_ADX = '''class SuperTrendADX(BaseStrategy):
    name = "SuperTrend + ADX"
    description = "Sigue tendencia con SuperTrend, filtra por fuerza de tendencia con ADX. Largos y cortos."

    def __init__(self, st_period=10, st_mult=3.0, adx_period=14, adx_min=20, **kwargs):
        super().__init__(st_period=st_period, st_mult=st_mult,
                          adx_period=adx_period, adx_min=adx_min, **kwargs)
        self.st_period = st_period
        self.st_mult = st_mult
        self.adx_period = adx_period
        self.adx_min = adx_min

    def generate_signals(self, data):
        df = data.copy()
        df["st_line"], df["st_dir"] = Indicators.supertrend(df, self.st_period, self.st_mult)
        _, _, df["adx"] = Indicators.adx(df, self.adx_period)

        # Entrar solo cuando SuperTrend cambia de direccion Y el ADX confirma tendencia fuerte
        flip_up = (df["st_dir"] == 1) & (df["st_dir"].shift(1) == -1)
        flip_down = (df["st_dir"] == -1) & (df["st_dir"].shift(1) == 1)
        trend_strong = df["adx"] > self.adx_min

        df["signal"] = 0
        df.loc[flip_up & trend_strong, "signal"] = 1
        df.loc[flip_down & trend_strong, "signal"] = -1

        # Stop loss en la propia linea de SuperTrend
        df["stop_loss"] = np.where(df["signal"] != 0, df["st_line"], np.nan)
        return df

    def get_param_space(self):
        return {
            "st_period": ("int", 5, 25),
            "st_mult": ("float", 1.5, 5.0),
            "adx_min": ("int", 10, 40),
        }

    def get_indicator_columns(self):
        return ["st_line"]
'''

TEMPLATES = {
    "Plantilla en blanco": BLANK,
    "Tendencia (solo largos)": TREND_LONG_ONLY,
    "Reversion (solo cortos)": REVERSION_SHORT_ONLY,
    "Confluencia multi-indicador (largos y cortos)": MULTI_INDICATOR_CONFLUENCE,
    "SuperTrend + ADX (largos y cortos)": SUPERTREND_ADX,
}
