"""
strategies/base.py
Clase base para todas las estrategias de trading + librería de indicadores.

Las estrategias no se tocan — la arquitectura plug-and-play del original
es correcta y se preserva íntegramente.
"""
from abc import ABC, abstractmethod
from typing import Optional, List
import pandas as pd
import numpy as np

from core.validators import validate_ohlcv


class BaseStrategy(ABC):
    """
    Clase base abstracta para estrategias de trading.

    Todas las estrategias heredan de esta clase e implementan
    `generate_signals`. Los parámetros se pasan como kwargs.

    Añadir una estrategia nueva = crear un archivo .py en strategies/
    La estrategia aparece automáticamente en toda la plataforma.
    """

    name: str = "Base Strategy"
    description: str = "Estrategia base"
    version: str = "1.0.0"
    author: str = "CapitalQuant"

    def __init__(self, **params):
        self.params = params
        self._validate_params()

    def _validate_params(self):
        """Validar parámetros — sobreescribir si es necesario."""
        pass

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Genera señales de trading sobre el DataFrame de precios.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV con columnas: open, high, low, close, volume
            Index: DatetimeIndex

        Returns
        -------
        pd.DataFrame con las columnas originales más:
            - signal    : int    1=Long, -1=Short, 0=Sin posición
            - stop_loss : float  (opcional)
            - take_profit: float  (opcional)
            - [indicadores calculados para visualización]
        """
        ...

    def generate_signals_safe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Wrapper con validación automática de entrada."""
        validate_ohlcv(data, name=self.name)
        return self.generate_signals(data)

    def get_param_space(self) -> dict:
        """
        Define el espacio de parámetros para optimización con Optuna.

        Returns dict: {nombre: (tipo, min, max) | (tipo, opciones)}
        Ejemplo: {"rsi_period": ("int", 5, 50)}
        """
        return {}

    def get_indicator_columns(self) -> List[str]:
        """Columnas de indicadores a mostrar en el gráfico."""
        return []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"


# ---------------------------------------------------------------------------
# Librería de indicadores técnicos
# ---------------------------------------------------------------------------

class Indicators:
    """Indicadores técnicos basados en pandas/numpy. Sin dependencias externas."""

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close_prev = df["high"], df["low"], df["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, min_periods=period).mean()

    @staticmethod
    def donchian(df: pd.DataFrame, period: int = 20) -> tuple:
        upper = df["high"].rolling(period).max()
        lower = df["low"].rolling(period).min()
        mid = (upper + lower) / 2
        return upper, lower, mid

    @staticmethod
    def bollinger(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple:
        mid = series.rolling(period).mean()
        std = series.rolling(period).std()
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        return upper, mid, lower

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple:
        lowest_low = df["low"].rolling(k_period).min()
        highest_high = df["high"].rolling(k_period).max()
        k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(d_period).mean()
        return k, d

    @staticmethod
    def momentum(series: pd.Series, period: int = 14) -> pd.Series:
        return series / series.shift(period) - 1

    @staticmethod
    def volume_sma(series: pd.Series, period: int = 20) -> pd.Series:
        return series.rolling(period).mean()

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """VWAP intraday — reseteado por sesión."""
        typical = (df["high"] + df["low"] + df["close"]) / 3
        return (typical * df["volume"]).cumsum() / df["volume"].cumsum()

    # -----------------------------------------------------------------
    # TENDENCIA
    # -----------------------------------------------------------------

    @staticmethod
    def wma(series: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average — pondera linealmente los precios más recientes."""
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def dema(series: pd.Series, period: int) -> pd.Series:
        """Double EMA — reduce el lag de una EMA simple."""
        ema1 = series.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        return 2 * ema1 - ema2

    @staticmethod
    def tema(series: pd.Series, period: int) -> pd.Series:
        """Triple EMA — aún menos lag que DEMA."""
        ema1 = series.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3

    @staticmethod
    def hma(series: pd.Series, period: int) -> pd.Series:
        """Hull Moving Average — muy reactiva y suave a la vez."""
        half = max(int(period / 2), 1)
        sqrt_p = max(int(np.sqrt(period)), 1)
        diff = 2 * Indicators.wma(series, half) - Indicators.wma(series, period)
        return Indicators.wma(diff, sqrt_p)

    @staticmethod
    def kama(series: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
        """Kaufman Adaptive Moving Average — se acelera en tendencia, se frena en rango."""
        change = (series - series.shift(period)).abs()
        volatility = series.diff().abs().rolling(period).sum()
        er = (change / volatility.replace(0, np.nan)).fillna(0)
        fast_sc, slow_sc = 2 / (fast + 1), 2 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        vals, sc_vals = series.values, sc.values
        out = np.full(len(vals), np.nan)
        if len(vals) > period:
            out[period] = vals[period]
            for i in range(period + 1, len(vals)):
                prev = out[i - 1] if not np.isnan(out[i - 1]) else vals[i - 1]
                sc_i = sc_vals[i] if not np.isnan(sc_vals[i]) else slow_sc ** 2
                out[i] = prev + sc_i * (vals[i] - prev)
        return pd.Series(out, index=series.index)

    @staticmethod
    def vwma(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Volume Weighted Moving Average."""
        pv = df["close"] * df["volume"]
        return pv.rolling(period).sum() / df["volume"].rolling(period).sum()

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> tuple:
        """Average Directional Index (Wilder). Devuelve (+DI, -DI, ADX)."""
        high, low, close = df["high"], df["low"], df["close"]
        up_move, down_move = high.diff(), -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr = pd.concat([
            high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(com=period - 1, min_periods=period).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(com=period - 1, min_periods=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(com=period - 1, min_periods=period).mean() / atr
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(com=period - 1, min_periods=period).mean()
        return plus_di, minus_di, adx

    @staticmethod
    def aroon(df: pd.DataFrame, period: int = 25) -> tuple:
        """Aroon Up / Aroon Down / Aroon Oscillator."""
        high, low = df["high"], df["low"]
        aroon_up = high.rolling(period + 1).apply(lambda x: np.argmax(x) / period * 100, raw=True)
        aroon_down = low.rolling(period + 1).apply(lambda x: np.argmin(x) / period * 100, raw=True)
        return aroon_up, aroon_down, aroon_up - aroon_down

    @staticmethod
    def parabolic_sar(df: pd.DataFrame, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
        """Parabolic SAR (Wilder) — puntos de stop-and-reverse que siguen la tendencia."""
        high, low, close = df["high"].values, df["low"].values, df["close"].values
        n = len(close)
        sar = np.full(n, np.nan)
        if n < 2:
            return pd.Series(sar, index=df.index)
        trend, ep, af = 1, high[0], af_step
        sar[0] = low[0]
        for i in range(1, n):
            prev_sar = sar[i - 1]
            if trend == 1:
                sar_i = prev_sar + af * (ep - prev_sar)
                sar_i = min(sar_i, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
                if low[i] < sar_i:
                    trend, sar_i, ep, af = -1, ep, low[i], af_step
                elif high[i] > ep:
                    ep, af = high[i], min(af + af_step, af_max)
            else:
                sar_i = prev_sar + af * (ep - prev_sar)
                sar_i = max(sar_i, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
                if high[i] > sar_i:
                    trend, sar_i, ep, af = 1, ep, high[i], af_step
                elif low[i] < ep:
                    ep, af = low[i], min(af + af_step, af_max)
            sar[i] = sar_i
        return pd.Series(sar, index=df.index)

    @staticmethod
    def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> tuple:
        """SuperTrend. Devuelve (línea supertrend, dirección: 1=alcista, -1=bajista)."""
        atr = Indicators.atr(df, period)
        hl2 = (df["high"] + df["low"]) / 2
        upper_basic = (hl2 + multiplier * atr).values
        lower_basic = (hl2 - multiplier * atr).values
        close = df["close"].values
        n = len(close)
        final_upper, final_lower = np.full(n, np.nan), np.full(n, np.nan)
        st, direction = np.full(n, np.nan), np.ones(n)

        for i in range(n):
            if i == 0 or np.isnan(upper_basic[i]):
                final_upper[i], final_lower[i] = upper_basic[i], lower_basic[i]
                continue
            # Si el ATR todavía no tenía suficientes datos en la barra anterior
            # (final_upper[i-1] es NaN), arrancar en fresco con el valor básico
            # en vez de comparar contra NaN (esa comparación siempre da False y
            # dejaría la serie en NaN para siempre).
            final_upper[i] = (
                upper_basic[i] if np.isnan(final_upper[i - 1]) else
                (upper_basic[i] if (upper_basic[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1])
                 else final_upper[i - 1])
            )
            final_lower[i] = (
                lower_basic[i] if np.isnan(final_lower[i - 1]) else
                (lower_basic[i] if (lower_basic[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1])
                 else final_lower[i - 1])
            )
        for i in range(n):
            if i == 0 or np.isnan(final_upper[i]):
                st[i], direction[i] = final_upper[i], 1
                continue
            if st[i - 1] == final_upper[i - 1]:
                st[i], direction[i] = (final_upper[i], -1) if close[i] <= final_upper[i] else (final_lower[i], 1)
            else:
                st[i], direction[i] = (final_lower[i], 1) if close[i] >= final_lower[i] else (final_upper[i], -1)
        return pd.Series(st, index=df.index), pd.Series(direction, index=df.index)

    @staticmethod
    def ichimoku(df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26,
                 senkou_b_period: int = 52, displacement: int = 26) -> tuple:
        """Ichimoku Kinko Hyo. Devuelve (tenkan, kijun, senkou_a, senkou_b, chikou)."""
        high, low = df["high"], df["low"]
        tenkan = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2
        kijun = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(displacement)
        senkou_b = ((high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2).shift(displacement)
        chikou = df["close"].shift(-displacement)
        return tenkan, kijun, senkou_a, senkou_b, chikou

    @staticmethod
    def trix(series: pd.Series, period: int = 15) -> pd.Series:
        """TRIX — tasa de cambio de una EMA triple, filtra el ruido de corto plazo."""
        ema1 = series.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return ema3.pct_change() * 100

    @staticmethod
    def vortex(df: pd.DataFrame, period: int = 14) -> tuple:
        """Vortex Indicator. Devuelve (VI+, VI-)."""
        high, low, close = df["high"], df["low"], df["close"]
        prev_close, prev_low, prev_high = close.shift(1), low.shift(1), high.shift(1)
        vm_plus = (high - prev_low).abs()
        vm_minus = (low - prev_high).abs()
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        vi_plus = vm_plus.rolling(period).sum() / tr.rolling(period).sum()
        vi_minus = vm_minus.rolling(period).sum() / tr.rolling(period).sum()
        return vi_plus, vi_minus

    @staticmethod
    def mass_index(df: pd.DataFrame, period: int = 9, sum_period: int = 25) -> pd.Series:
        """Mass Index — detecta reversiones por expansión del rango high-low."""
        hl = df["high"] - df["low"]
        ema1 = hl.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        return (ema1 / ema2).rolling(sum_period).sum()

    @staticmethod
    def dpo(series: pd.Series, period: int = 20) -> pd.Series:
        """Detrended Price Oscillator — elimina la tendencia para ver ciclos."""
        shift = int(period / 2) + 1
        return series.shift(shift) - series.rolling(period).mean()

    @staticmethod
    def linreg_slope(series: pd.Series, period: int = 14) -> pd.Series:
        """Pendiente de la regresión lineal sobre una ventana móvil."""
        idx = np.arange(period)

        def _slope(x):
            return np.nan if np.isnan(x).any() else np.polyfit(idx, x, 1)[0]

        return series.rolling(period).apply(_slope, raw=True)

    @staticmethod
    def linreg_line(series: pd.Series, period: int = 14) -> pd.Series:
        """Valor proyectado de la regresión lineal al final de la ventana."""
        idx = np.arange(period)

        def _val(x):
            if np.isnan(x).any():
                return np.nan
            slope, intercept = np.polyfit(idx, x, 1)
            return slope * (period - 1) + intercept

        return series.rolling(period).apply(_val, raw=True)

    @staticmethod
    def choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Choppiness Index — cerca de 100 = mercado en rango, cerca de 0 = tendencia fuerte."""
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr_sum = tr.rolling(period).sum()
        rng = high.rolling(period).max() - low.rolling(period).min()
        return 100 * np.log10(atr_sum / rng) / np.log10(period)

    @staticmethod
    def coppock_curve(series: pd.Series, roc1: int = 14, roc2: int = 11, wma_period: int = 10) -> pd.Series:
        """Coppock Curve — momentum de largo plazo, clásico para señales de fondo de mercado."""
        roc_sum = Indicators.roc(series, roc1) + Indicators.roc(series, roc2)
        return Indicators.wma(roc_sum, wma_period)

    # -----------------------------------------------------------------
    # MOMENTUM
    # -----------------------------------------------------------------

    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Commodity Channel Index."""
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        return (tp - sma) / (0.015 * mad.replace(0, np.nan))

    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Williams %R — oscilador de sobrecompra/sobreventa, rango [-100, 0]."""
        highest_high = df["high"].rolling(period).max()
        lowest_low = df["low"].rolling(period).min()
        return -100 * (highest_high - df["close"]) / (highest_high - lowest_low)

    @staticmethod
    def roc(series: pd.Series, period: int = 12) -> pd.Series:
        """Rate of Change (%)."""
        return (series / series.shift(period) - 1) * 100

    @staticmethod
    def awesome_oscillator(df: pd.DataFrame, fast: int = 5, slow: int = 34) -> pd.Series:
        """Awesome Oscillator (Bill Williams) — momentum sobre el precio medio."""
        median_price = (df["high"] + df["low"]) / 2
        return median_price.rolling(fast).mean() - median_price.rolling(slow).mean()

    @staticmethod
    def ultimate_oscillator(df: pd.DataFrame, period1: int = 7, period2: int = 14, period3: int = 28) -> pd.Series:
        """Ultimate Oscillator — combina 3 horizontes temporales."""
        close, high, low = df["close"], df["high"], df["low"]
        prev_close = close.shift(1)
        bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
        tr = pd.concat([high, prev_close], axis=1).max(axis=1) - pd.concat([low, prev_close], axis=1).min(axis=1)
        avg1 = bp.rolling(period1).sum() / tr.rolling(period1).sum()
        avg2 = bp.rolling(period2).sum() / tr.rolling(period2).sum()
        avg3 = bp.rolling(period3).sum() / tr.rolling(period3).sum()
        return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

    @staticmethod
    def fisher_transform(df: pd.DataFrame, period: int = 9) -> tuple:
        """Fisher Transform — convierte precios en una distribución cercana a la normal (aproximación vectorizada)."""
        median_price = (df["high"] + df["low"]) / 2
        max_h = median_price.rolling(period).max()
        min_l = median_price.rolling(period).min()
        raw = 2 * ((median_price - min_l) / (max_h - min_l).replace(0, np.nan) - 0.5)
        value = raw.clip(-0.999, 0.999).ewm(span=5, adjust=False).mean()
        fisher = 0.5 * np.log((1 + value) / (1 - value))
        return fisher, fisher.shift(1)

    @staticmethod
    def tsi(series: pd.Series, long_period: int = 25, short_period: int = 13) -> pd.Series:
        """True Strength Index."""
        diff = series.diff()
        ema1 = diff.ewm(span=long_period, adjust=False).mean()
        ema2 = ema1.ewm(span=short_period, adjust=False).mean()
        abs_ema1 = diff.abs().ewm(span=long_period, adjust=False).mean()
        abs_ema2 = abs_ema1.ewm(span=short_period, adjust=False).mean()
        return 100 * ema2 / abs_ema2.replace(0, np.nan)

    @staticmethod
    def kst(series: pd.Series, r1: int = 10, r2: int = 15, r3: int = 20, r4: int = 30,
            sma1: int = 10, sma2: int = 10, sma3: int = 10, sma4: int = 15, signal: int = 9) -> tuple:
        """Know Sure Thing. Devuelve (KST, señal)."""
        roc1 = Indicators.roc(series, r1).rolling(sma1).mean()
        roc2 = Indicators.roc(series, r2).rolling(sma2).mean()
        roc3 = Indicators.roc(series, r3).rolling(sma3).mean()
        roc4 = Indicators.roc(series, r4).rolling(sma4).mean()
        kst_line = roc1 * 1 + roc2 * 2 + roc3 * 3 + roc4 * 4
        return kst_line, kst_line.rolling(signal).mean()

    @staticmethod
    def ppo(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Percentage Price Oscillator — como el MACD pero en % (comparable entre activos)."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        ppo_line = 100 * (ema_fast - ema_slow) / ema_slow
        signal_line = ppo_line.ewm(span=signal, adjust=False).mean()
        return ppo_line, signal_line, ppo_line - signal_line

    @staticmethod
    def stoch_rsi(series: pd.Series, rsi_period: int = 14, stoch_period: int = 14,
                  k_smooth: int = 3, d_smooth: int = 3) -> tuple:
        """Stochastic RSI — aplica el oscilador estocástico sobre el RSI."""
        rsi = Indicators.rsi(series, rsi_period)
        min_rsi, max_rsi = rsi.rolling(stoch_period).min(), rsi.rolling(stoch_period).max()
        stoch = 100 * (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan)
        k = stoch.rolling(k_smooth).mean()
        return k, k.rolling(d_smooth).mean()

    # -----------------------------------------------------------------
    # VOLATILIDAD
    # -----------------------------------------------------------------

    @staticmethod
    def keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10,
                          multiplier: float = 2.0) -> tuple:
        """Keltner Channels — bandas basadas en ATR alrededor de una EMA."""
        mid = df["close"].ewm(span=ema_period, adjust=False).mean()
        atr = Indicators.atr(df, atr_period)
        return mid + multiplier * atr, mid, mid - multiplier * atr

    @staticmethod
    def std_dev(series: pd.Series, period: int = 20) -> pd.Series:
        """Desviación estándar móvil."""
        return series.rolling(period).std()

    @staticmethod
    def historical_volatility(series: pd.Series, period: int = 20, trading_periods: int = 252) -> pd.Series:
        """Volatilidad histórica anualizada (%) a partir de retornos logarítmicos."""
        log_ret = np.log(series / series.shift(1))
        return log_ret.rolling(period).std() * np.sqrt(trading_periods) * 100

    @staticmethod
    def natr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR normalizado (%) — permite comparar volatilidad entre activos de distinto precio."""
        return 100 * Indicators.atr(df, period) / df["close"]

    @staticmethod
    def bollinger_percent_b(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """%B — posición del precio dentro de las bandas de Bollinger (0=banda inferior, 1=banda superior)."""
        upper, mid, lower = Indicators.bollinger(series, period, std_dev)
        return (series - lower) / (upper - lower).replace(0, np.nan)

    @staticmethod
    def bollinger_bandwidth(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """Ancho de las bandas de Bollinger relativo a la media — mide expansión/contracción."""
        upper, mid, lower = Indicators.bollinger(series, period, std_dev)
        return (upper - lower) / mid

    # -----------------------------------------------------------------
    # VOLUMEN
    # -----------------------------------------------------------------

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """On Balance Volume."""
        direction = np.sign(df["close"].diff()).fillna(0)
        return (direction * df["volume"]).cumsum()

    @staticmethod
    def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Money Flow Index — RSI ponderado por volumen."""
        tp = (df["high"] + df["low"] + df["close"]) / 3
        raw_flow = tp * df["volume"]
        direction = np.sign(tp.diff()).fillna(0)
        pos_flow = raw_flow.where(direction > 0, 0.0).rolling(period).sum()
        neg_flow = raw_flow.where(direction < 0, 0.0).rolling(period).sum()
        mfr = pos_flow / neg_flow.replace(0, np.nan)
        return 100 - (100 / (1 + mfr))

    @staticmethod
    def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Chaikin Money Flow."""
        mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).replace(0, np.nan)
        mfv = mfm * df["volume"]
        return mfv.rolling(period).sum() / df["volume"].rolling(period).sum()

    @staticmethod
    def ad_line(df: pd.DataFrame) -> pd.Series:
        """Accumulation/Distribution Line."""
        mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).replace(0, np.nan)
        return (mfm.fillna(0) * df["volume"]).cumsum()

    @staticmethod
    def chaikin_oscillator(df: pd.DataFrame, fast: int = 3, slow: int = 10) -> pd.Series:
        """Chaikin Oscillator — MACD aplicado sobre la línea A/D."""
        ad = Indicators.ad_line(df)
        return ad.ewm(span=fast, adjust=False).mean() - ad.ewm(span=slow, adjust=False).mean()

    @staticmethod
    def force_index(df: pd.DataFrame, period: int = 13) -> pd.Series:
        """Force Index — combina precio y volumen para medir la fuerza del movimiento."""
        raw = df["close"].diff() * df["volume"]
        return raw.ewm(span=period, adjust=False).mean()

    @staticmethod
    def eom(df: pd.DataFrame, period: int = 14, volume_divisor: float = 1e8) -> pd.Series:
        """Ease of Movement — cuánto se mueve el precio en relación al volumen."""
        distance = (df["high"] + df["low"]) / 2 - (df["high"].shift(1) + df["low"].shift(1)) / 2
        box_ratio = (df["volume"] / volume_divisor) / (df["high"] - df["low"]).replace(0, np.nan)
        return (distance / box_ratio).rolling(period).mean()

    @staticmethod
    def pvt(df: pd.DataFrame) -> pd.Series:
        """Price Volume Trend."""
        return (df["close"].pct_change() * df["volume"]).cumsum()

    @staticmethod
    def volume_oscillator(series: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
        """Oscilador de volumen (%) — diferencia entre dos medias móviles de volumen."""
        fast_ma, slow_ma = series.rolling(fast).mean(), series.rolling(slow).mean()
        return 100 * (fast_ma - slow_ma) / slow_ma

    # -----------------------------------------------------------------
    # OTROS
    # -----------------------------------------------------------------

    @staticmethod
    def pivot_points(df: pd.DataFrame) -> tuple:
        """
        Pivot points clásicos calculados fila a fila (high/low/close de esa misma barra).
        Para pivots de la sesión ANTERIOR aplicados a la actual, usa .shift(1) sobre el resultado.
        Devuelve (pivot, r1, s1, r2, s2, r3, s3).
        """
        pivot = (df["high"] + df["low"] + df["close"]) / 3
        r1, s1 = 2 * pivot - df["low"], 2 * pivot - df["high"]
        r2, s2 = pivot + (df["high"] - df["low"]), pivot - (df["high"] - df["low"])
        r3, s3 = df["high"] + 2 * (pivot - df["low"]), df["low"] - 2 * (df["high"] - pivot)
        return pivot, r1, s1, r2, s2, r3, s3

    @staticmethod
    def zscore(series: pd.Series, period: int = 20) -> pd.Series:
        """Z-score móvil — cuántas desviaciones estándar se aleja el precio de su media."""
        mean, std = series.rolling(period).mean(), series.rolling(period).std()
        return (series - mean) / std.replace(0, np.nan)

    @staticmethod
    def rolling_correlation(series_a: pd.Series, series_b: pd.Series, period: int = 20) -> pd.Series:
        """Correlación móvil entre dos series (ej: activo vs índice)."""
        return series_a.rolling(period).corr(series_b)
