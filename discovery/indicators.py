"""
Aletheia Discovery Engine - Indicator Library
Vectorized technical indicators implemented purely with pandas/numpy
(no TA-Lib dependency, fully portable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def bollinger_bands(series: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, period)
    std = series.rolling(period, min_periods=period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "width": width, "pct_b": pct_b})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int = 14, d_period: int = 3, smooth: int = 3) -> pd.DataFrame:
    lowest_low = low.rolling(k_period, min_periods=k_period).min()
    highest_high = high.rolling(k_period, min_periods=k_period).max()
    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    k = raw_k.rolling(smooth, min_periods=smooth).mean()
    d = k.rolling(d_period, min_periods=d_period).mean()
    return pd.DataFrame({"k": k, "d": d})


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr_atr = atr(high, low, close, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / tr_atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / tr_atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_val})


def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    return series.diff(period)


def roc(series: pd.Series, period: int = 10) -> pd.Series:
    return series.pct_change(period) * 100


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    typical = (high + low + close) / 3
    sma_tp = sma(typical, period)
    mean_dev = typical.rolling(period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    return (typical - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest_high = high.rolling(period, min_periods=period).max()
    lowest_low = low.rolling(period, min_periods=period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)


def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                      period: int = 20, atr_mult: float = 2.0) -> pd.DataFrame:
    mid = ema(close, period)
    band = atr(high, low, close, period) * atr_mult
    return pd.DataFrame({"mid": mid, "upper": mid + band, "lower": mid - band})


def donchian_channels(high: pd.Series, low: pd.Series, period: int = 20) -> pd.DataFrame:
    upper = high.rolling(period, min_periods=period).max()
    lower = low.rolling(period, min_periods=period).min()
    mid = (upper + lower) / 2
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": mid})


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    mean = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std()
    return (series - mean) / std.replace(0, np.nan)


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """SuperTrend estándar (bandas ATR alrededor del punto medio, con
    'memoria' de la banda que sobrevive mientras no haya reversión de
    tendencia). Iterativo por naturaleza (cada banda depende de si la
    anterior sobrevivió), se calcula UNA vez por dataset igual que el resto
    del banco de indicadores, no por individuo evaluado."""
    atr_val = atr(high, low, close, period).to_numpy()
    hl2 = ((high + low) / 2).to_numpy()
    n = len(close)
    upperband = hl2 + multiplier * atr_val
    lowerband = hl2 - multiplier * atr_val
    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    direction = np.ones(n, dtype=np.int8)
    close_v = close.to_numpy()
    for i in range(1, n):
        if np.isnan(atr_val[i]):
            direction[i] = direction[i - 1]
            final_upper[i] = final_upper[i - 1]
            final_lower[i] = final_lower[i - 1]
            continue
        final_upper[i] = (upperband[i] if (upperband[i] < final_upper[i - 1]
                           or close_v[i - 1] > final_upper[i - 1]) else final_upper[i - 1])
        final_lower[i] = (lowerband[i] if (lowerband[i] > final_lower[i - 1]
                           or close_v[i - 1] < final_lower[i - 1]) else final_lower[i - 1])
        if close_v[i] > final_upper[i - 1]:
            direction[i] = 1
        elif close_v[i] < final_lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    line = np.where(direction > 0, final_lower, final_upper)
    return pd.DataFrame({"line": line, "direction": direction}, index=close.index)


def ichimoku(high: pd.Series, low: pd.Series,
             tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52) -> pd.DataFrame:
    """Tenkan-sen / Kijun-sen / nube (Senkou A/B). A propósito NO se
    desplazan los tramos de la nube 26 velas hacia adelante como en el
    gráfico tradicional de Ichimoku (eso introduciría datos futuros en la
    vela actual): aquí la nube representa el valor calculable EN ESE
    MOMENTO con información pasada, para poder usarse como condición de
    entrada sin look-ahead."""
    tenkan = (high.rolling(tenkan_period, min_periods=tenkan_period).max()
              + low.rolling(tenkan_period, min_periods=tenkan_period).min()) / 2
    kijun = (high.rolling(kijun_period, min_periods=kijun_period).max()
             + low.rolling(kijun_period, min_periods=kijun_period).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high.rolling(senkou_b_period, min_periods=senkou_b_period).max()
                + low.rolling(senkou_b_period, min_periods=senkou_b_period).min()) / 2
    return pd.DataFrame({"tenkan": tenkan, "kijun": kijun,
                          "senkou_a": senkou_a, "senkou_b": senkou_b})


def parabolic_sar(high: pd.Series, low: pd.Series,
                   af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """SAR parabólico de Wilder, iterativo (cada punto depende del anterior
    y del punto extremo de la tendencia vigente). Calculado una vez por
    dataset, no por individuo evaluado."""
    high_v, low_v = high.to_numpy(), low.to_numpy()
    n = len(high_v)
    psar = np.zeros(n)
    if n == 0:
        return pd.Series(psar, index=high.index)
    bull = True
    af = af_start
    ep = high_v[0]
    psar[0] = low_v[0]
    for i in range(1, n):
        prev = psar[i - 1]
        psar[i] = prev + af * (ep - prev)
        if bull:
            psar[i] = min(psar[i], low_v[i - 1], low_v[i - 2] if i >= 2 else low_v[i - 1])
        else:
            psar[i] = max(psar[i], high_v[i - 1], high_v[i - 2] if i >= 2 else high_v[i - 1])
        reverse = False
        if bull and low_v[i] < psar[i]:
            bull, reverse = False, True
            psar[i] = ep
            ep = low_v[i]
            af = af_start
        elif not bull and high_v[i] > psar[i]:
            bull, reverse = True, True
            psar[i] = ep
            ep = high_v[i]
            af = af_start
        if not reverse:
            if bull and high_v[i] > ep:
                ep = high_v[i]
                af = min(af + af_increment, af_max)
            elif not bull and low_v[i] < ep:
                ep = low_v[i]
                af = min(af + af_increment, af_max)
    return pd.Series(psar, index=high.index)


def rolling_percentile_rank(series: pd.Series, window: int = 100, min_periods: int = 20) -> pd.Series:
    """Percentil [0, 1] del valor actual dentro de su propia ventana móvil
    de `window` barras — usado para clasificar volatilidad (ATR) alta/baja
    de forma relativa y adaptativa por activo, en vez de un umbral fijo que
    no tiene sentido igual en Forex que en cripto."""
    def _rank(x: np.ndarray) -> float:
        return float((x[-1] >= x).mean())
    return series.rolling(window, min_periods=min_periods).apply(_rank, raw=True)


def compute_indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Precompute a broad bank of indicators once per dataset, at several
    parameterizations, so strategy generation/backtesting can reuse them
    without recomputation (critical for evaluating thousands of strategies)."""
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    feats = pd.DataFrame(index=df.index)

    for p in (5, 10, 20, 50, 100, 200):
        feats[f"sma_{p}"] = sma(c, p)
        feats[f"ema_{p}"] = ema(c, p)

    for p in (7, 14, 21):
        feats[f"rsi_{p}"] = rsi(c, p)

    macd_df = macd(c)
    feats["macd"] = macd_df["macd"]
    feats["macd_signal"] = macd_df["signal"]
    feats["macd_hist"] = macd_df["hist"]

    for p in (14, 21):
        feats[f"atr_{p}"] = atr(h, l, c, p)

    for p in (20,):
        bb = bollinger_bands(c, p)
        feats[f"bb_upper_{p}"] = bb["upper"]
        feats[f"bb_lower_{p}"] = bb["lower"]
        feats[f"bb_mid_{p}"] = bb["mid"]
        feats[f"bb_pctb_{p}"] = bb["pct_b"]
        feats[f"bb_width_{p}"] = bb["width"]

    stoch_df = stochastic(h, l, c)
    feats["stoch_k"] = stoch_df["k"]
    feats["stoch_d"] = stoch_df["d"]

    adx_df = adx(h, l, c)
    feats["adx"] = adx_df["adx"]
    feats["plus_di"] = adx_df["plus_di"]
    feats["minus_di"] = adx_df["minus_di"]

    for p in (10, 20):
        feats[f"mom_{p}"] = momentum(c, p)
        feats[f"roc_{p}"] = roc(c, p)

    feats["cci_20"] = cci(h, l, c, 20)
    feats["willr_14"] = williams_r(h, l, c, 14)

    dc = donchian_channels(h, l, 20)
    feats["donchian_upper_20"] = dc["upper"]
    feats["donchian_lower_20"] = dc["lower"]

    feats["obv"] = obv(c, v)
    feats["zscore_20"] = zscore(c, 20)

    kc = keltner_channels(h, l, c, 20, 2.0)
    feats["keltner_upper_20"] = kc["upper"]
    feats["keltner_lower_20"] = kc["lower"]
    feats["keltner_mid_20"] = kc["mid"]

    st_df = supertrend(h, l, c, 10, 3.0)
    feats["supertrend_line"] = st_df["line"]
    feats["supertrend_dir"] = st_df["direction"]

    ich = ichimoku(h, l)
    feats["ichimoku_tenkan"] = ich["tenkan"]
    feats["ichimoku_kijun"] = ich["kijun"]
    feats["ichimoku_senkou_a"] = ich["senkou_a"]
    feats["ichimoku_senkou_b"] = ich["senkou_b"]

    feats["psar"] = parabolic_sar(h, l)

    feats["atr_percentile_14"] = rolling_percentile_rank(feats["atr_14"], window=100, min_periods=20)

    return feats
