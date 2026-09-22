"""
Aletheia Discovery Engine - Code Generator
Translates a discovered rule (long_rule / short_rule, DNF over the atomic
condition library in rule_engine.py) into a standalone Python strategy class
matching the `BaseStrategy` contract used by the user's external
optimization platform (Constructor de Estrategias):

    class MiEstrategia(BaseStrategy):
        name = "..."
        description = "..."
        def __init__(self, <param>=<default>, **kwargs): ...
        def generate_signals(self, data): ...     # returns df con 'signal',
                                                    # 'stop_loss' y 'take_profit'
                                                    # (niveles en precio, en
                                                    # multiplos de ATR — igual
                                                    # que evaluo el genetico)
        def get_param_space(self): ...             # {param: ("int", lo, hi)}
        def get_indicator_columns(self): ...        # [column names]

Only indicator PERIODS (and a couple of directly analogous numeric knobs)
are exposed as tunable constructor parameters / param_space entries — exact
thresholds discovered by the genetic search are kept as literals, so the
exported strategy preserves the exact logic that was found, and the external
platform's job is purely to retune periods per asset.

Generated code is fully self-contained (pure pandas/numpy formulas mirroring
core/indicators.py) — it does not depend on any `Indicators` helper class
from the external platform, since its exact API surface is unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .rule_engine import CONDITION_LIBRARY, Rule, rule_to_text


@dataclass
class Setup:
    column: str
    params: dict[str, float]         # param_name -> default value
    code_lines: list[str]            # lines that compute df[column] (may use self.<param_name>)
    depends_on: list[str] = field(default_factory=list)   # other setup columns this one needs


def _param_range(default: float) -> tuple[str, float, float]:
    if isinstance(default, float) and not float(default).is_integer():
        lo = max(0.5, round(default * 0.4, 2))
        hi = round(default * 3.0, 2)
        return ("float", lo, hi)
    default = int(default)
    lo = max(2, int(default * 0.3))
    hi = max(lo + 1, int(default * 4))
    return ("int", lo, hi)


# --------------------------------------------------------------------------
# Setup registry: one entry per indicator column that might be needed.
# Built once; conditions reference these by column name and we deduplicate
# automatically when composing a strategy (several conditions often share
# the same underlying indicator column).
# --------------------------------------------------------------------------

def _sma_setup(period: int) -> Setup:
    p = f"sma_{period}_period"
    col = f"sma_{period}"
    return Setup(col, {p: period}, [f"df['{col}'] = df['close'].rolling(self.{p}).mean()"])


def _ema_setup(period: int) -> Setup:
    p = f"ema_{period}_period"
    col = f"ema_{period}"
    return Setup(col, {p: period},
                 [f"df['{col}'] = df['close'].ewm(span=self.{p}, adjust=False, min_periods=self.{p}).mean()"])


def _rsi_setup(period: int) -> Setup:
    p = f"rsi_{period}_period"
    col = f"rsi_{period}"
    code = [
        f"_delta = df['close'].diff()",
        f"_gain = _delta.clip(lower=0.0)",
        f"_loss = -_delta.clip(upper=0.0)",
        f"_avg_gain = _gain.ewm(alpha=1/self.{p}, min_periods=self.{p}, adjust=False).mean()",
        f"_avg_loss = _loss.ewm(alpha=1/self.{p}, min_periods=self.{p}, adjust=False).mean()",
        f"_rs = _avg_gain / _avg_loss.replace(0, float('nan'))",
        f"df['{col}'] = (100 - (100 / (1 + _rs))).fillna(50.0)",
    ]
    return Setup(col, {p: period}, code)


def _macd_setup() -> list[Setup]:
    params = {"macd_fast": 12, "macd_slow": 26, "macd_signal_period": 9}
    code = [
        "_ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False, min_periods=self.macd_fast).mean()",
        "_ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False, min_periods=self.macd_slow).mean()",
        "df['macd'] = _ema_fast - _ema_slow",
        "df['macd_signal'] = df['macd'].ewm(span=self.macd_signal_period, adjust=False, "
        "min_periods=self.macd_signal_period).mean()",
        "df['macd_hist'] = df['macd'] - df['macd_signal']",
    ]
    return [
        Setup("macd", params, code),
        Setup("macd_signal", {}, [], depends_on=["macd"]),
        Setup("macd_hist", {}, [], depends_on=["macd"]),
    ]


def _atr_setup(period: int, param_name: str) -> Setup:
    col = f"atr_{param_name}"
    code = [
        f"_prev_close = df['close'].shift(1)",
        f"_tr = pd.concat([df['high']-df['low'], (df['high']-_prev_close).abs(), "
        f"(df['low']-_prev_close).abs()], axis=1).max(axis=1)",
        f"df['{col}'] = _tr.ewm(alpha=1/self.{param_name}, min_periods=self.{param_name}, adjust=False).mean()",
    ]
    return Setup(col, {param_name: period}, code)


def _adx_setup(period: int = 14) -> list[Setup]:
    p = "adx_period"
    atr_setup = _atr_setup(period, p)
    code = [
        "_up_move = df['high'].diff()",
        "_down_move = -df['low'].diff()",
        "_plus_dm = _up_move.where((_up_move > _down_move) & (_up_move > 0), 0.0)",
        "_minus_dm = _down_move.where((_down_move > _up_move) & (_down_move > 0), 0.0)",
        f"_plus_di = 100 * _plus_dm.ewm(alpha=1/self.{p}, min_periods=self.{p}, adjust=False).mean() "
        f"/ df['{atr_setup.column}'].replace(0, float('nan'))",
        f"_minus_di = 100 * _minus_dm.ewm(alpha=1/self.{p}, min_periods=self.{p}, adjust=False).mean() "
        f"/ df['{atr_setup.column}'].replace(0, float('nan'))",
        "_dx = 100 * (_plus_di - _minus_di).abs() / (_plus_di + _minus_di).replace(0, float('nan'))",
        f"df['adx'] = _dx.ewm(alpha=1/self.{p}, min_periods=self.{p}, adjust=False).mean()",
        "df['plus_di'] = _plus_di",
        "df['minus_di'] = _minus_di",
    ]
    adx_setup = Setup("adx", {p: period}, code, depends_on=[atr_setup.column])
    return [atr_setup, adx_setup,
            Setup("plus_di", {}, [], depends_on=["adx"]),
            Setup("minus_di", {}, [], depends_on=["adx"])]


def _bollinger_setup(period: int = 20) -> list[Setup]:
    p, std_p = "bollinger_period", "bollinger_std"
    mid_col = f"bb_mid_{period}"
    code = [
        f"df['{mid_col}'] = df['close'].rolling(self.{p}).mean()",
        f"_bb_std = df['close'].rolling(self.{p}).std()",
        f"df['bb_upper_{period}'] = df['{mid_col}'] + self.{std_p} * _bb_std",
        f"df['bb_lower_{period}'] = df['{mid_col}'] - self.{std_p} * _bb_std",
        f"df['bb_pctb_{period}'] = (df['close'] - df['bb_lower_{period}']) / "
        f"(df['bb_upper_{period}'] - df['bb_lower_{period}']).replace(0, float('nan'))",
        f"df['bb_width_{period}'] = (df['bb_upper_{period}'] - df['bb_lower_{period}']) / "
        f"df['{mid_col}'].replace(0, float('nan'))",
    ]
    main = Setup(mid_col, {p: period, std_p: 2.0}, code)
    return [main] + [
        Setup(f"bb_upper_{period}", {}, [], depends_on=[mid_col]),
        Setup(f"bb_lower_{period}", {}, [], depends_on=[mid_col]),
        Setup(f"bb_pctb_{period}", {}, [], depends_on=[mid_col]),
        Setup(f"bb_width_{period}", {}, [], depends_on=[mid_col]),
    ]


def _donchian_setup(period: int = 20) -> list[Setup]:
    p = "donchian_period"
    upper_col, lower_col = f"donchian_upper_{period}", f"donchian_lower_{period}"
    code = [
        f"df['{upper_col}'] = df['high'].rolling(self.{p}).max()",
        f"df['{lower_col}'] = df['low'].rolling(self.{p}).min()",
    ]
    return [Setup(upper_col, {p: period}, code), Setup(lower_col, {}, [], depends_on=[upper_col])]


def _stochastic_setup(k_period: int = 14, d_period: int = 3, smooth: int = 3) -> list[Setup]:
    pk, pd_, ps = "stoch_k_period", "stoch_d_period", "stoch_smooth"
    code = [
        f"_lowest_low = df['low'].rolling(self.{pk}).min()",
        f"_highest_high = df['high'].rolling(self.{pk}).max()",
        f"_raw_k = 100 * (df['close'] - _lowest_low) / (_highest_high - _lowest_low).replace(0, float('nan'))",
        f"df['stoch_k'] = _raw_k.rolling(self.{ps}).mean()",
        f"df['stoch_d'] = df['stoch_k'].rolling(self.{pd_}).mean()",
    ]
    return [Setup("stoch_k", {pk: k_period, ps: smooth, pd_: d_period}, code),
            Setup("stoch_d", {}, [], depends_on=["stoch_k"])]


def _cci_setup(period: int = 20) -> Setup:
    p = "cci_period"
    col = "cci_20"
    code = [
        "_typical = (df['high'] + df['low'] + df['close']) / 3",
        f"_sma_tp = _typical.rolling(self.{p}).mean()",
        f"_mean_dev = _typical.rolling(self.{p}).apply(lambda x: (x - x.mean()).abs().mean(), raw=False)",
        f"df['{col}'] = (_typical - _sma_tp) / (0.015 * _mean_dev.replace(0, float('nan')))",
    ]
    return Setup(col, {p: period}, code)


def _willr_setup(period: int = 14) -> Setup:
    p = "willr_period"
    col = "willr_14"
    code = [
        f"_highest_high = df['high'].rolling(self.{p}).max()",
        f"_lowest_low = df['low'].rolling(self.{p}).min()",
        f"df['{col}'] = -100 * (_highest_high - df['close']) / (_highest_high - _lowest_low).replace(0, float('nan'))",
    ]
    return Setup(col, {p: period}, code)


def _mom_setup(period: int) -> Setup:
    p = f"mom_{period}_period"
    col = f"mom_{period}"
    return Setup(col, {p: period}, [f"df['{col}'] = df['close'].diff(self.{p})"])


def _zscore_setup(period: int = 20) -> Setup:
    p = "zscore_period"
    col = "zscore_20"
    code = [
        f"_mean = df['close'].rolling(self.{p}).mean()",
        f"_std = df['close'].rolling(self.{p}).std()",
        f"df['{col}'] = (df['close'] - _mean) / _std.replace(0, float('nan'))",
    ]
    return Setup(col, {p: period}, code)


def _obv_setup() -> Setup:
    code = [
        "_direction = df['close'].diff().fillna(0.0).apply(lambda x: (x > 0) - (x < 0))",
        "df['obv'] = (_direction * df['volume']).cumsum()",
    ]
    return Setup("obv", {}, code)


def _volume_spike_setup(window: int = 20) -> Setup:
    p = "volume_spike_window"
    col = "volume_ma"
    return Setup(col, {p: window}, [f"df['{col}'] = df['volume'].rolling(self.{p}, min_periods=1).mean()"])


def _keltner_setup(period: int = 20, mult: float = 2.0) -> list[Setup]:
    p, m = "keltner_period", "keltner_mult"
    mid_col = f"keltner_mid_{period}"
    upper_col, lower_col = f"keltner_upper_{period}", f"keltner_lower_{period}"
    code = [
        f"df['{mid_col}'] = df['close'].ewm(span=self.{p}, adjust=False, min_periods=self.{p}).mean()",
        "_prev_close_kc = df['close'].shift(1)",
        "_tr_kc = pd.concat([df['high']-df['low'], (df['high']-_prev_close_kc).abs(), "
        "(df['low']-_prev_close_kc).abs()], axis=1).max(axis=1)",
        f"_atr_kc = _tr_kc.ewm(alpha=1/self.{p}, min_periods=self.{p}, adjust=False).mean()",
        f"df['{upper_col}'] = df['{mid_col}'] + self.{m} * _atr_kc",
        f"df['{lower_col}'] = df['{mid_col}'] - self.{m} * _atr_kc",
    ]
    main = Setup(mid_col, {p: period, m: mult}, code)
    return [main,
            Setup(upper_col, {}, [], depends_on=[mid_col]),
            Setup(lower_col, {}, [], depends_on=[mid_col])]


def _supertrend_setup(period: int = 10, mult: float = 3.0) -> list[Setup]:
    p, m = "supertrend_period", "supertrend_mult"
    atr_setup = _atr_setup(period, p)
    code = [
        f"df['supertrend_dir'] = _supertrend_direction(df['close'].to_numpy(), df['high'].to_numpy(), "
        f"df['low'].to_numpy(), df['{atr_setup.column}'].to_numpy(), self.{m})",
    ]
    st_setup = Setup("supertrend_dir", {p: period, m: mult}, code, depends_on=[atr_setup.column])
    return [atr_setup, st_setup]


def _ichimoku_setup(tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> list[Setup]:
    pt, pk, pb = "ichimoku_tenkan_period", "ichimoku_kijun_period", "ichimoku_senkou_b_period"
    code = [
        f"df['ichimoku_tenkan'] = (df['high'].rolling(self.{pt}).max() + df['low'].rolling(self.{pt}).min()) / 2",
        f"df['ichimoku_kijun'] = (df['high'].rolling(self.{pk}).max() + df['low'].rolling(self.{pk}).min()) / 2",
        "df['ichimoku_senkou_a'] = (df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2",
        f"df['ichimoku_senkou_b'] = (df['high'].rolling(self.{pb}).max() + df['low'].rolling(self.{pb}).min()) / 2",
    ]
    main = Setup("ichimoku_tenkan", {pt: tenkan, pk: kijun, pb: senkou_b}, code)
    return [main,
            Setup("ichimoku_kijun", {}, [], depends_on=["ichimoku_tenkan"]),
            Setup("ichimoku_senkou_a", {}, [], depends_on=["ichimoku_tenkan"]),
            Setup("ichimoku_senkou_b", {}, [], depends_on=["ichimoku_tenkan"])]


def _psar_setup() -> Setup:
    code = ["df['psar'] = _parabolic_sar(df['high'].to_numpy(), df['low'].to_numpy())"]
    return Setup("psar", {}, code)


def _atr_percentile_setup(period: int = 14, window: int = 100) -> list[Setup]:
    p, w = "atr_pct_period", "atr_pct_window"
    atr_setup = _atr_setup(period, p)
    col = "atr_percentile_14"
    code = [
        "_rank_fn_apct = lambda x: float((x[-1] >= x).mean())",
        f"df['{col}'] = df['{atr_setup.column}'].rolling(self.{w}, min_periods=20).apply(_rank_fn_apct, raw=True)",
    ]
    pct_setup = Setup(col, {w: window}, code, depends_on=[atr_setup.column])
    return [atr_setup, pct_setup]


def _regime_setup(timeframe: str, overrides: Optional[dict]) -> list[Setup]:
    """Recalcula el régimen de mercado con la MISMA calibración congelada
    que se usó al descubrir la regla (`regime_calib_overrides`, guardado en
    la definición por ui/views/discovery.py), reusando directamente
    `regime.service.compute_regime_summary` — no se reimplementa el
    detector aquí para evitar divergencias con el resto de CapitalQuant."""
    ov = overrides or {}
    code = [
        "from config.settings import TIMEFRAMES",
        "from regime.service import compute_regime_summary, REGIME_LABELS",
        f"_ann_factor_rg = TIMEFRAMES.get({timeframe!r}, {{}}).get('ann_factor', 252)",
        f"_rg_overrides = {ov!r}",
        "_rg_summary = compute_regime_summary(df, ann_factor=_ann_factor_rg, overrides=_rg_overrides)",
        "df['regime_bullish'] = (_rg_summary.regime_series == REGIME_LABELS[0]).fillna(False)",
        "df['regime_bearish'] = (_rg_summary.regime_series == REGIME_LABELS[1]).fillna(False)",
        "df['regime_consolidation'] = (_rg_summary.regime_series == REGIME_LABELS[2]).fillna(False)",
        "df['regime_confidence'] = _rg_summary.confidence_series.reindex(df.index)",
        "df['regime_strength'] = _rg_summary.strength_series.reindex(df.index)",
    ]
    main = Setup("regime_bullish", {}, code)
    aliases = ["regime_bearish", "regime_consolidation", "regime_confidence", "regime_strength"]
    return [main] + [Setup(c, {}, [], depends_on=["regime_bullish"]) for c in aliases]


# --------------------------------------------------------------------------
# Per-condition codegen: expr string (pandas boolean expression referencing
# df[...] columns and, where relevant, literal thresholds discovered by the
# genetic search) + which Setup(s) it depends on.
# --------------------------------------------------------------------------

def _cond_codegen(cid: str, regime_overrides: Optional[dict] = None,
                   timeframe: str = "") -> tuple[str, list[Setup]]:
    """Return (boolean_expr, [Setup,...]) for a given condition id. Mirrors
    the exact structure/values used to build that id in rule_engine.py.

    `regime_overrides`/`timeframe` solo se usan por las condiciones de la
    familia "Regime" (ver `_regime_setup`); el resto de familias los ignora."""
    spec = CONDITION_LIBRARY[cid]

    # ---- RSI ----
    if cid.startswith("rsi") and "_below_" in cid:
        p, lvl = cid.replace("rsi", "").split("_below_")
        return f"(df['rsi_{p}'] < {lvl})", [_rsi_setup(int(p))]
    if cid.startswith("rsi") and "_above_" in cid:
        p, lvl = cid.replace("rsi", "").split("_above_")
        return f"(df['rsi_{p}'] > {lvl})", [_rsi_setup(int(p))]
    if cid.startswith("rsi") and cid.endswith("_cross_up_50"):
        p = cid.replace("rsi", "").replace("_cross_up_50", "")
        return (f"((df['rsi_{p}'].shift(1) <= 50) & (df['rsi_{p}'] > 50))", [_rsi_setup(int(p))])
    if cid.startswith("rsi") and cid.endswith("_cross_down_50"):
        p = cid.replace("rsi", "").replace("_cross_down_50", "")
        return (f"((df['rsi_{p}'].shift(1) >= 50) & (df['rsi_{p}'] < 50))", [_rsi_setup(int(p))])

    # ---- MACD ----
    if cid == "macd_cross_up_signal":
        return ("((df['macd'].shift(1) <= df['macd_signal'].shift(1)) & (df['macd'] > df['macd_signal']))",
                _macd_setup())
    if cid == "macd_cross_down_signal":
        return ("((df['macd'].shift(1) >= df['macd_signal'].shift(1)) & (df['macd'] < df['macd_signal']))",
                _macd_setup())
    if cid == "macd_hist_positive":
        return "(df['macd_hist'] > 0)", _macd_setup()
    if cid == "macd_hist_negative":
        return "(df['macd_hist'] < 0)", _macd_setup()
    if cid == "macd_above_zero":
        return "(df['macd'] > 0)", _macd_setup()
    if cid == "macd_below_zero":
        return "(df['macd'] < 0)", _macd_setup()

    # ---- Moving averages ----
    if "_above_" in cid and (cid.startswith("sma_") or cid.startswith("ema_")):
        kind, rest = cid.split("_", 1)
        fast, slow = rest.replace(f"{kind}_", "").split("_above_")
        slow = slow.replace(f"{kind}_", "")
        setup_fn = _sma_setup if kind == "sma" else _ema_setup
        return (f"(df['{kind}_{fast}'] > df['{kind}_{slow}'])",
                [setup_fn(int(fast)), setup_fn(int(slow))])
    if "_below_" in cid and (cid.startswith("sma_") or cid.startswith("ema_")):
        kind, rest = cid.split("_", 1)
        fast, slow = rest.replace(f"{kind}_", "").split("_below_")
        slow = slow.replace(f"{kind}_", "")
        setup_fn = _sma_setup if kind == "sma" else _ema_setup
        return (f"(df['{kind}_{fast}'] < df['{kind}_{slow}'])",
                [setup_fn(int(fast)), setup_fn(int(slow))])
    if "_cross_up_" in cid and (cid.startswith("sma_") or cid.startswith("ema_")):
        kind, rest = cid.split("_", 1)
        fast, slow = rest.replace(f"{kind}_", "").split("_cross_up_")
        slow = slow.replace(f"{kind}_", "")
        setup_fn = _sma_setup if kind == "sma" else _ema_setup
        return (f"((df['{kind}_{fast}'].shift(1) <= df['{kind}_{slow}'].shift(1)) & "
                f"(df['{kind}_{fast}'] > df['{kind}_{slow}']))",
                [setup_fn(int(fast)), setup_fn(int(slow))])
    if "_cross_down_" in cid and (cid.startswith("sma_") or cid.startswith("ema_")):
        kind, rest = cid.split("_", 1)
        fast, slow = rest.replace(f"{kind}_", "").split("_cross_down_")
        slow = slow.replace(f"{kind}_", "")
        setup_fn = _sma_setup if kind == "sma" else _ema_setup
        return (f"((df['{kind}_{fast}'].shift(1) >= df['{kind}_{slow}'].shift(1)) & "
                f"(df['{kind}_{fast}'] < df['{kind}_{slow}']))",
                [setup_fn(int(fast)), setup_fn(int(slow))])
    if cid.startswith("close_above_sma"):
        p = cid.replace("close_above_sma", "")
        return f"(df['close'] > df['sma_{p}'])", [_sma_setup(int(p))]
    if cid.startswith("close_below_sma"):
        p = cid.replace("close_below_sma", "")
        return f"(df['close'] < df['sma_{p}'])", [_sma_setup(int(p))]

    # ---- ADX ----
    if cid.startswith("adx_above_"):
        thr = cid.replace("adx_above_", "")
        return f"(df['adx'] > {thr})", _adx_setup()
    if cid.startswith("adx_below_"):
        thr = cid.replace("adx_below_", "")
        return f"(df['adx'] < {thr})", _adx_setup()
    if cid == "plus_di_above_minus_di":
        return "(df['plus_di'] > df['minus_di'])", _adx_setup()
    if cid == "minus_di_above_plus_di":
        return "(df['minus_di'] > df['plus_di'])", _adx_setup()
    if cid == "plus_di_cross_up_minus_di":
        return ("((df['plus_di'].shift(1) <= df['minus_di'].shift(1)) & (df['plus_di'] > df['minus_di']))",
                _adx_setup())
    if cid == "minus_di_cross_up_plus_di":
        return ("((df['minus_di'].shift(1) <= df['plus_di'].shift(1)) & (df['minus_di'] > df['plus_di']))",
                _adx_setup())

    # ---- Bollinger ----
    if cid == "bb_pctb_oversold":
        return "(df['bb_pctb_20'] < 0.05)", _bollinger_setup()
    if cid == "bb_pctb_overbought":
        return "(df['bb_pctb_20'] > 0.95)", _bollinger_setup()
    if cid == "bb_breakout_up":
        return ("((df['close'].shift(1) <= df['bb_upper_20'].shift(1)) & (df['close'] > df['bb_upper_20']))",
                _bollinger_setup())
    if cid == "bb_breakout_down":
        return ("((df['close'].shift(1) >= df['bb_lower_20'].shift(1)) & (df['close'] < df['bb_lower_20']))",
                _bollinger_setup())
    if cid == "bb_squeeze":
        return ("(df['bb_width_20'] < df['bb_width_20'].rolling(100, min_periods=20).quantile(0.20))",
                _bollinger_setup())

    # ---- Donchian ----
    if cid == "donchian_breakout_up":
        return "(df['close'] > df['donchian_upper_20'].shift(1))", _donchian_setup()
    if cid == "donchian_breakout_down":
        return "(df['close'] < df['donchian_lower_20'].shift(1))", _donchian_setup()

    # ---- Stochastic ----
    if cid == "stoch_oversold":
        return "(df['stoch_k'] < 20)", _stochastic_setup()
    if cid == "stoch_overbought":
        return "(df['stoch_k'] > 80)", _stochastic_setup()
    if cid == "stoch_cross_up":
        return ("((df['stoch_k'].shift(1) <= df['stoch_d'].shift(1)) & (df['stoch_k'] > df['stoch_d']))",
                _stochastic_setup())
    if cid == "stoch_cross_down":
        return ("((df['stoch_k'].shift(1) >= df['stoch_d'].shift(1)) & (df['stoch_k'] < df['stoch_d']))",
                _stochastic_setup())

    # ---- CCI ----
    if cid.startswith("cci_below_neg"):
        lvl = cid.replace("cci_below_neg", "")
        return f"(df['cci_20'] < -{lvl})", [_cci_setup()]
    if cid.startswith("cci_above_"):
        lvl = cid.replace("cci_above_", "")
        return f"(df['cci_20'] > {lvl})", [_cci_setup()]

    # ---- Williams %R ----
    if cid == "willr_oversold":
        return "(df['willr_14'] < -80)", [_willr_setup()]
    if cid == "willr_overbought":
        return "(df['willr_14'] > -20)", [_willr_setup()]

    # ---- Momentum ----
    if cid.startswith("mom") and cid.endswith("_positive"):
        p = cid.replace("mom", "").replace("_positive", "")
        return f"(df['mom_{p}'] > 0)", [_mom_setup(int(p))]
    if cid.startswith("mom") and cid.endswith("_negative"):
        p = cid.replace("mom", "").replace("_negative", "")
        return f"(df['mom_{p}'] < 0)", [_mom_setup(int(p))]
    if cid.startswith("mom") and cid.endswith("_cross_up_zero"):
        p = cid.replace("mom", "").replace("_cross_up_zero", "")
        return (f"((df['mom_{p}'].shift(1) <= 0) & (df['mom_{p}'] > 0))", [_mom_setup(int(p))])
    if cid.startswith("mom") and cid.endswith("_cross_down_zero"):
        p = cid.replace("mom", "").replace("_cross_down_zero", "")
        return (f"((df['mom_{p}'].shift(1) >= 0) & (df['mom_{p}'] < 0))", [_mom_setup(int(p))])

    # ---- Z-Score ----
    if cid.startswith("zscore_below_neg"):
        z = cid.replace("zscore_below_neg", "")
        return f"(df['zscore_20'] < -{z})", [_zscore_setup()]
    if cid.startswith("zscore_above_"):
        z = cid.replace("zscore_above_", "")
        return f"(df['zscore_20'] > {z})", [_zscore_setup()]

    # ---- Volume / OBV ----
    if cid == "obv_rising":
        return "(df['obv'] > df['obv'].shift(10))", [_obv_setup()]
    if cid == "obv_falling":
        return "(df['obv'] < df['obv'].shift(10))", [_obv_setup()]
    if cid == "volume_spike":
        return "(df['volume'] > df['volume_ma'] * 1.5)", [_volume_spike_setup()]

    # ---- Régimen de mercado ----
    if cid == "regime_is_bullish":
        return "(df['regime_bullish'])", _regime_setup(timeframe, regime_overrides)
    if cid == "regime_is_bearish":
        return "(df['regime_bearish'])", _regime_setup(timeframe, regime_overrides)
    if cid == "regime_is_consolidation":
        return "(df['regime_consolidation'])", _regime_setup(timeframe, regime_overrides)
    if cid == "regime_confidence_high":
        return "(df['regime_confidence'] > 0.6)", _regime_setup(timeframe, regime_overrides)
    if cid == "regime_strength_bullish":
        return "(df['regime_strength'] > 0.3)", _regime_setup(timeframe, regime_overrides)
    if cid == "regime_strength_bearish":
        return "(df['regime_strength'] < -0.3)", _regime_setup(timeframe, regime_overrides)

    # ---- Keltner ----
    if cid == "keltner_breakout_up":
        return ("((df['close'].shift(1) <= df['keltner_upper_20'].shift(1)) & "
                "(df['close'] > df['keltner_upper_20']))", _keltner_setup(20, 2.0))
    if cid == "keltner_breakout_down":
        return ("((df['close'].shift(1) >= df['keltner_lower_20'].shift(1)) & "
                "(df['close'] < df['keltner_lower_20']))", _keltner_setup(20, 2.0))
    if cid == "keltner_inside":
        return ("((df['close'] <= df['keltner_upper_20']) & (df['close'] >= df['keltner_lower_20']))",
                _keltner_setup(20, 2.0))

    # ---- SuperTrend ----
    if cid == "supertrend_bullish":
        return "(df['supertrend_dir'] > 0)", _supertrend_setup(10, 3.0)
    if cid == "supertrend_bearish":
        return "(df['supertrend_dir'] < 0)", _supertrend_setup(10, 3.0)
    if cid == "supertrend_flip_bullish":
        return ("((df['supertrend_dir'].shift(1) <= 0) & (df['supertrend_dir'] > 0))",
                _supertrend_setup(10, 3.0))
    if cid == "supertrend_flip_bearish":
        return ("((df['supertrend_dir'].shift(1) >= 0) & (df['supertrend_dir'] < 0))",
                _supertrend_setup(10, 3.0))

    # ---- Ichimoku ----
    if cid == "ichimoku_tenkan_cross_up_kijun":
        return ("((df['ichimoku_tenkan'].shift(1) <= df['ichimoku_kijun'].shift(1)) & "
                "(df['ichimoku_tenkan'] > df['ichimoku_kijun']))", _ichimoku_setup())
    if cid == "ichimoku_tenkan_cross_down_kijun":
        return ("((df['ichimoku_tenkan'].shift(1) >= df['ichimoku_kijun'].shift(1)) & "
                "(df['ichimoku_tenkan'] < df['ichimoku_kijun']))", _ichimoku_setup())
    if cid == "ichimoku_price_above_cloud":
        return ("(df['close'] > df[['ichimoku_senkou_a', 'ichimoku_senkou_b']].max(axis=1))",
                _ichimoku_setup())
    if cid == "ichimoku_price_below_cloud":
        return ("(df['close'] < df[['ichimoku_senkou_a', 'ichimoku_senkou_b']].min(axis=1))",
                _ichimoku_setup())

    # ---- Parabolic SAR ----
    if cid == "psar_bullish":
        return "(df['psar'] < df['close'])", [_psar_setup()]
    if cid == "psar_bearish":
        return "(df['psar'] > df['close'])", [_psar_setup()]
    if cid == "psar_flip_bullish":
        return ("((df['psar'].shift(1) >= df['close'].shift(1)) & (df['psar'] < df['close']))",
                [_psar_setup()])
    if cid == "psar_flip_bearish":
        return ("((df['psar'].shift(1) <= df['close'].shift(1)) & (df['psar'] > df['close']))",
                [_psar_setup()])

    # ---- Percentil de volatilidad ----
    if cid == "vol_percentile_low":
        return "(df['atr_percentile_14'] < 0.20)", _atr_percentile_setup()
    if cid == "vol_percentile_high":
        return "(df['atr_percentile_14'] > 0.80)", _atr_percentile_setup()

    # ---- Patrones de vela (formulas directas sobre OHLC, sin Setup) ----
    if cid == "bullish_engulfing":
        return ("((df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open']) & "
                "(df['close'] >= df['open'].shift(1)) & (df['open'] <= df['close'].shift(1)))", [])
    if cid == "bearish_engulfing":
        return ("((df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open']) & "
                "(df['open'] >= df['close'].shift(1)) & (df['close'] <= df['open'].shift(1)))", [])
    if cid == "pin_bar_bullish":
        return ("(((df[['open','close']].min(axis=1) - df['low']) > "
                "2 * (df['high'] - df[['open','close']].max(axis=1)).clip(lower=1e-9)) & "
                "(df['close'] > df['open']))", [])
    if cid == "pin_bar_bearish":
        return ("(((df['high'] - df[['open','close']].max(axis=1)) > "
                "2 * (df[['open','close']].min(axis=1) - df['low']).clip(lower=1e-9)) & "
                "(df['close'] < df['open']))", [])

    raise ValueError(f"No hay codegen registrado para la condicion '{cid}' ({spec.description})")


# --------------------------------------------------------------------------
# Rule -> pandas boolean expression (DNF: OR of ANDs)
# --------------------------------------------------------------------------

def _rule_to_expr(rule: Rule, setups: dict[str, Setup], regime_overrides: Optional[dict] = None,
                   timeframe: str = "") -> Optional[str]:
    if not rule:
        return None
    clause_exprs = []
    for clause in rule:
        if not clause:
            continue
        cond_exprs = []
        for cid in clause:
            expr, cond_setups = _cond_codegen(cid, regime_overrides=regime_overrides, timeframe=timeframe)
            cond_exprs.append(expr)
            for s in cond_setups:
                if s.column not in setups:
                    setups[s.column] = s
        clause_exprs.append("(" + " & ".join(cond_exprs) + ")")
    if not clause_exprs:
        return None
    if len(clause_exprs) == 1:
        return clause_exprs[0]
    return "(\n        " + "\n        | ".join(clause_exprs) + "\n    )"


def _order_setups(setups: dict[str, Setup]) -> list[Setup]:
    """Topological-ish ordering so dependent setups (e.g. bb_upper depends
    on bb_mid being computed first) come after what they depend on."""
    ordered: list[Setup] = []
    emitted: set[str] = set()

    def _emit(col: str):
        if col in emitted or col not in setups:
            return
        s = setups[col]
        for dep in s.depends_on:
            _emit(dep)
        if col not in emitted:
            ordered.append(s)
            emitted.add(col)

    for col in setups:
        _emit(col)
    return ordered


def generate_strategy_code(definition: dict, class_name: str = "DescubiertaAletheia",
                            symbol: str = "", timeframe: str = "") -> str:
    """Translate a Genetic_Discovered strategy definition (as produced by
    genetic_discovery.individual_to_definition) into a standalone Python
    class matching the external platform's BaseStrategy contract."""
    long_rule: Rule = definition.get("long_rule", [])
    short_rule: Rule = definition.get("short_rule", [])
    regime_overrides = definition.get("regime_calib_overrides")

    setups: dict[str, Setup] = {}
    long_expr = _rule_to_expr(long_rule, setups, regime_overrides=regime_overrides, timeframe=timeframe)
    short_expr = _rule_to_expr(short_rule, setups, regime_overrides=regime_overrides, timeframe=timeframe)

    ordered_setups = _order_setups(setups)

    all_params: dict[str, float] = {}
    for s in ordered_setups:
        for pname, pdefault in s.params.items():
            all_params[pname] = pdefault

    # Riesgo (SL/TP) evolucionado por el genetico junto con la regla — se
    # expone tambien como parametro tuneable, por si se quiere reoptimizar
    # desde la pagina de Optimizacion de CapitalQuant.
    risk_stop_loss_atr = float(definition.get("stop_loss_atr", 2.0))
    risk_take_profit_atr = float(definition.get("take_profit_atr", 3.0))
    risk_atr_period = int(definition.get("atr_period", 14))
    all_params["stop_loss_atr"] = risk_stop_loss_atr
    all_params["take_profit_atr"] = risk_take_profit_atr
    all_params["sl_tp_atr_period"] = risk_atr_period

    init_args = ", ".join(f"{name}={_fmt(val)}" for name, val in all_params.items())
    init_signature = f"def __init__(self{', ' + init_args if init_args else ''}, **kwargs):"
    super_kwargs = ", ".join(f"{name}={name}" for name in all_params) if all_params else ""
    super_call = f"super().__init__({super_kwargs}, **kwargs)" if super_kwargs else "super().__init__(**kwargs)"
    self_assignments = "\n        ".join(f"self.{name} = {name}" for name in all_params) or "pass"

    setup_code_lines: list[str] = []
    for s in ordered_setups:
        setup_code_lines.extend(s.code_lines)
    setup_code = "\n        ".join(setup_code_lines) if setup_code_lines else "pass"

    signal_lines = []
    if long_expr and short_expr:
        signal_lines.append(f"_long_condition = {long_expr}")
        signal_lines.append(f"_short_condition = {short_expr}")
        signal_lines.append("_conflict = _long_condition & _short_condition  # ambas activas a la vez: se anulan")
        signal_lines.append("df.loc[_long_condition & ~_conflict, 'signal'] = 1")
        signal_lines.append("df.loc[_short_condition & ~_conflict, 'signal'] = -1")
    elif long_expr:
        signal_lines.append(f"_long_condition = {long_expr}")
        signal_lines.append("df.loc[_long_condition, 'signal'] = 1")
    elif short_expr:
        signal_lines.append(f"_short_condition = {short_expr}")
        signal_lines.append("df.loc[_short_condition, 'signal'] = -1")
    signal_code = "\n        ".join(signal_lines) if signal_lines else "# (sin condiciones definidas)"

    risk_code = (
        "_prev_close_risk = df['close'].shift(1)\n"
        "        _tr_risk = pd.concat([df['high'] - df['low'], (df['high'] - _prev_close_risk).abs(), "
        "(df['low'] - _prev_close_risk).abs()], axis=1).max(axis=1)\n"
        "        df['_atr_risk'] = _tr_risk.ewm(alpha=1/self.sl_tp_atr_period, "
        "min_periods=self.sl_tp_atr_period, adjust=False).mean()\n"
        "        df['stop_loss'] = np.nan\n"
        "        df['take_profit'] = np.nan\n"
        "        _long_entries = df['signal'] == 1\n"
        "        _short_entries = df['signal'] == -1\n"
        "        df.loc[_long_entries, 'stop_loss'] = df['close'] - self.stop_loss_atr * df['_atr_risk']\n"
        "        df.loc[_long_entries, 'take_profit'] = df['close'] + self.take_profit_atr * df['_atr_risk']\n"
        "        df.loc[_short_entries, 'stop_loss'] = df['close'] + self.stop_loss_atr * df['_atr_risk']\n"
        "        df.loc[_short_entries, 'take_profit'] = df['close'] - self.take_profit_atr * df['_atr_risk']"
    )

    param_space_lines = []
    for name, default in all_params.items():
        kind, lo, hi = _param_range(default)
        param_space_lines.append(f'            "{name}": ("{kind}", {_fmt(lo)}, {_fmt(hi)}),')
    param_space_code = "\n".join(param_space_lines) if param_space_lines else "            pass"

    indicator_columns = [s.column for s in ordered_setups if s.code_lines]
    indicator_columns_code = ", ".join(f'"{c}"' for c in indicator_columns)

    families = definition.get("families_used", [])
    long_text = definition.get("long_rule_text", "(sin regla long)")
    short_text = definition.get("short_rule_text", "(sin regla short)")

    header = f'''"""
Estrategia exportada desde Aletheia Discovery Engine.
Simbolo de origen: {symbol or "N/A"}   Temporalidad de origen: {timeframe or "N/A"}
Familias de indicadores combinadas: {", ".join(families) if families else "N/A"}

Regla LONG:  {long_text}
Regla SHORT: {short_text}

Los periodos de cada indicador quedan expuestos como parametros tuneables
(get_param_space) para que esta plataforma los optimice por activo. Los
umbrales/niveles descubiertos por el algoritmo genetico se dejan fijos tal
como fueron encontrados, preservando la logica exacta descubierta.
"""
import pandas as pd
import numpy as np

'''

    if "supertrend_dir" in setups:
        header += '''
def _supertrend_direction(close, high, low, atr_val, mult):
    """Direccion de SuperTrend (1 alcista, -1 bajista), iterativa: cada
    banda 'sobrevive' mientras no haya reversion de tendencia (misma logica
    que discovery/indicators.py:supertrend)."""
    n = len(close)
    hl2 = (high + low) / 2.0
    upperband = hl2 + mult * atr_val
    lowerband = hl2 - mult * atr_val
    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    direction = np.ones(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(atr_val[i]):
            direction[i] = direction[i - 1]
            final_upper[i] = final_upper[i - 1]
            final_lower[i] = final_lower[i - 1]
            continue
        final_upper[i] = (upperband[i] if (upperband[i] < final_upper[i - 1]
                          or close[i - 1] > final_upper[i - 1]) else final_upper[i - 1])
        final_lower[i] = (lowerband[i] if (lowerband[i] > final_lower[i - 1]
                          or close[i - 1] < final_lower[i - 1]) else final_lower[i - 1])
        if close[i] > final_upper[i - 1]:
            direction[i] = 1
        elif close[i] < final_lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    return direction

'''

    if "psar" in setups:
        header += '''
def _parabolic_sar(high, low, af_start=0.02, af_increment=0.02, af_max=0.2):
    """SAR parabolico de Wilder, iterativo (misma logica que
    discovery/indicators.py:parabolic_sar)."""
    n = len(high)
    psar = np.zeros(n)
    if n == 0:
        return psar
    bull = True
    af = af_start
    ep = high[0]
    psar[0] = low[0]
    for i in range(1, n):
        prev = psar[i - 1]
        psar[i] = prev + af * (ep - prev)
        if bull:
            psar[i] = min(psar[i], low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
        else:
            psar[i] = max(psar[i], high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
        reverse = False
        if bull and low[i] < psar[i]:
            bull, reverse = False, True
            psar[i] = ep
            ep = low[i]
            af = af_start
        elif not bull and high[i] > psar[i]:
            bull, reverse = True, True
            psar[i] = ep
            ep = high[i]
            af = af_start
        if not reverse:
            if bull and high[i] > ep:
                ep = high[i]
                af = min(af + af_increment, af_max)
            elif not bull and low[i] < ep:
                ep = low[i]
                af = min(af + af_increment, af_max)
    return psar

'''


    code = header + f'''class {class_name}(BaseStrategy):
    name = "{class_name}"
    description = "Estrategia descubierta por Aletheia ({", ".join(families) if families else "N/A"})"

    {init_signature}
        {super_call}
        {self_assignments}

    def generate_signals(self, data):
        df = data.copy()
        {setup_code}

        df["signal"] = 0
        {signal_code}

        {risk_code}
        return df

    def get_param_space(self):
        return {{
{param_space_code}
        }}

    def get_indicator_columns(self):
        return [{indicator_columns_code}]
'''
    return code


def _fmt(val) -> str:
    if isinstance(val, float):
        return repr(round(val, 6))
    return repr(val)


def compile_definition_to_class(definition: dict, class_name: str = "ValidacionTemporal",
                                 symbol: str = "", timeframe: str = ""):
    """
    Genera y compila una definición del Hall of Fame a una clase
    BaseStrategy real, EN MEMORIA (sin escribir a disco), para pruebas
    puntuales como la validación de generalización.

    Importante: NO pasa por `strategies.code_compiler.compile_strategy_code`
    (el sandbox regex+AST del Constructor de Estrategias) porque ese
    sandbox existe para texto libre tecleado por un humano y rechaza
    cualquier `import` — y el código que produce `generate_strategy_code`
    siempre trae `import pandas as pd` / `import numpy as np` de cabecera.
    Es el mismo nivel de confianza que ya usan `save_user_strategy_file` +
    `refresh_registry()` (strategies/__init__.py): código generado
    programáticamente por este módulo, no texto arbitrario de un usuario.
    """
    from strategies.base import BaseStrategy, Indicators
    import pandas as pd
    import numpy as np

    code = generate_strategy_code(definition, class_name=class_name, symbol=symbol, timeframe=timeframe)
    namespace = {"pd": pd, "np": np, "BaseStrategy": BaseStrategy, "Indicators": Indicators}
    exec(compile(code, f"<discovery_codegen_{class_name}>", "exec"), namespace)
    return namespace[class_name]
