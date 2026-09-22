"""
Aletheia Discovery Engine - Rule Engine
Atomic-condition library spanning many independent indicator families, plus
a DNF (Disjunctive Normal Form) rule representation: a strategy's entry
logic is (clause_1 AND ... ) OR (clause_2 AND ...) OR ..., where each
condition inside a clause can come from ANY indicator family. This is what
makes strategy discovery combinatorial across indicators, not just across
periods of a single fixed template.

Indicator periods used here are FIXED at representative, well-established
values (RSI 14, ADX 14, MACD 12/26/9, Bollinger 20, etc.). This engine's
job is to discover *which indicators combine into a profitable rule* — not
to fine-tune periods, which is handled downstream in the user's dedicated
optimization platform.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Atomic condition specification
# --------------------------------------------------------------------------

@dataclass
class ConditionSpec:
    id: str
    family: str            # indicator family tag, used to enforce cross-family diversity
    bias: str               # "bullish", "bearish", or "neutral" (used only to seed init smartly)
    description: str
    func: Callable[[pd.DataFrame, pd.DataFrame], pd.Series]   # (df, feats) -> bool Series


def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) <= b.shift(1)) & (a > b)


def _cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) >= b.shift(1)) & (a < b)


def _col_or_false(f: pd.DataFrame, col: str) -> pd.Series:
    """Devuelve f[col] si la columna existe, o una Serie de puros False
    (misma longitud/índice) si no. Usado por condiciones que dependen de
    features opcionales (p.ej. régimen de mercado) que no siempre están
    presentes en `feats` — así una regla que las usa simplemente nunca
    dispara en vez de romper toda la evaluación con un KeyError."""
    if col in f.columns:
        return f[col]
    return pd.Series(False, index=f.index)


def build_condition_library() -> dict[str, ConditionSpec]:
    """Programmatically build ~110 atomic conditions across 12 independent
    indicator families. All reference columns produced by
    indicators.compute_indicator_frame(), so no extra computation is needed
    at evaluation time."""
    lib: dict[str, ConditionSpec] = {}

    def add(cid, family, bias, desc, func):
        lib[cid] = ConditionSpec(cid, family, bias, desc, func)

    # ---- RSI (family: RSI) ----------------------------------------------
    for p in (7, 14, 21):
        for lvl in (20, 25, 30):
            add(f"rsi{p}_below_{lvl}", "RSI", "bullish", f"RSI({p}) < {lvl} (sobreventa)",
                (lambda p=p, lvl=lvl: lambda df, f: f[f"rsi_{p}"] < lvl)())
        for lvl in (70, 75, 80):
            add(f"rsi{p}_above_{lvl}", "RSI", "bearish", f"RSI({p}) > {lvl} (sobrecompra)",
                (lambda p=p, lvl=lvl: lambda df, f: f[f"rsi_{p}"] > lvl)())
        add(f"rsi{p}_cross_up_50", "RSI", "bullish", f"RSI({p}) cruza al alza el nivel 50",
            (lambda p=p: lambda df, f: _cross_up(f[f"rsi_{p}"], pd.Series(50.0, index=f.index)))())
        add(f"rsi{p}_cross_down_50", "RSI", "bearish", f"RSI({p}) cruza a la baja el nivel 50",
            (lambda p=p: lambda df, f: _cross_down(f[f"rsi_{p}"], pd.Series(50.0, index=f.index)))())

    # ---- MACD (family: MACD) ---------------------------------------------
    add("macd_cross_up_signal", "MACD", "bullish", "MACD cruza al alza su linea de señal",
        lambda df, f: _cross_up(f["macd"], f["macd_signal"]))
    add("macd_cross_down_signal", "MACD", "bearish", "MACD cruza a la baja su linea de señal",
        lambda df, f: _cross_down(f["macd"], f["macd_signal"]))
    add("macd_hist_positive", "MACD", "bullish", "Histograma MACD > 0",
        lambda df, f: f["macd_hist"] > 0)
    add("macd_hist_negative", "MACD", "bearish", "Histograma MACD < 0",
        lambda df, f: f["macd_hist"] < 0)
    add("macd_above_zero", "MACD", "bullish", "Linea MACD por encima de cero",
        lambda df, f: f["macd"] > 0)
    add("macd_below_zero", "MACD", "bearish", "Linea MACD por debajo de cero",
        lambda df, f: f["macd"] < 0)

    # ---- Moving averages / trend (family: MA) -----------------------------
    ma_pairs = [("sma", 5, 20), ("sma", 10, 50), ("sma", 20, 50), ("sma", 50, 200),
                ("ema", 5, 20), ("ema", 10, 50), ("ema", 20, 50), ("ema", 50, 200)]
    for kind, fast, slow in ma_pairs:
        add(f"{kind}_{fast}_above_{kind}_{slow}", "MA", "bullish",
            f"{kind.upper()}({fast}) por encima de {kind.upper()}({slow})",
            (lambda kind=kind, fast=fast, slow=slow: lambda df, f: f[f"{kind}_{fast}"] > f[f"{kind}_{slow}"])())
        add(f"{kind}_{fast}_below_{kind}_{slow}", "MA", "bearish",
            f"{kind.upper()}({fast}) por debajo de {kind.upper()}({slow})",
            (lambda kind=kind, fast=fast, slow=slow: lambda df, f: f[f"{kind}_{fast}"] < f[f"{kind}_{slow}"])())
        add(f"{kind}_{fast}_cross_up_{kind}_{slow}", "MA", "bullish",
            f"{kind.upper()}({fast}) cruza al alza {kind.upper()}({slow})",
            (lambda kind=kind, fast=fast, slow=slow: lambda df, f: _cross_up(f[f"{kind}_{fast}"], f[f"{kind}_{slow}"]))())
        add(f"{kind}_{fast}_cross_down_{kind}_{slow}", "MA", "bearish",
            f"{kind.upper()}({fast}) cruza a la baja {kind.upper()}({slow})",
            (lambda kind=kind, fast=fast, slow=slow: lambda df, f: _cross_down(f[f"{kind}_{fast}"], f[f"{kind}_{slow}"]))())

    for p in (20, 50, 100, 200):
        add(f"close_above_sma{p}", "MA", "bullish", f"Precio por encima de SMA({p})",
            (lambda p=p: lambda df, f: df["close"] > f[f"sma_{p}"])())
        add(f"close_below_sma{p}", "MA", "bearish", f"Precio por debajo de SMA({p})",
            (lambda p=p: lambda df, f: df["close"] < f[f"sma_{p}"])())

    # ---- ADX / directional trend strength (family: ADX) --------------------
    for thr in (20, 25, 30):
        add(f"adx_above_{thr}", "ADX", "neutral", f"ADX > {thr} (tendencia fuerte)",
            (lambda thr=thr: lambda df, f: f["adx"] > thr)())
        add(f"adx_below_{thr}", "ADX", "neutral", f"ADX < {thr} (mercado en rango)",
            (lambda thr=thr: lambda df, f: f["adx"] < thr)())
    add("plus_di_above_minus_di", "ADX", "bullish", "+DI por encima de -DI",
        lambda df, f: f["plus_di"] > f["minus_di"])
    add("minus_di_above_plus_di", "ADX", "bearish", "-DI por encima de +DI",
        lambda df, f: f["minus_di"] > f["plus_di"])
    add("plus_di_cross_up_minus_di", "ADX", "bullish", "+DI cruza al alza -DI",
        lambda df, f: _cross_up(f["plus_di"], f["minus_di"]))
    add("minus_di_cross_up_plus_di", "ADX", "bearish", "-DI cruza al alza +DI",
        lambda df, f: _cross_up(f["minus_di"], f["plus_di"]))

    # ---- Bollinger Bands (family: Bollinger) --------------------------------
    add("bb_pctb_oversold", "Bollinger", "bullish", "Precio cerca/bajo banda inferior (pctB<0.05)",
        lambda df, f: f["bb_pctb_20"] < 0.05)
    add("bb_pctb_overbought", "Bollinger", "bearish", "Precio cerca/sobre banda superior (pctB>0.95)",
        lambda df, f: f["bb_pctb_20"] > 0.95)
    add("bb_breakout_up", "Bollinger", "bullish", "Precio rompe banda superior de Bollinger",
        lambda df, f: _cross_up(df["close"], f["bb_upper_20"]))
    add("bb_breakout_down", "Bollinger", "bearish", "Precio rompe banda inferior de Bollinger",
        lambda df, f: _cross_down(df["close"], f["bb_lower_20"]))
    add("bb_squeeze", "Bollinger", "neutral", "Bandas de Bollinger estrechas (baja volatilidad, pre-ruptura)",
        lambda df, f: f["bb_width_20"] < f["bb_width_20"].rolling(100, min_periods=20).quantile(0.20))

    # ---- Donchian channel breakout (family: Donchian) -----------------------
    add("donchian_breakout_up", "Donchian", "bullish", "Ruptura del canal de Donchian(20) al alza",
        lambda df, f: df["close"] > f["donchian_upper_20"].shift(1))
    add("donchian_breakout_down", "Donchian", "bearish", "Ruptura del canal de Donchian(20) a la baja",
        lambda df, f: df["close"] < f["donchian_lower_20"].shift(1))

    # ---- Stochastic oscillator (family: Stochastic) --------------------------
    add("stoch_oversold", "Stochastic", "bullish", "Estocastico %K < 20 (sobreventa)",
        lambda df, f: f["stoch_k"] < 20)
    add("stoch_overbought", "Stochastic", "bearish", "Estocastico %K > 80 (sobrecompra)",
        lambda df, f: f["stoch_k"] > 80)
    add("stoch_cross_up", "Stochastic", "bullish", "%K cruza al alza %D",
        lambda df, f: _cross_up(f["stoch_k"], f["stoch_d"]))
    add("stoch_cross_down", "Stochastic", "bearish", "%K cruza a la baja %D",
        lambda df, f: _cross_down(f["stoch_k"], f["stoch_d"]))

    # ---- CCI (family: CCI) ---------------------------------------------------
    for lvl in (100, 150, 200):
        add(f"cci_below_neg{lvl}", "CCI", "bullish", f"CCI(20) < -{lvl} (sobreventa extrema)",
            (lambda lvl=lvl: lambda df, f: f["cci_20"] < -lvl)())
        add(f"cci_above_{lvl}", "CCI", "bearish", f"CCI(20) > {lvl} (sobrecompra extrema)",
            (lambda lvl=lvl: lambda df, f: f["cci_20"] > lvl)())

    # ---- Williams %R (family: WilliamsR) --------------------------------------
    add("willr_oversold", "WilliamsR", "bullish", "Williams %R(14) < -80 (sobreventa)",
        lambda df, f: f["willr_14"] < -80)
    add("willr_overbought", "WilliamsR", "bearish", "Williams %R(14) > -20 (sobrecompra)",
        lambda df, f: f["willr_14"] > -20)

    # ---- Momentum / Rate of Change (family: Momentum) ---------------------------
    for p in (10, 20):
        add(f"mom{p}_positive", "Momentum", "bullish", f"Momentum({p}) > 0",
            (lambda p=p: lambda df, f: f[f"mom_{p}"] > 0)())
        add(f"mom{p}_negative", "Momentum", "bearish", f"Momentum({p}) < 0",
            (lambda p=p: lambda df, f: f[f"mom_{p}"] < 0)())
        add(f"mom{p}_cross_up_zero", "Momentum", "bullish", f"Momentum({p}) cruza al alza cero",
            (lambda p=p: lambda df, f: _cross_up(f[f"mom_{p}"], pd.Series(0.0, index=f.index)))())
        add(f"mom{p}_cross_down_zero", "Momentum", "bearish", f"Momentum({p}) cruza a la baja cero",
            (lambda p=p: lambda df, f: _cross_down(f[f"mom_{p}"], pd.Series(0.0, index=f.index)))())

    # ---- Z-Score mean reversion (family: ZScore) --------------------------------
    for z in (1.5, 2.0, 2.5):
        add(f"zscore_below_neg{z}", "ZScore", "bullish", f"Z-Score(20) < -{z} (precio muy por debajo de su media)",
            (lambda z=z: lambda df, f: f["zscore_20"] < -z)())
        add(f"zscore_above_{z}", "ZScore", "bearish", f"Z-Score(20) > {z} (precio muy por encima de su media)",
            (lambda z=z: lambda df, f: f["zscore_20"] > z)())

    # ---- OBV / volume flow (family: Volume) --------------------------------------
    add("obv_rising", "Volume", "bullish", "OBV en ascenso (ultimas 10 barras)",
        lambda df, f: f["obv"] > f["obv"].shift(10))
    add("obv_falling", "Volume", "bearish", "OBV en descenso (ultimas 10 barras)",
        lambda df, f: f["obv"] < f["obv"].shift(10))
    add("volume_spike", "Volume", "neutral", "Volumen actual > 1.5x su media movil de 20 barras",
        lambda df, f: df["volume"] > df["volume"].rolling(20, min_periods=10).mean() * 1.5)

    # ---- Régimen de mercado (family: Regime) --------------------------------------
    # Columnas opcionales (regime_bullish/regime_bearish/regime_consolidation/
    # regime_confidence/regime_strength) — ver regime/calibration.py:
    # regime_feature_frame(). Si no se mergearon en `feats` (p.ej. contextos que
    # no pasan por la UI del Descubridor), _col_or_false degrada a "nunca
    # dispara" en vez de romper. Con esto el propio genético puede DESCUBRIR en
    # qué régimen conviene una regla, en vez de que el filtro de régimen sea
    # obligatoriamente externo/manual.
    add("regime_is_bullish", "Regime", "bullish", "Régimen actual: Tendencia Alcista",
        lambda df, f: _col_or_false(f, "regime_bullish"))
    add("regime_is_bearish", "Regime", "bearish", "Régimen actual: Tendencia Bajista",
        lambda df, f: _col_or_false(f, "regime_bearish"))
    add("regime_is_consolidation", "Regime", "neutral", "Régimen actual: Consolidación",
        lambda df, f: _col_or_false(f, "regime_consolidation"))
    add("regime_confidence_high", "Regime", "neutral", "Confianza del régimen vigente > 0.6",
        lambda df, f: _col_or_false(f, "regime_confidence") > 0.6)
    add("regime_strength_bullish", "Regime", "bullish", "Fuerza de régimen > 0.3 (sesgo alcista)",
        lambda df, f: _col_or_false(f, "regime_strength") > 0.3)
    add("regime_strength_bearish", "Regime", "bearish", "Fuerza de régimen < -0.3 (sesgo bajista)",
        lambda df, f: _col_or_false(f, "regime_strength") < -0.3)

    # ---- Keltner Channels (family: Keltner) ---------------------------------------
    add("keltner_breakout_up", "Keltner", "bullish", "Precio rompe banda superior de Keltner(20)",
        lambda df, f: _cross_up(df["close"], f["keltner_upper_20"]))
    add("keltner_breakout_down", "Keltner", "bearish", "Precio rompe banda inferior de Keltner(20)",
        lambda df, f: _cross_down(df["close"], f["keltner_lower_20"]))
    add("keltner_inside", "Keltner", "neutral", "Precio dentro del canal de Keltner(20)",
        lambda df, f: (df["close"] <= f["keltner_upper_20"]) & (df["close"] >= f["keltner_lower_20"]))

    # ---- SuperTrend (family: SuperTrend) -------------------------------------------
    add("supertrend_bullish", "SuperTrend", "bullish", "SuperTrend(10, 3.0) en tendencia alcista",
        lambda df, f: f["supertrend_dir"] > 0)
    add("supertrend_bearish", "SuperTrend", "bearish", "SuperTrend(10, 3.0) en tendencia bajista",
        lambda df, f: f["supertrend_dir"] < 0)
    add("supertrend_flip_bullish", "SuperTrend", "bullish", "SuperTrend cambia de bajista a alcista",
        lambda df, f: (f["supertrend_dir"].shift(1) <= 0) & (f["supertrend_dir"] > 0))
    add("supertrend_flip_bearish", "SuperTrend", "bearish", "SuperTrend cambia de alcista a bajista",
        lambda df, f: (f["supertrend_dir"].shift(1) >= 0) & (f["supertrend_dir"] < 0))

    # ---- Ichimoku (family: Ichimoku) ------------------------------------------------
    add("ichimoku_tenkan_cross_up_kijun", "Ichimoku", "bullish", "Tenkan-sen cruza al alza Kijun-sen",
        lambda df, f: _cross_up(f["ichimoku_tenkan"], f["ichimoku_kijun"]))
    add("ichimoku_tenkan_cross_down_kijun", "Ichimoku", "bearish", "Tenkan-sen cruza a la baja Kijun-sen",
        lambda df, f: _cross_down(f["ichimoku_tenkan"], f["ichimoku_kijun"]))
    add("ichimoku_price_above_cloud", "Ichimoku", "bullish", "Precio por encima de la nube (Kumo)",
        lambda df, f: df["close"] > f[["ichimoku_senkou_a", "ichimoku_senkou_b"]].max(axis=1))
    add("ichimoku_price_below_cloud", "Ichimoku", "bearish", "Precio por debajo de la nube (Kumo)",
        lambda df, f: df["close"] < f[["ichimoku_senkou_a", "ichimoku_senkou_b"]].min(axis=1))

    # ---- Parabolic SAR (family: ParabolicSAR) --------------------------------------
    add("psar_bullish", "ParabolicSAR", "bullish", "Parabolic SAR por debajo del precio (tendencia alcista)",
        lambda df, f: f["psar"] < df["close"])
    add("psar_bearish", "ParabolicSAR", "bearish", "Parabolic SAR por encima del precio (tendencia bajista)",
        lambda df, f: f["psar"] > df["close"])
    add("psar_flip_bullish", "ParabolicSAR", "bullish", "Parabolic SAR cambia de encima a debajo del precio",
        lambda df, f: (f["psar"].shift(1) >= df["close"].shift(1)) & (f["psar"] < df["close"]))
    add("psar_flip_bearish", "ParabolicSAR", "bearish", "Parabolic SAR cambia de debajo a encima del precio",
        lambda df, f: (f["psar"].shift(1) <= df["close"].shift(1)) & (f["psar"] > df["close"]))

    # ---- Percentil de volatilidad / régimen de volatilidad (family: VolRegime) -----
    add("vol_percentile_low", "VolRegime", "neutral", "ATR(14) en el 20% mas bajo de los ultimos 100 periodos (baja volatilidad)",
        lambda df, f: f["atr_percentile_14"] < 0.20)
    add("vol_percentile_high", "VolRegime", "neutral", "ATR(14) en el 20% mas alto de los ultimos 100 periodos (alta volatilidad)",
        lambda df, f: f["atr_percentile_14"] > 0.80)

    # ---- Patrones de vela simples (family: Candlestick) ----------------------------
    add("bullish_engulfing", "Candlestick", "bullish", "Envolvente alcista (la vela actual envuelve a la anterior bajista)",
        lambda df, f: (df["close"].shift(1) < df["open"].shift(1)) & (df["close"] > df["open"])
        & (df["close"] >= df["open"].shift(1)) & (df["open"] <= df["close"].shift(1)))
    add("bearish_engulfing", "Candlestick", "bearish", "Envolvente bajista (la vela actual envuelve a la anterior alcista)",
        lambda df, f: (df["close"].shift(1) > df["open"].shift(1)) & (df["close"] < df["open"])
        & (df["open"] >= df["close"].shift(1)) & (df["close"] <= df["open"].shift(1)))
    add("pin_bar_bullish", "Candlestick", "bullish", "Pin bar alcista (mecha inferior larga, cierre en el tercio superior)",
        lambda df, f: ((df[["open", "close"]].min(axis=1) - df["low"])
                       > 2 * (df["high"] - df[["open", "close"]].max(axis=1)).clip(lower=1e-9))
        & (df["close"] > df["open"]))
    add("pin_bar_bearish", "Candlestick", "bearish", "Pin bar bajista (mecha superior larga, cierre en el tercio inferior)",
        lambda df, f: ((df["high"] - df[["open", "close"]].max(axis=1))
                       > 2 * (df[["open", "close"]].min(axis=1) - df["low"]).clip(lower=1e-9))
        & (df["close"] < df["open"]))

    return lib


CONDITION_LIBRARY = build_condition_library()
FAMILY_TAGS = sorted({c.family for c in CONDITION_LIBRARY.values()})


def evaluate_condition(cid: str, df: pd.DataFrame, feats: pd.DataFrame) -> pd.Series:
    spec = CONDITION_LIBRARY[cid]
    return spec.func(df, feats).fillna(False).astype(bool)


# --------------------------------------------------------------------------
# DNF rule representation:  Rule = list[Clause],  Clause = list[condition_id]
# Signal = OR over clauses of (AND over conditions in the clause)
# --------------------------------------------------------------------------

Clause = list[str]
Rule = list[Clause]


def evaluate_rule(rule: Rule, df: pd.DataFrame, feats: pd.DataFrame) -> pd.Series:
    if not rule:
        return pd.Series(False, index=df.index)
    clause_results = []
    for clause in rule:
        if not clause:
            continue
        result = pd.Series(True, index=df.index)
        for cid in clause:
            result &= evaluate_condition(cid, df, feats)
        clause_results.append(result)
    if not clause_results:
        return pd.Series(False, index=df.index)
    out = clause_results[0]
    for r in clause_results[1:]:
        out |= r
    return out


def rule_to_text(rule: Rule) -> str:
    if not rule:
        return "(regla vacia)"
    clause_strs = []
    for clause in rule:
        cond_strs = [CONDITION_LIBRARY[cid].description for cid in clause]
        clause_strs.append(" Y ".join(cond_strs))
    return "  O  ".join(f"({c})" for c in clause_strs)


def rule_signature(rule: Rule) -> str:
    """Canonical, order-independent signature used to detect functional
    duplicates (same set of clauses, regardless of order)."""
    canon = sorted(tuple(sorted(clause)) for clause in rule)
    payload = json.dumps(canon, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def rule_families(rule: Rule) -> set[str]:
    fams = set()
    for clause in rule:
        for cid in clause:
            fams.add(CONDITION_LIBRARY[cid].family)
    return fams


# --------------------------------------------------------------------------
# Random rule generation (used both for GA population init and pure random
# search baselines)
# --------------------------------------------------------------------------

def random_clause(rng: random.Random, bias_pref: Optional[str] = None,
                   min_conditions: int = 1, max_conditions: int = 3,
                   prefer_diverse_families: bool = True) -> Clause:
    n = rng.randint(min_conditions, max_conditions)
    pool = list(CONDITION_LIBRARY.values())
    if bias_pref:
        biased = [c for c in pool if c.bias in (bias_pref, "neutral")]
        if len(biased) >= n:
            pool = biased

    chosen: list[ConditionSpec] = []
    used_families: set[str] = set()
    candidates = pool[:]
    rng.shuffle(candidates)

    for c in candidates:
        if len(chosen) >= n:
            break
        if prefer_diverse_families and c.family in used_families and len(used_families) < n:
            continue
        chosen.append(c)
        used_families.add(c.family)

    if len(chosen) < n:
        remaining = [c for c in candidates if c not in chosen]
        chosen.extend(remaining[: n - len(chosen)])

    return [c.id for c in chosen]


def random_rule(rng: random.Random, bias_pref: Optional[str] = None,
                 min_clauses: int = 1, max_clauses: int = 2,
                 min_conditions: int = 1, max_conditions: int = 3) -> Rule:
    n_clauses = rng.randint(min_clauses, max_clauses)
    return [random_clause(rng, bias_pref, min_conditions, max_conditions) for _ in range(n_clauses)]


def rule_to_json(rule: Rule) -> str:
    return json.dumps(rule)


def rule_from_json(s: str) -> Rule:
    return json.loads(s)
