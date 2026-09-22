"""
Aletheia Discovery Engine - Genetic Discovery Engine
Evolutionary search over the space of DNF trading rules built from the
cross-family atomic condition library in rule_engine.py. This is the actual
"discovery" mechanism: it does NOT sweep periods of fixed templates — it
searches over *which indicators, from which families, combined how* (AND/OR)
produce a profitable, robust entry rule for long and short sides
independently.

Algorithm: steady-state genetic algorithm with tournament selection,
clause-level crossover, multi-operator mutation, elitism, random-immigrant
diversity injection, and a persistent hall-of-fame that survives across
generations (deduplicated by rule signature).
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .rule_engine import (
    CONDITION_LIBRARY, FAMILY_TAGS, Rule, Clause,
    evaluate_rule, rule_to_text, rule_signature, rule_families,
    random_rule, random_clause,
)
from .backtest_engine import run_backtest_signal, BacktestResult
from .indicators import atr as atr_fn

logger = logging.getLogger("aletheia.genetic_discovery")

DEFAULT_RISK = dict(stop_loss_atr=2.0, take_profit_atr=3.0, atr_period=14, trailing_stop_atr=None)
WARMUP_BARS = 210  # covers the longest lookback used in the indicator bank (sma_200)


@dataclass
class Individual:
    long_rule: Rule
    short_rule: Rule
    individual_id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    fitness: float = -999.0
    metrics: Optional[dict] = None
    # fitness combinado train/holdout usado para reproducción y hall of
    # fame; `shared_fitness` es una copia penalizada por aglomeración
    # (niching) usada SOLO para selección/torneo, nunca para reportar ni
    # para admitir en el hall of fame.
    shared_fitness: float = -999.0
    metrics_train: Optional[dict] = None
    metrics_holdout: Optional[dict] = None
    # Desglose walk-forward: metricas y fitness en cada una de las n_folds
    # ventanas secuenciales evaluadas. metrics_train/metrics_holdout se
    # mantienen como alias (primera/ultima ventana) por compatibilidad.
    fold_metrics: Optional[list] = None
    fold_fitnesses: Optional[list] = None
    # Risk genes — evolved jointly with entry logic instead of fixed at DEFAULT_RISK,
    # so the GA can match tight SL/TP to mean-reversion rules and wide SL/TP to
    # trend-following rules instead of forcing every individual through 2.0/3.0 ATR.
    stop_loss_atr: float = DEFAULT_RISK["stop_loss_atr"]
    take_profit_atr: float = DEFAULT_RISK["take_profit_atr"]
    atr_period: int = DEFAULT_RISK["atr_period"]

    def signature(self) -> str:
        return f"{rule_signature(self.long_rule)}|{rule_signature(self.short_rule)}"

    def risk_params(self) -> dict:
        return dict(stop_loss_atr=self.stop_loss_atr, take_profit_atr=self.take_profit_atr,
                    atr_period=self.atr_period, trailing_stop_atr=None)


@dataclass
class GeneticConfig:
    population_size: int = 80
    n_generations: int = 20
    tournament_size: int = 4
    elite_fraction: float = 0.10
    crossover_rate: float = 0.65
    mutation_rate: float = 0.30
    random_immigrant_fraction: float = 0.10
    min_clauses: int = 1
    max_clauses: int = 2
    min_conditions: int = 1
    max_conditions: int = 3
    min_trades: int = 15
    target_trades: int = 150       # trade count considered "healthy frequency" for fitness scoring
    fitness_objective: str = "composite"   # composite | high_winrate_frequency | high_profitability | sharpe | calmar | profit_factor
    allow_short: bool = True
    allow_long: bool = True
    n_jobs: int = -1
    seed: int = 42
    initial_capital: float = 10_000.0
    hall_of_fame_size: int = 60
    # Evolved risk-management search space (SL/TP are now genes, not fixed constants)
    evolve_risk: bool = True
    sl_atr_min: float = 0.8
    sl_atr_max: float = 4.0
    tp_atr_min: float = 1.0
    tp_atr_max: float = 8.0
    risk_mutation_sigma_frac: float = 0.20  # perturbation size as a fraction of each gene's range
    # Costos del simulador rápido — deben coincidir con los defaults del
    # motor real (core.types.BacktestConfig) para que el ranking de fitness
    # durante la evolución sea consistente con la verificación posterior.
    commission: float = 0.001
    slippage: float = 0.0005
    risk_per_trade: float = 0.02
    # Validación fuera de muestra: walk-forward de N ventanas secuenciales
    # (en vez de un único split train/holdout). El fitness final de un
    # individuo es el MÍNIMO de sus fitness en cada ventana — una regla que
    # solo funciona en un régimen de mercado concreto (una ventana buena,
    # otra mala) queda penalizada exactamente como una regla mala, en vez de
    # premiarse por promediar bien. Con n_folds=2 este esquema se reduce al
    # comportamiento anterior (train/holdout). Es el estándar de facto en
    # validación de estrategias (Pardo 1992/2008; Lopez de Prado 2018) — ver
    # docs/discovery_methodology.md.
    use_oos_validation: bool = True
    n_folds: int = 3
    train_fraction: float = 0.75   # vestigial, ya no se usa (se mantiene por compatibilidad de firma)
    min_trades_holdout: Optional[int] = None  # override manual del mínimo de trades POR VENTANA; None => max(5, min_trades // n_folds)
    # Diversidad: sin esto, la población tiende a converger a variaciones de
    # la misma combinación de indicadores (lo que se ve como "muchas
    # estrategias parecidas" en el Hall of Fame).
    niche_similarity_threshold: float = 0.6   # Jaccard >= esto se considera "misma familia de reglas"
    niche_penalty_weight: float = 0.06        # penalización de fitness por vecino similar en la misma generación
    hof_diversity_threshold: float = 0.6      # Jaccard máximo permitido entre dos miembros del Hall of Fame final


def _random_risk(rng: random.Random, cfg: GeneticConfig) -> tuple[float, float]:
    if not cfg.evolve_risk:
        return DEFAULT_RISK["stop_loss_atr"], DEFAULT_RISK["take_profit_atr"]
    sl = round(rng.uniform(cfg.sl_atr_min, cfg.sl_atr_max), 2)
    tp = round(rng.uniform(cfg.tp_atr_min, cfg.tp_atr_max), 2)
    return sl, tp


def _init_individual(rng: random.Random, cfg: GeneticConfig) -> Individual:
    if cfg.allow_long:
        long_rule = random_rule(rng, bias_pref="bullish", min_clauses=cfg.min_clauses,
                                 max_clauses=cfg.max_clauses, min_conditions=cfg.min_conditions,
                                 max_conditions=cfg.max_conditions)
    else:
        long_rule = []
    if cfg.allow_short:
        short_rule = random_rule(rng, bias_pref="bearish", min_clauses=cfg.min_clauses,
                                  max_clauses=cfg.max_clauses, min_conditions=cfg.min_conditions,
                                  max_conditions=cfg.max_conditions)
    else:
        short_rule = []
    sl, tp = _random_risk(rng, cfg)
    return Individual(long_rule=long_rule, short_rule=short_rule, stop_loss_atr=sl, take_profit_atr=tp)


def _combined_signal(ind: Individual, df: pd.DataFrame, feats: pd.DataFrame) -> pd.Series:
    long_active = evaluate_rule(ind.long_rule, df, feats) if ind.long_rule else pd.Series(False, index=df.index)
    short_active = evaluate_rule(ind.short_rule, df, feats) if ind.short_rule else pd.Series(False, index=df.index)
    conflict = long_active & short_active
    long_only = long_active & ~conflict
    short_only = short_active & ~conflict
    signal = pd.Series(0, index=df.index, dtype=int)
    signal[long_only] = 1
    signal[short_only] = -1
    return signal


def composite_fitness(metrics: dict, min_trades: int, target_trades: int) -> float:
    if metrics["n_trades"] < min_trades:
        return -999.0
    norm_sharpe = np.clip(metrics["sharpe"] / 3.0, -1.5, 2.0)
    norm_calmar = np.clip(metrics["calmar"] / 3.0, -1.5, 2.0)
    norm_pf = np.clip((metrics["profit_factor"] - 1.0) / 2.0, -1.5, 2.0)
    norm_wr = np.clip((metrics["win_rate"] - 0.5) * 2, -1.0, 1.0)
    norm_freq = float(np.clip(metrics["n_trades"] / max(target_trades, 1), 0.0, 1.2))
    trade_penalty = -0.3 if metrics["n_trades"] > 3000 else 0.0  # discourage overtrading/noise-fitting
    return float(0.28 * norm_sharpe + 0.17 * norm_calmar + 0.17 * norm_pf
                 + 0.20 * norm_wr + 0.18 * norm_freq + trade_penalty)


def high_winrate_frequency_fitness(metrics: dict, min_trades: int, target_trades: int) -> float:
    """Prioritizes win rate and trade frequency over raw Sharpe — for finding
    strategies that trade often AND win often, even if individual trades are
    modest (useful on lower timeframes like M15 where many small, reliable
    setups are preferable to a few large lucky ones)."""
    if metrics["n_trades"] < min_trades:
        return -999.0
    norm_wr = np.clip((metrics["win_rate"] - 0.5) * 3.0, -1.5, 1.5)
    norm_freq = float(np.clip(metrics["n_trades"] / max(target_trades, 1), 0.0, 1.5))
    norm_pf = np.clip((metrics["profit_factor"] - 1.0) / 2.0, -1.5, 2.0)
    norm_sharpe = np.clip(metrics["sharpe"] / 3.0, -1.0, 1.5)
    trade_penalty = -0.3 if metrics["n_trades"] > 5000 else 0.0
    return float(0.40 * norm_wr + 0.30 * norm_freq + 0.20 * norm_pf + 0.10 * norm_sharpe + trade_penalty)


def high_profitability_fitness(metrics: dict, min_trades: int, target_trades: int) -> float:
    """Optimizes for actual money made per unit of pain endured, instead of
    letting a mediocre-but-frequent strategy "buy" a decent composite score.
    Rewards CAGR, Sortino (downside-only volatility) and Calmar (return per
    unit of max drawdown) directly, and applies an explicit, steep penalty
    as max drawdown grows past comfortable levels — so a strategy with fewer,
    cleaner trades and a controlled drawdown beats a high-frequency strategy
    that bleeds -20% along the way, which composite's soft weighting let slip
    through. Frequency and win rate are still counted, but as tie-breakers,
    not primary drivers."""
    if metrics["n_trades"] < min_trades:
        return -999.0
    dd = abs(metrics["max_drawdown"])  # e.g. 0.20 for -20%
    # Steep, convex drawdown penalty: barely noticeable under 10% DD, severe past 25%.
    dd_penalty = -6.0 * dd ** 2
    norm_cagr = np.clip(metrics["cagr"] / 0.30, -1.5, 2.5)       # 30% CAGR ~ full score
    norm_sortino = np.clip(metrics["sortino"] / 3.0, -1.5, 2.5)
    norm_calmar = np.clip(metrics["calmar"] / 3.0, -1.5, 2.5)
    norm_pf = np.clip((metrics["profit_factor"] - 1.0) / 2.0, -1.0, 1.5)
    norm_freq = float(np.clip(metrics["n_trades"] / max(target_trades, 1), 0.0, 0.6))
    norm_wr = np.clip((metrics["win_rate"] - 0.5) * 1.0, -0.5, 0.5)
    trade_penalty = -0.3 if metrics["n_trades"] > 3000 else 0.0
    return float(0.25 * norm_cagr + 0.25 * norm_sortino + 0.25 * norm_calmar
                 + 0.10 * norm_pf + 0.08 * norm_freq + 0.07 * norm_wr
                 + dd_penalty + trade_penalty)


def _make_fitness_func(objective: str, min_trades: int, target_trades: int) -> Callable[[dict], float]:
    """Construye la función de fitness para una investigación concreta.

    Además de los objetivos históricos se admiten los objetivos de
    investigación multi-activo de CapitalQuant Research Lab. Cada corrida
    sigue teniendo UN objetivo principal: no se mezclan métricas en una sola
    búsqueda salvo que el usuario seleccione explícitamente ``composite``.
    """
    if objective == "composite":
        return lambda m: composite_fitness(m, min_trades, target_trades)
    if objective == "high_winrate_frequency":
        return lambda m: high_winrate_frequency_fitness(m, min_trades, target_trades)
    if objective == "high_profitability":
        return lambda m: high_profitability_fitness(m, min_trades, target_trades)
    if objective == "sharpe":
        return lambda m: m["sharpe"] if m["n_trades"] >= min_trades else -999.0
    if objective == "calmar":
        return lambda m: m["calmar"] if m["n_trades"] >= min_trades else -999.0
    if objective == "profit_factor":
        return lambda m: min(m["profit_factor"], 10.0) if m["n_trades"] >= min_trades else -999.0
    if objective == "win_rate":
        return lambda m: m["win_rate"] if m["n_trades"] >= min_trades else -999.0
    if objective == "cagr":
        return lambda m: m["cagr"] if m["n_trades"] >= min_trades else -999.0
    if objective == "expectancy":
        return lambda m: m["expectancy"] if m["n_trades"] >= min_trades else -999.0
    if objective == "drawdown":
        # max_drawdown es negativo; maximizarlo equivale a buscar el DD más
        # cercano a cero. Se limita para evitar que un caso degenerado gane.
        return lambda m: max(float(m["max_drawdown"]), -1.0) if m["n_trades"] >= min_trades else -999.0
    if objective in {"stability", "robustness"}:
        # Estos dos objetivos se calculan sobre las ventanas WFO en
        # _evaluate_individual(), donde existe la información necesaria.
        # En cada fold devolvemos un fitness neutro; el valor final se
        # reemplaza por el score de estabilidad/robustez de todas las
        # ventanas.
        return lambda m: 0.0 if m["n_trades"] >= min_trades else -999.0
    raise ValueError(f"Objetivo de fitness desconocido: {objective}")


def _stability_fitness(fold_metrics: list[dict]) -> float:
    """Score 0..1 que premia consistencia entre ventanas WFO."""
    if not fold_metrics:
        return -999.0
    sharpes = np.asarray([float(m.get("sharpe", 0.0)) for m in fold_metrics], dtype=float)
    calmars = np.asarray([float(m.get("calmar", 0.0)) for m in fold_metrics], dtype=float)
    positive = float(np.mean((sharpes > 0) & (calmars > 0)))
    dispersion = float(np.std(sharpes)) / (1.0 + abs(float(np.mean(sharpes))))
    return float(positive - min(dispersion, 1.0))


def _robustness_fitness(fold_metrics: list[dict]) -> float:
    """Score que prioriza un suelo de desempeño razonable en todas las ventanas."""
    if not fold_metrics:
        return -999.0
    sharpes = np.asarray([float(m.get("sharpe", 0.0)) for m in fold_metrics], dtype=float)
    calmars = np.asarray([float(m.get("calmar", 0.0)) for m in fold_metrics], dtype=float)
    pfs = np.asarray([min(float(m.get("profit_factor", 0.0)), 10.0) for m in fold_metrics], dtype=float)
    # Percentil bajo: una estrategia robusta no depende de una sola ventana.
    floor = 0.40 * np.percentile(sharpes, 25) + 0.35 * np.percentile(calmars, 25)
    pf_component = 0.25 * np.clip((np.percentile(pfs, 25) - 1.0) / 2.0, -1.0, 2.0)
    return float(floor + pf_component)


def _fold_boundaries(n_bars: int, n_folds: int, warmup: int,
                      regime_active: Optional[np.ndarray] = None) -> list[tuple[int, int, int]]:
    """Divide [0, n_bars) en n_folds ventanas secuenciales contiguas.
    Devuelve (data_start, seg_start, seg_end) por ventana: data_start incluye
    hasta `warmup` barras prestadas de justo antes de la ventana (para que
    los indicadores de esa ventana no arranquen en NaN), seg_start/seg_end
    delimitan la ventana real donde SÍ se permite operar.

    `regime_active` (opcional): array booleano de longitud n_bars, True en
    las velas que sobreviven el filtro de régimen. Cuando se pasa (y hay
    régimen suficiente para dividir en n_folds partes no vacías), los cortes
    de ventana dejan de ser "mismo número de velas de calendario" y pasan a
    ser "mismo número de velas EN RÉGIMEN" — contiguos en el calendario, pero
    calibrados para que cada ventana reciba ~1/n_folds de las ocurrencias del
    régimen elegido, sin importar cómo estén repartidas en el tiempo.

    Por qué importa: el régimen se detecta como segmentos temporales (ver
    regime/service.py), no como una etiqueta uniformemente distribuida — es
    perfectamente normal que un régimen concreto esté concentrado en el 10%
    del historial. Partir el calendario en tercios iguales y aplicar el
    régimen DESPUÉS puede dejar una ventana con casi ninguna vela válida, y
    como el individuo se descarta en cuanto UNA ventana no alcanza el mínimo
    de trades, ningún individuo pasaría nunca — sin importar qué tan buena
    sea la regla. Repartir por densidad de régimen elimina ese falso negativo
    estructural."""
    n_folds = max(1, n_folds)
    if regime_active is not None and n_folds > 1:
        cum = np.cumsum(regime_active.astype(np.int64))
        total_active = int(cum[-1]) if len(cum) else 0
        if total_active >= n_folds:
            bounds = []
            seg_start = 0
            for k in range(1, n_folds):
                target = total_active * k / n_folds
                idx = int(np.searchsorted(cum, target, side="left"))
                idx = max(seg_start + 1, min(idx, n_bars - 1))
                w = min(warmup, seg_start)
                data_start = seg_start - w
                bounds.append((data_start, seg_start, idx))
                seg_start = idx
            w = min(warmup, seg_start)
            data_start = seg_start - w
            bounds.append((data_start, seg_start, n_bars))
            return bounds
        # Régimen demasiado escaso para repartir en n_folds partes no
        # vacías: cae al reparto por calendario de abajo (se seguirá
        # avisando si eso deja alguna ventana sin trades suficientes).

    fold_size = n_bars // n_folds
    bounds = []
    for k in range(n_folds):
        seg_start = k * fold_size
        seg_end = n_bars if k == n_folds - 1 else (k + 1) * fold_size
        w = min(warmup, seg_start)
        data_start = seg_start - w
        bounds.append((data_start, seg_start, seg_end))
    return bounds


def _evaluate_individual(long_rule: Rule, short_rule: Rule, stop_loss_atr: float, take_profit_atr: float,
                          atr_period: int, df: pd.DataFrame, feats: pd.DataFrame,
                          timeframe: str, fitness_key: str, initial_capital: float,
                          min_trades: int, target_trades: int, commission: float, slippage: float,
                          risk_per_trade: float, use_oos: bool, n_folds: int,
                          min_trades_fold_override: Optional[int],
                          regime_mask: Optional[pd.Series] = None) -> Optional[dict]:
    """Picklable worker for parallel fitness evaluation across processes.

    Walk-forward de n_folds ventanas secuenciales (cuando use_oos=True): el
    dataset se parte en n_folds tramos contiguos que cubren TODO el rango
    seleccionado, cada uno con su propio warmup de indicadores prestado del
    tramo anterior. El fitness de un individuo es el MÍNIMO de sus fitness en
    cada ventana — para llegar al Hall of Fame, una regla tiene que
    mantenerse rentable en CADA ventana temporal, no solo en promedio ni
    solo en la más favorable. Esto es lo que la literatura de validación de
    estrategias (Pardo; Lopez de Prado) identifica como el estándar de facto
    frente a un único split train/test, que puede "aprobar" una regla que
    tuvo la suerte de que su única ventana de holdout cayera en un régimen
    favorable.

    Con n_folds<=1 o use_oos=False, se evalúa sobre todo el rango sin split
    (modo rápido, sin garantía de robustez fuera de muestra — pensado para
    iteración exploratoria, no para el Hall of Fame final).

    `regime_mask` (opcional): Serie booleana alineada con `df` (índice
    completo, la misma convención que `regime.service.regime_entry_mask`).
    Cuando se pasa, toda entrada fuera del/los régimen(es) permitido(s) se
    pone a 0 ANTES de contar trades/backtestear — igual que hace el filtro
    de régimen de Backtesting/Optimización (`ui/components/regime_filter.py`).
    El indicador se sigue calculando sobre TODO el historial de cada
    ventana (nunca se recorta el DataFrame), así que ningún warmup ni
    ninguna ventana walk-forward se ve afectado por el filtro — solo se
    descarta la señal en las velas de régimen distinto. Esto es lo que
    permite que el genético busque reglas cuyo desempeño se mida
    ÚNICAMENTE dentro del régimen elegido, en vez de una regla genérica
    evaluada sobre la mezcla completa de regímenes.

    Cuando hay `regime_mask` Y `n_folds>1`, los cortes de cada ventana ya
    NO dividen el calendario en partes iguales: dividen las OCURRENCIAS del
    régimen en partes iguales (ver `_fold_boundaries`). Así, si el régimen
    elegido está concentrado en un tramo del historial, cada ventana igual
    recibe ~1/n_folds de esas velas en vez de que una ventana se quede casi
    sin ninguna solo por coincidencia de calendario.
    """
    ind = Individual(long_rule=long_rule, short_rule=short_rule,
                     stop_loss_atr=stop_loss_atr, take_profit_atr=take_profit_atr, atr_period=atr_period)

    def _run(df_slice: pd.DataFrame, feats_slice: pd.DataFrame, warmup: int,
              min_tr: int, tgt_tr: int) -> Optional[dict]:
        try:
            signal = _combined_signal(ind, df_slice, feats_slice)
        except Exception:
            return None
        if regime_mask is not None:
            mask_slice = regime_mask.reindex(df_slice.index).fillna(False)
            signal = signal.where(mask_slice, 0)
        if int((signal != 0).sum()) == 0:
            # Corte barato antes de simular: sin ninguna vela con señal
            # activa, no puede haber ni un solo trade — no hace falta
            # correr el backtest para saberlo.
            return {"metrics": None, "fitness": -999.0}
        try:
            result = run_backtest_signal(
                ind.individual_id, signal, df_slice, warmup=warmup, timeframe=timeframe,
                initial_capital=initial_capital, commission=commission, slippage=slippage,
                risk_per_trade=risk_per_trade, **ind.risk_params(),
            )
        except Exception:
            return None
        # min_tr se exige sobre TRADES REALES (round-trip: entrada -> salida
        # por SL/TP/señal contraria), no sobre velas con señal activa.
        # Antes se contaba `(signal != 0).sum()`, que para condiciones tipo
        # "nivel" (adx_above_25, rsi_above_50...) sobreconraba enormemente:
        # una condición que se mantiene verdadera 200 velas seguidas
        # producía UN solo trade real (la posición se mantiene abierta
        # mientras la señal no cambie — ver `_simulate` más arriba), pero
        # pasaba el filtro de "mínimo 5 trades" con margen de sobra. Eso
        # dejaba entrar al Hall of Fame reglas con 1-2 trades reales,
        # estadísticamente frágiles, que luego lucían distinto (o directamente
        # se caían) al verificar con el motor real o al reenviarlas a
        # Backtesting — la causa más probable de que "no haya coherencia
        # entre lo encontrado y la prueba en backtest" incluso ya filtrando
        # por régimen en ambos lados.
        if len(result.trades) < min_tr:
            return {"metrics": None, "fitness": -999.0}
        fitness_fn = _make_fitness_func(fitness_key, min_tr, tgt_tr)
        return {"metrics": result.metrics, "fitness": fitness_fn(result.metrics)}

    effective_folds = max(1, n_folds) if use_oos else 1

    if effective_folds <= 1:
        res = _run(df, feats, WARMUP_BARS, min_trades, target_trades)
        if res is None:
            return None
        metrics = res["metrics"]
        fitness_value = res["fitness"]
        if fitness_key == "stability":
            fitness_value = _stability_fitness([metrics])
        elif fitness_key == "robustness":
            fitness_value = _robustness_fitness([metrics])
        return {"metrics": metrics, "fitness": fitness_value,
                "metrics_train": metrics, "metrics_holdout": metrics,
                "fold_metrics": [metrics], "fold_fitnesses": [res["fitness"]]}

    n_bars = len(df)
    regime_active = None
    if regime_mask is not None:
        regime_active = regime_mask.reindex(df.index).fillna(False).to_numpy()
    bounds = _fold_boundaries(n_bars, effective_folds, WARMUP_BARS, regime_active=regime_active)
    min_trades_fold = (min_trades_fold_override if min_trades_fold_override is not None
                        else max(5, min_trades // effective_folds))
    target_trades_fold = max(1, target_trades // effective_folds)

    fold_metrics: list = []
    fold_fitnesses: list = []
    for data_start, seg_start, seg_end in bounds:
        df_slice = df.iloc[data_start:seg_end]
        feats_slice = feats.iloc[data_start:seg_end]
        warmup = seg_start - data_start
        res = _run(df_slice, feats_slice, warmup, min_trades_fold, target_trades_fold)
        if res is None:
            return None
        fold_metrics.append(res["metrics"])
        fold_fitnesses.append(res["fitness"])
        if res["fitness"] <= -999.0:
            # Corte anticipado: una ventana ya invalida al individuo (no
            # necesita rentabilidad, solo min_trades) — evita correr las
            # ventanas restantes de un individuo ya descartado.
            return {"metrics": fold_metrics[-1], "fitness": -999.0,
                    "metrics_train": fold_metrics[0], "metrics_holdout": fold_metrics[-1],
                    "fold_metrics": fold_metrics, "fold_fitnesses": fold_fitnesses}

    overall_fitness = min(fold_fitnesses)
    if fitness_key == "stability":
        overall_fitness = _stability_fitness(fold_metrics)
    elif fitness_key == "robustness":
        overall_fitness = _robustness_fitness(fold_metrics)
    representative_metrics = fold_metrics[-1]  # ventana mas reciente = regimen mas representativo hoy
    return {"metrics": representative_metrics, "fitness": overall_fitness,
            "metrics_train": fold_metrics[0], "metrics_holdout": fold_metrics[-1],
            "fold_metrics": fold_metrics, "fold_fitnesses": fold_fitnesses}


def _tournament_select(rng: random.Random, population: list[Individual], k: int) -> Individual:
    contenders = rng.sample(population, min(k, len(population)))
    return max(contenders, key=lambda ind: ind.shared_fitness)


def _condition_set(ind: Individual) -> frozenset:
    """Conjunto de IDs de condiciones atómicas usadas por un individuo
    (long + short combinados) — la unidad sobre la que medimos diversidad."""
    ids: set = set()
    for clause in ind.long_rule:
        ids.update(clause)
    for clause in ind.short_rule:
        ids.update(clause)
    return frozenset(ids)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def _apply_fitness_sharing(population: list[Individual], threshold: float, penalty_weight: float) -> None:
    """Niching por conteo: penaliza el fitness de selección (no el fitness
    real) de cada individuo en proporción a cuántos otros individuos válidos
    de la MISMA generación ocupan la misma región del espacio de reglas
    (Jaccard de condiciones >= threshold). Esto evita que el torneo/elitismo
    concentre toda la población en una sola combinación de indicadores solo
    porque esa combinación llegó primero a un óptimo local — la razón por la
    que el Hall of Fame mostraba tantas variantes casi idénticas."""
    valid_idx = [i for i, ind in enumerate(population) if ind.fitness > -999.0]
    sets = [_condition_set(ind) for ind in population]
    for i, ind in enumerate(population):
        if ind.fitness <= -999.0:
            ind.shared_fitness = ind.fitness
            continue
        neighbors = 0
        for j in valid_idx:
            if j == i:
                continue
            if _jaccard(sets[i], sets[j]) >= threshold:
                neighbors += 1
        ind.shared_fitness = ind.fitness - penalty_weight * neighbors


def _diversify_hof(candidates: list[Individual], max_size: int, similarity_threshold: float) -> list[Individual]:
    """Selecciona greedy por fitness descendente, pero rechaza un candidato
    si es demasiado similar (Jaccard de condiciones) a alguno ya admitido —
    así el Hall of Fame final que ve el usuario contiene estrategias
    genuinamente distintas en vez de N variaciones de la misma combinación."""
    ordered = sorted(candidates, key=lambda ind: ind.fitness, reverse=True)
    admitted: list[Individual] = []
    admitted_sets: list[frozenset] = []
    for ind in ordered:
        if len(admitted) >= max_size:
            break
        cond_set = _condition_set(ind)
        if any(_jaccard(cond_set, other) >= similarity_threshold for other in admitted_sets):
            continue
        admitted.append(ind)
        admitted_sets.append(cond_set)
    # Si el filtro de diversidad dejó espacio libre (pocos candidatos
    # suficientemente distintos), rellena con los mejores restantes aunque
    # se parezcan, para no devolver un Hall of Fame más chico de lo pedido.
    if len(admitted) < max_size:
        remaining = [ind for ind in ordered if ind not in admitted]
        admitted.extend(remaining[: max_size - len(admitted)])
    return admitted


def _dedupe_clauses(rule: Rule) -> Rule:
    seen = set()
    out = []
    for clause in rule:
        key = tuple(sorted(clause))
        if key and key not in seen:
            seen.add(key)
            out.append(clause)
    return out


def _crossover_rule(rng: random.Random, rule_a: Rule, rule_b: Rule) -> Rule:
    """Clause-level crossover: mix clauses from both parents."""
    all_clauses = rule_a + rule_b
    if not all_clauses:
        return []
    rng.shuffle(all_clauses)
    n = rng.randint(1, max(1, min(2, len(all_clauses))))
    child = _dedupe_clauses(all_clauses[:n]) or [all_clauses[0]]
    return child


def _crossover_risk(rng: random.Random, parent_a: Individual, parent_b: Individual) -> tuple[float, float]:
    """Blend crossover for risk genes (vs. pick-one for rule clauses) since SL/TP
    are continuous and averaging two reasonable values tends to stay reasonable."""
    w = rng.random()
    sl = round(w * parent_a.stop_loss_atr + (1 - w) * parent_b.stop_loss_atr, 2)
    tp = round(w * parent_a.take_profit_atr + (1 - w) * parent_b.take_profit_atr, 2)
    return sl, tp


def _mutate_risk(rng: random.Random, sl: float, tp: float, cfg: GeneticConfig) -> tuple[float, float]:
    if not cfg.evolve_risk:
        return sl, tp
    sl_sigma = (cfg.sl_atr_max - cfg.sl_atr_min) * cfg.risk_mutation_sigma_frac
    tp_sigma = (cfg.tp_atr_max - cfg.tp_atr_min) * cfg.risk_mutation_sigma_frac
    new_sl = float(np.clip(rng.gauss(sl, sl_sigma), cfg.sl_atr_min, cfg.sl_atr_max))
    new_tp = float(np.clip(rng.gauss(tp, tp_sigma), cfg.tp_atr_min, cfg.tp_atr_max))
    return round(new_sl, 2), round(new_tp, 2)


def _mutate_rule(rng: random.Random, rule: Rule, cfg: GeneticConfig, bias_pref: Optional[str]) -> Rule:
    if not rule:
        return random_rule(rng, bias_pref, cfg.min_clauses, cfg.max_clauses,
                            cfg.min_conditions, cfg.max_conditions)
    rule = [clause[:] for clause in rule]
    op = rng.choice(["add_condition", "remove_condition", "replace_condition",
                     "add_clause", "remove_clause", "swap_family"])
    all_ids = list(CONDITION_LIBRARY.keys())

    if op == "add_condition" and rule:
        ci = rng.randrange(len(rule))
        if len(rule[ci]) < cfg.max_conditions:
            rule[ci].append(rng.choice(all_ids))
    elif op == "remove_condition":
        candidates = [c for c in rule if len(c) > cfg.min_conditions]
        if candidates:
            clause = rng.choice(candidates)
            clause.pop(rng.randrange(len(clause)))
    elif op == "replace_condition" and rule:
        ci = rng.randrange(len(rule))
        if rule[ci]:
            pos = rng.randrange(len(rule[ci]))
            rule[ci][pos] = rng.choice(all_ids)
    elif op == "add_clause" and len(rule) < cfg.max_clauses:
        rule.append(random_clause(rng, bias_pref, cfg.min_conditions, cfg.max_conditions))
    elif op == "remove_clause" and len(rule) > cfg.min_clauses:
        rule.pop(rng.randrange(len(rule)))
    elif op == "swap_family" and rule:
        ci = rng.randrange(len(rule))
        if rule[ci]:
            pos = rng.randrange(len(rule[ci]))
            current_family = CONDITION_LIBRARY[rule[ci][pos]].family
            alt = [cid for cid, spec in CONDITION_LIBRARY.items() if spec.family != current_family]
            if alt:
                rule[ci][pos] = rng.choice(alt)

    rule = [list(dict.fromkeys(clause)) for clause in rule if clause]
    rule = _dedupe_clauses(rule)
    if not rule:
        rule = [random_clause(rng, bias_pref, cfg.min_conditions, cfg.max_conditions)]
    return rule


@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    n_valid: int
    hall_of_fame_size: int
    # Snapshot serializable de los mejores candidatos encontrados hasta esta
    # generación. Solo se usa para visualización en vivo.
    top_candidates: list[dict] = field(default_factory=list)


@dataclass
class GeneticDiscoveryResult:
    hall_of_fame: list[Individual]
    generation_stats: list[GenerationStats]
    final_population: list[Individual]
    n_evaluated_total: int
    trial_sharpes: list[float] = field(default_factory=list)


class GeneticDiscoveryEngine:
    def __init__(self, config: Optional[GeneticConfig] = None):
        self.cfg = config or GeneticConfig()

    def run(self, df: pd.DataFrame, feats: pd.DataFrame, timeframe: str = "H1",
            progress_callback: Optional[Callable[[GenerationStats], None]] = None,
            regime_mask: Optional[pd.Series] = None) -> GeneticDiscoveryResult:
        cfg = self.cfg
        rng = random.Random(cfg.seed)

        population = [_init_individual(rng, cfg) for _ in range(cfg.population_size)]
        hall_of_fame: dict[str, Individual] = {}
        stats_history: list[GenerationStats] = []
        n_evaluated_total = 0
        trial_sharpes: list[float] = []

        for generation in range(cfg.n_generations):
            long_rules = [ind.long_rule for ind in population]
            short_rules = [ind.short_rule for ind in population]

            eval_results = Parallel(n_jobs=cfg.n_jobs, backend="loky", batch_size=16)(
                delayed(_evaluate_individual)(lr, sr, ind.stop_loss_atr, ind.take_profit_atr, ind.atr_period,
                                               df, feats, timeframe, cfg.fitness_objective,
                                               cfg.initial_capital, cfg.min_trades, cfg.target_trades,
                                               cfg.commission, cfg.slippage, cfg.risk_per_trade,
                                               cfg.use_oos_validation, cfg.n_folds, cfg.min_trades_holdout,
                                               regime_mask)
                for lr, sr, ind in zip(long_rules, short_rules, population)
            )
            n_evaluated_total += len(eval_results)

            valid_count = 0
            for ind, res in zip(population, eval_results):
                if res is None:
                    ind.fitness = -999.0
                    ind.metrics = None
                    ind.metrics_train = None
                    ind.metrics_holdout = None
                    ind.fold_metrics = None
                    ind.fold_fitnesses = None
                    continue
                ind.fitness = res["fitness"]
                ind.metrics = res["metrics"]
                ind.metrics_train = res.get("metrics_train")
                ind.metrics_holdout = res.get("metrics_holdout")
                ind.fold_metrics = res.get("fold_metrics")
                ind.fold_fitnesses = res.get("fold_fitnesses")
                # Sharpe de CADA ventana evaluada (no solo del ganador) —
                # alimenta el Deflated Sharpe Ratio, que necesita la
                # dispersión de Sharpe entre todos los intentos de la
                # búsqueda para descontar el sesgo de tests múltiples.
                for fm in (ind.fold_metrics or []):
                    if fm and "sharpe" in fm:
                        trial_sharpes.append(fm["sharpe"])
                if ind.fitness > -999.0:
                    valid_count += 1
                    sig = ind.signature()
                    if sig not in hall_of_fame or ind.fitness > hall_of_fame[sig].fitness:
                        hall_of_fame[sig] = ind

            _apply_fitness_sharing(population, cfg.niche_similarity_threshold, cfg.niche_penalty_weight)

            fitnesses = [ind.fitness for ind in population if ind.fitness > -999.0]
            best_fitness = max((ind.fitness for ind in population), default=-999.0)
            mean_fitness = float(np.mean(fitnesses)) if fitnesses else -999.0

            live_candidates = []
            for live_ind in sorted(hall_of_fame.values(), key=lambda x: x.fitness, reverse=True)[:10]:
                lm = live_ind.metrics or {}
                live_candidates.append({
                    "strategy_id": str(live_ind.individual_id),
                    "fitness": float(live_ind.fitness),
                    "sharpe": float(lm.get("sharpe", 0.0) or 0.0),
                    "calmar": float(lm.get("calmar", 0.0) or 0.0),
                    "profit_factor": float(lm.get("profit_factor", 0.0) or 0.0),
                    "win_rate": float(lm.get("win_rate", 0.0) or 0.0),
                    "cagr": float(lm.get("cagr", 0.0) or 0.0),
                    "expectancy": float(lm.get("expectancy", 0.0) or 0.0),
                    "max_drawdown": float(lm.get("max_drawdown", 0.0) or 0.0),
                    "n_trades": int(lm.get("n_trades", 0) or 0),
                    "long_rule": str(live_ind.long_rule),
                    "short_rule": str(live_ind.short_rule),
                    "stop_loss_atr": float(live_ind.stop_loss_atr),
                    "take_profit_atr": float(live_ind.take_profit_atr),
                })
            stat = GenerationStats(generation=generation, best_fitness=best_fitness,
                                    mean_fitness=mean_fitness, n_valid=valid_count,
                                    hall_of_fame_size=len(hall_of_fame),
                                    top_candidates=live_candidates)
            stats_history.append(stat)
            if generation == 0 and valid_count == 0:
                usable_bars = max(len(df) - WARMUP_BARS, 0)
                logger.warning(
                    "Generacion 0: NINGUN individuo alcanzo min_trades=%d (poblacion=%d). "
                    "Barras totales=%d, barras utilizables tras warmup=%d (WARMUP_BARS=%d). "
                    "Si usable_bars/min_trades es bajo, es esperable que TODAS las generaciones "
                    "queden en -999: amplia el rango de fechas, usa temporalidad mas fina, o "
                    "baja min_trades.", cfg.min_trades, cfg.population_size, len(df),
                    usable_bars, WARMUP_BARS,
                )
            if progress_callback:
                progress_callback(stat)
            logger.info("Gen %d: best=%.3f mean=%.3f validos=%d hof=%d",
                        generation, best_fitness, mean_fitness, valid_count, len(hall_of_fame))

            if generation == cfg.n_generations - 1:
                break

            population.sort(key=lambda ind: ind.shared_fitness, reverse=True)
            n_elite = max(1, int(cfg.population_size * cfg.elite_fraction))
            n_immigrants = max(1, int(cfg.population_size * cfg.random_immigrant_fraction))
            n_offspring = cfg.population_size - n_elite - n_immigrants

            next_population = [population[i] for i in range(n_elite)]

            for _ in range(n_offspring):
                parent_a = _tournament_select(rng, population, cfg.tournament_size)
                parent_b = _tournament_select(rng, population, cfg.tournament_size)

                if rng.random() < cfg.crossover_rate:
                    child_long = _crossover_rule(rng, parent_a.long_rule, parent_b.long_rule) if cfg.allow_long else []
                    child_short = _crossover_rule(rng, parent_a.short_rule, parent_b.short_rule) if cfg.allow_short else []
                    child_sl, child_tp = _crossover_risk(rng, parent_a, parent_b)
                else:
                    child_long = [clause[:] for clause in parent_a.long_rule] if cfg.allow_long else []
                    child_short = [clause[:] for clause in parent_a.short_rule] if cfg.allow_short else []
                    child_sl, child_tp = parent_a.stop_loss_atr, parent_a.take_profit_atr

                if cfg.allow_long and rng.random() < cfg.mutation_rate:
                    child_long = _mutate_rule(rng, child_long, cfg, "bullish")
                if cfg.allow_short and rng.random() < cfg.mutation_rate:
                    child_short = _mutate_rule(rng, child_short, cfg, "bearish")
                if rng.random() < cfg.mutation_rate:
                    child_sl, child_tp = _mutate_risk(rng, child_sl, child_tp, cfg)

                next_population.append(Individual(long_rule=child_long, short_rule=child_short,
                                                   stop_loss_atr=child_sl, take_profit_atr=child_tp))

            for _ in range(n_immigrants):
                next_population.append(_init_individual(rng, cfg))

            population = next_population[:cfg.population_size]

        hof_sorted = _diversify_hof(list(hall_of_fame.values()), cfg.hall_of_fame_size, cfg.hof_diversity_threshold)

        return GeneticDiscoveryResult(
            hall_of_fame=hof_sorted, generation_stats=stats_history,
            final_population=population, n_evaluated_total=n_evaluated_total,
            trial_sharpes=trial_sharpes,
        )


def individual_to_definition(ind: Individual) -> dict:
    return {
        "family": "Genetic_Discovered",
        "individual_id": ind.individual_id,
        "long_rule": ind.long_rule,
        "short_rule": ind.short_rule,
        "long_rule_text": rule_to_text(ind.long_rule) if ind.long_rule else "(sin regla long)",
        "short_rule_text": rule_to_text(ind.short_rule) if ind.short_rule else "(sin regla short)",
        "families_used": sorted(rule_families(ind.long_rule) | rule_families(ind.short_rule)),
        **ind.risk_params(),
    }


def signal_from_definition(definition: dict, df: pd.DataFrame, feats: pd.DataFrame) -> pd.Series:
    ind = Individual(long_rule=definition["long_rule"], short_rule=definition["short_rule"],
                      stop_loss_atr=definition.get("stop_loss_atr", DEFAULT_RISK["stop_loss_atr"]),
                      take_profit_atr=definition.get("take_profit_atr", DEFAULT_RISK["take_profit_atr"]),
                      atr_period=definition.get("atr_period", DEFAULT_RISK["atr_period"]))
    return _combined_signal(ind, df, feats)


def build_execution_frame(definition: dict, df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Construye un DataFrame con columnas 'signal', 'stop_loss' y 'take_profit'
    usando EXACTAMENTE la misma fórmula que discovery/codegen.py hornea dentro
    del método `generate_signals` de la estrategia exportada (mismo ATR —Wilder,
    alpha=1/period—, mismo punto de anclaje en `close`, mismo múltiplo evolucionado
    de stop_loss_atr/take_profit_atr). Esto permite verificar un individuo del
    Hall of Fame contra engine/backtester.py con la certeza de que el resultado
    coincide con lo que producirá la estrategia una vez exportada y ejecutada en
    la página Backtesting real — no es una aproximación aparte.
    """
    signal = signal_from_definition(definition, df, feats)
    atr_period = int(definition.get("atr_period", DEFAULT_RISK["atr_period"]))
    sl_atr = float(definition.get("stop_loss_atr", DEFAULT_RISK["stop_loss_atr"]))
    tp_atr = float(definition.get("take_profit_atr", DEFAULT_RISK["take_profit_atr"]))

    atr_risk = atr_fn(df["high"], df["low"], df["close"], atr_period)

    out = df.copy()
    out["signal"] = signal.reindex(out.index).fillna(0).astype(int)
    out["stop_loss"] = np.nan
    out["take_profit"] = np.nan

    long_entries = out["signal"] == 1
    short_entries = out["signal"] == -1
    out.loc[long_entries, "stop_loss"] = out.loc[long_entries, "close"] - sl_atr * atr_risk[long_entries]
    out.loc[long_entries, "take_profit"] = out.loc[long_entries, "close"] + tp_atr * atr_risk[long_entries]
    out.loc[short_entries, "stop_loss"] = out.loc[short_entries, "close"] + sl_atr * atr_risk[short_entries]
    out.loc[short_entries, "take_profit"] = out.loc[short_entries, "close"] - tp_atr * atr_risk[short_entries]
    return out
