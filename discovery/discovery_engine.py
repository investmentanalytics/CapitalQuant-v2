"""
discovery/discovery_engine.py
Orquesta el pipeline de DESCUBRIMIENTO (genético / cruzado entre familias de
indicadores), donde "descubrir" significa encontrar nuevas combinaciones de
indicadores (vía genetic_discovery.py), no barrer periodos de una plantilla
fija:

  1. Evoluciona una población de reglas DNF (genetic_discovery) a través de
     generaciones, combinando condiciones atómicas de familias de
     indicadores potencialmente distintas con AND/OR.
  2. Mantiene un hall-of-fame de las mejores reglas, funcionalmente
     distintas, encontradas a lo largo de todas las generaciones.
  3. Devuelve ese leaderboard para que la UI lo muestre y el usuario decida
     cuáles enviar a Backtesting (motor real de CapitalQuant).

A propósito, este módulo YA NO hace walk-forward, Monte Carlo, bootstrap ni
persistencia en base de datos propia — eso es responsabilidad exclusiva de
CapitalQuant (páginas Backtesting/Optimización) una vez la estrategia se
exporta como un BaseStrategy real. Duplicarlo aquí sería redundante.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .genetic_discovery import (
    GeneticDiscoveryEngine, GeneticConfig, GenerationStats, individual_to_definition,
    build_execution_frame, WARMUP_BARS,
)

from core.types import BacktestConfig
from core.metrics import deflated_sharpe_ratio, annualization_factor
from core.multiple_testing import benjamini_hochberg, dsr_to_pvalue
from engine.backtester import BacktestEngine

logger = logging.getLogger("capitalquant.discovery_engine")


@dataclass
class DiscoveryConfig:
    population_size: int = 100
    n_generations: int = 25
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
    target_trades: int = 150
    fitness_objective: str = "composite"
    allow_short: bool = True
    allow_long: bool = True
    n_jobs: int = -1
    seed: int = 42
    initial_capital: float = 10_000.0
    hall_of_fame_size: int = 60
    # Espacio de búsqueda evolutivo para SL/TP (los genes de riesgo se
    # evolucionan junto con la regla en vez de quedar fijos)
    evolve_risk: bool = True
    sl_atr_min: float = 0.8
    sl_atr_max: float = 4.0
    tp_atr_min: float = 1.0
    tp_atr_max: float = 8.0

    # Validación fuera de muestra: walk-forward de N ventanas secuenciales
    # (ver GeneticConfig para el detalle). Un individuo solo llega al Hall
    # of Fame si se mantiene rentable en CADA ventana, no solo en promedio.
    use_oos_validation: bool = True
    n_folds: int = 3
    min_trades_holdout: Optional[int] = None
    # Diversidad — ver GeneticConfig.
    niche_similarity_threshold: float = 0.6
    niche_penalty_weight: float = 0.06
    hof_diversity_threshold: float = 0.6

    # ------------------------------------------------------------------
    # Verificación con el motor real (engine/backtester.py)
    # ------------------------------------------------------------------
    # Tras la búsqueda evolutiva, cada individuo del Hall of Fame se vuelve a
    # correr con el motor de backtesting REAL de CapitalQuant (mismo que usa
    # la página Backtesting), sobre el mismo rango de fechas, para que las
    # métricas que ve el usuario aquí sean las que va a obtener al enviar la
    # estrategia a Backtesting — no una estimación del simulador rápido
    # interno. Los defaults de abajo replican los defaults de la página
    # Backtesting (ui/views/backtesting.py); si el usuario piensa correr el
    # backtest real con otra comisión/capital/riesgo, puede ajustarlos aquí
    # para que la verificación sea comparable.
    verify_with_real_engine: bool = True
    verify_initial_capital: float = 100_000.0
    verify_commission: float = 0.001     # 0.10%, default de la página Backtesting
    verify_slippage: float = 0.0005      # 0.05%, default de BacktestConfig
    verify_risk_per_trade: float = 0.02  # 2%, default de la página Backtesting
    verify_allow_long: bool = True

    # ------------------------------------------------------------------
    # FDR — False Discovery Rate (Benjamini-Hochberg), ver
    # core/multiple_testing.py. Se aplica DESPUÉS del DSR, sobre el
    # conjunto completo del Hall of Fame: el DSR corrige cada estrategia
    # por cuántas combinaciones probó el genético para llegar a ella; el
    # FDR corrige el Hall of Fame completo por estar mirando muchas
    # estrategias candidatas simultáneamente y preguntar cuántas de ellas
    # son ruido. alpha=0.10 es el nivel convencional para descubrimiento
    # exploratorio (tolera un 10% de falsos descubrimientos esperados
    # entre las declaradas significativas).
    fdr_alpha: float = 0.10

    def to_genetic_config(self) -> GeneticConfig:
        return GeneticConfig(
            population_size=self.population_size, n_generations=self.n_generations,
            tournament_size=self.tournament_size, elite_fraction=self.elite_fraction,
            crossover_rate=self.crossover_rate, mutation_rate=self.mutation_rate,
            random_immigrant_fraction=self.random_immigrant_fraction,
            min_clauses=self.min_clauses, max_clauses=self.max_clauses,
            min_conditions=self.min_conditions, max_conditions=self.max_conditions,
            min_trades=self.min_trades, target_trades=self.target_trades,
            fitness_objective=self.fitness_objective,
            allow_short=self.allow_short, allow_long=self.allow_long, n_jobs=self.n_jobs, seed=self.seed,
            # Capital/comisión/slippage/riesgo alineados 1:1 con los mismos
            # valores que luego usará la verificación con el motor real
            # (verify_*), para que el fitness que guía la evolución ya sea
            # consistente con lo que esa verificación va a confirmar.
            initial_capital=self.verify_initial_capital, hall_of_fame_size=self.hall_of_fame_size,
            commission=self.verify_commission, slippage=self.verify_slippage,
            risk_per_trade=self.verify_risk_per_trade,
            evolve_risk=self.evolve_risk, sl_atr_min=self.sl_atr_min, sl_atr_max=self.sl_atr_max,
            tp_atr_min=self.tp_atr_min, tp_atr_max=self.tp_atr_max,
            use_oos_validation=self.use_oos_validation, n_folds=self.n_folds,
            min_trades_holdout=self.min_trades_holdout,
            niche_similarity_threshold=self.niche_similarity_threshold,
            niche_penalty_weight=self.niche_penalty_weight,
            hof_diversity_threshold=self.hof_diversity_threshold,
        )


@dataclass
class DiscoveryProgress:
    stage: str
    current: int
    total: int
    message: str
    top_candidates: list[dict] | None = None


class DiscoveryEngine:
    def __init__(self, config: Optional[DiscoveryConfig] = None):
        self.config = config or DiscoveryConfig()

    def run(self, df: pd.DataFrame, feats: pd.DataFrame, symbol: str, timeframe: str,
            progress_callback: Optional[Callable[[DiscoveryProgress], None]] = None,
            regime_mask: Optional[pd.Series] = None) -> dict:
        """
        `regime_mask`: Serie booleana alineada con `df.index`, True solo en
        las velas del/los régimen(es) de mercado elegido(s) (ver
        `regime.service.regime_entry_mask`). Cuando se pasa, tanto la
        búsqueda genética como la verificación con el motor real puntúan
        cada individuo ÚNICAMENTE por sus trades dentro de ese régimen —
        el objetivo es encontrar reglas que funcionen específicamente ahí,
        no una regla genérica evaluada sobre la mezcla de todos los
        regímenes. `df`/`feats` NO se recortan: se pasan completos para
        que los indicadores mantengan su warmup e historial normales; solo
        se descarta la señal (no las velas) fuera del régimen elegido.
        """
        cfg = self.config

        def _report(stage: str, current: int, total: int, message: str, top_candidates=None):
            if progress_callback:
                progress_callback(DiscoveryProgress(stage, current, total, message, top_candidates=top_candidates))
            logger.info("[%s] %d/%d - %s", stage, current, total, message)

        _report("evolve", 0, cfg.n_generations,
                 "Evolucionando combinaciones de indicadores multi-familia...")

        def _gen_cb(stat: GenerationStats):
            msg = (f"Generacion {stat.generation + 1}/{cfg.n_generations} — "
                   f"mejor fitness={stat.best_fitness:.3f}, "
                   f"promedio={stat.mean_fitness:.3f}, "
                   f"hall of fame={stat.hall_of_fame_size}")
            if stat.generation == 0 and stat.n_valid == 0:
                usable_bars = max(len(df) - WARMUP_BARS, 0)
                msg += (f" — ⚠️ NINGÚN individuo alcanzó min_trades={cfg.min_trades} "
                        f"({usable_bars} velas utilizables tras warmup de {WARMUP_BARS}). "
                        f"Es probable que TODAS las generaciones queden en -999: considera "
                        f"detener y ampliar el rango de fechas, usar temporalidad más fina, "
                        f"o bajar 'Mínimo de trades para ser válida'.")
            # Propagamos el snapshot en el objeto de progreso para que la UI
            # pueda actualizar la tabla de mejores estrategias en cada
            # generación, sin alterar el algoritmo de selección.
            try:
                setattr(stat, "top_candidates", getattr(stat, "top_candidates", []))
            except Exception:
                pass
            _report("evolve", stat.generation + 1, cfg.n_generations, msg,
                    top_candidates=getattr(stat, "top_candidates", []))

        genetic_engine = GeneticDiscoveryEngine(cfg.to_genetic_config())
        ga_result = genetic_engine.run(df, feats, timeframe=timeframe, progress_callback=_gen_cb,
                                        regime_mask=regime_mask)

        hall_of_fame = [ind for ind in ga_result.hall_of_fame if ind.metrics is not None]
        _report("evolve", cfg.n_generations, cfg.n_generations,
                 f"Busqueda evolutiva completada: {ga_result.n_evaluated_total} evaluaciones, "
                 f"{len(hall_of_fame)} estrategias distintas en el hall of fame.")

        leaderboard = []
        for ind in hall_of_fame:
            definition = individual_to_definition(ind)
            strategy_id = f"gen_{ind.individual_id}"
            name = _short_name(definition)
            leaderboard.append({
                "strategy_id": strategy_id, "name": name, "family": "Genetic_Discovered",
                "definition": definition,
                # `metrics` se sobreescribe más abajo con las cifras del motor
                # real si la verificación tiene éxito; `metrics_fast` conserva
                # siempre lo que vio el simulador rápido durante la evolución
                # (útil para depurar, nunca para decidir).
                "metrics": ind.metrics, "metrics_fast": ind.metrics,
                "fold_fitnesses": ind.fold_fitnesses,
                "verified": False,
                "dsr": None, "dsr_significant": None,
                "composite_score": ind.fitness,
                "long_rule_text": definition["long_rule_text"], "short_rule_text": definition["short_rule_text"],
                "families_used": definition["families_used"],
            })

        leaderboard.sort(key=lambda r: r["composite_score"], reverse=True)

        if cfg.verify_with_real_engine and leaderboard:
            _report("verify", 0, len(leaderboard),
                     "Verificando Hall of Fame con el motor de backtesting real...")
            n_ok = 0
            for i, entry in enumerate(leaderboard):
                verified_metrics = _verify_individual(entry["definition"], df, feats, timeframe, cfg,
                                                        regime_mask=regime_mask)
                if verified_metrics is not None:
                    entry["metrics"] = verified_metrics
                    entry["verified"] = True
                    n_ok += 1
                _report("verify", i + 1, len(leaderboard),
                         f"Verificado {i + 1}/{len(leaderboard)} con motor real "
                         f"({n_ok} ok, {i + 1 - n_ok} con fallback a estimación rápida)")
            leaderboard.sort(key=lambda r: r["composite_score"], reverse=True)

        # ------------------------------------------------------------
        # Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): corrige el
        # Sharpe verificado de cada estrategia por el numero de combinaciones
        # que el genetico probo para llegar a ella. Un Sharpe alto encontrado
        # tras evaluar decenas de miles de reglas es MENOS sorprendente que
        # el mismo Sharpe encontrado tras evaluar solo un puñado — el DSR
        # cuantifica eso como una probabilidad en vez de dejar que el Sharpe
        # crudo hable solo.
        # ------------------------------------------------------------
        n_trials = max(ga_result.n_evaluated_total, 1)
        ann = annualization_factor(timeframe)
        sharpe_std_annual = float(np.std(ga_result.trial_sharpes)) if len(ga_result.trial_sharpes) > 2 else 0.0
        sharpe_std_period = sharpe_std_annual / np.sqrt(ann) if ann > 0 else sharpe_std_annual
        for entry in leaderboard:
            if not entry.get("verified"):
                continue
            m = entry["metrics"]
            sr_period = m["sharpe"] / np.sqrt(ann) if ann > 0 else m["sharpe"]
            dsr = deflated_sharpe_ratio(
                observed_sharpe=sr_period, sharpe_std_across_trials=sharpe_std_period,
                n_trials=n_trials, n_periods=m.get("_n_periods", m["n_trades"]),
                skew=m.get("_skew", 0.0), kurtosis=m.get("_kurtosis", 3.0),
            )
            entry["dsr"] = dsr
            entry["dsr_significant"] = dsr >= 0.95

        # ------------------------------------------------------------
        # FDR (Benjamini-Hochberg) sobre TODO el Hall of Fame a la vez:
        # ver core/multiple_testing.py. p-valor de cada estrategia =
        # 1 - DSR (el DSR ya es una probabilidad de significancia). Las
        # entradas sin DSR (no verificadas con el motor real) entran como
        # None y quedan fuera del ajuste, sin afectar a las demás.
        # ------------------------------------------------------------
        p_values = [dsr_to_pvalue(entry.get("dsr")) for entry in leaderboard]
        fdr_result = benjamini_hochberg(p_values, alpha=cfg.fdr_alpha)
        for entry, adj_p, rejected in zip(
            leaderboard, fdr_result.adjusted_p_values, fdr_result.rejected
        ):
            has_dsr = entry.get("dsr") is not None
            entry["fdr_adjusted_p"] = adj_p if has_dsr else None
            entry["fdr_significant"] = rejected if has_dsr else None

        # ------------------------------------------------------------
        # Frontera de Pareto (Sharpe vs Drawdown) — puramente informativa:
        # NO participa en el fitness ni en qué entra al Hall of Fame (eso ya
        # pasó). Solo etiqueta, entre lo que ya sobrevivió, cuáles entradas
        # nadie domina simultáneamente en las dos métricas — para separar
        # visualmente "el mejor Sharpe absoluto" de "el mejor trade-off
        # riesgo/retorno", que no siempre son la misma estrategia.
        # ------------------------------------------------------------
        for entry in leaderboard:
            m = entry.get("metrics") or {}
            entry["pareto_sharpe"] = float(m.get("sharpe", float("-inf")))
            entry["pareto_drawdown_abs"] = abs(float(m.get("max_drawdown", 0.0) or 0.0))
        for entry in leaderboard:
            entry["pareto_optimal"] = not any(
                other is not entry
                and other["pareto_sharpe"] >= entry["pareto_sharpe"]
                and other["pareto_drawdown_abs"] <= entry["pareto_drawdown_abs"]
                and (other["pareto_sharpe"] > entry["pareto_sharpe"]
                     or other["pareto_drawdown_abs"] < entry["pareto_drawdown_abs"])
                for other in leaderboard
            )

        return {
            "n_generated": ga_result.n_evaluated_total,
            "n_survivors": len(hall_of_fame),
            "leaderboard": leaderboard,
            "generation_stats": ga_result.generation_stats,
            "symbol": symbol,
            "timeframe": timeframe,
            "n_trials": n_trials,
            "n_folds": cfg.n_folds if cfg.use_oos_validation else 1,
            "sharpe_std_trials": sharpe_std_annual,
            "fdr_alpha": cfg.fdr_alpha,
            "fdr_summary": fdr_result.summary() if fdr_result.n_tested else None,
        }


def _verify_individual(definition: dict, df: pd.DataFrame, feats: pd.DataFrame,
                        timeframe: str, cfg: "DiscoveryConfig",
                        regime_mask: Optional[pd.Series] = None) -> Optional[dict]:
    """Corre el individuo a través del motor de backtesting REAL
    (engine.backtester.BacktestEngine), sobre el mismo `df`/rango de fechas
    que usó el genético, y devuelve un dict de métricas en el mismo formato
    (fracciones, no porcentajes) que produce discovery/backtest_engine.py,
    para que la tabla del Hall of Fame no tenga que cambiar de formato.

    `regime_mask`: si se pasa, se bloquean (signal=0) las entradas fuera del
    régimen elegido antes de correr el motor real — para que la
    verificación quede en el mismo régimen sobre el que se optimizó, igual
    que el filtro de régimen de Backtesting/Optimización.

    Devuelve None si la verificación falla (p.ej. cero señales o excepción),
    en cuyo caso el llamador conserva la estimación rápida como fallback.
    """
    try:
        exec_frame = build_execution_frame(definition, df, feats)
        if regime_mask is not None:
            mask_aligned = regime_mask.reindex(exec_frame.index).fillna(False)
            exec_frame.loc[~mask_aligned, "signal"] = 0
        if int((exec_frame["signal"] != 0).sum()) == 0:
            return None

        bt_config = BacktestConfig(
            initial_capital=cfg.verify_initial_capital,
            commission=cfg.verify_commission,
            slippage=cfg.verify_slippage,
            risk_per_trade=cfg.verify_risk_per_trade,
            allow_long=cfg.verify_allow_long,
            allow_short=True,  # el propio individuo decide long/short vía su signal; no forzar aquí
            use_position_sizing=True,
        )
        engine = BacktestEngine(bt_config)
        results = engine.run(exec_frame, strategy_name=definition.get("individual_id", ""),
                              asset="", timeframe=timeframe)
    except Exception:
        logger.exception("Fallo verificando individuo %s con el motor real", definition.get("individual_id"))
        return None

    trades = results.trades or []
    n_trades = len(trades)
    if n_trades == 0:
        return None

    # pnl_pct de core.types.Trade ya está en % de retorno por operación —
    # se divide entre 100 para que quede en fracción, igual convención que
    # discovery/backtest_engine.py.
    pnl_fracs = [t.pnl_pct / 100.0 for t in trades]
    wins = [p for p in pnl_fracs if p > 0]
    losses = [p for p in pnl_fracs if p <= 0]
    win_rate = len(wins) / n_trades
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    avg_bars_held = float(np.mean([t.duration_bars for t in trades]))

    profit_factor = results.profit_factor
    if not np.isfinite(profit_factor):
        profit_factor = 999.0

    # Retornos por periodo (no por trade) de la curva de capital verificada
    # — insumo para el Deflated Sharpe Ratio, que necesita la asimetria y
    # curtosis reales de los retornos (los retornos de trading casi nunca
    # son normales) y el numero de observaciones detras del Sharpe.
    period_returns = results.equity_curve.pct_change().dropna()
    if len(period_returns) > 3 and period_returns.std() > 0:
        skew = float(period_returns.skew())
        kurt = float(period_returns.kurtosis()) + 3.0  # pandas da curtosis EXCEDENTE; la formula DSR usa curtosis cruda (normal=3)
    else:
        skew, kurt = 0.0, 3.0

    return {
        "total_return": results.net_profit_pct / 100.0,
        "cagr": results.cagr / 100.0,
        "sharpe": results.sharpe_ratio,
        "sortino": results.sortino_ratio,
        "calmar": results.calmar_ratio,
        "max_drawdown": results.max_drawdown_pct / 100.0,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": float(profit_factor),
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_bars_held": avg_bars_held,
        "max_win_streak": results.max_consecutive_wins,
        "max_loss_streak": results.max_consecutive_losses,
        "exposure": results.exposure_pct / 100.0,
        "final_equity": results.final_capital,
        # Internos, no se muestran en la tabla — solo para deflated_sharpe_ratio().
        "_skew": skew, "_kurtosis": kurt, "_n_periods": int(len(period_returns)),
    }


def _short_name(definition: dict) -> str:
    fams = "+".join(definition["families_used"][:5])
    return f"[{fams}] {definition['individual_id']}"


def run_discovery(df: pd.DataFrame, feats: pd.DataFrame, symbol: str, timeframe: str,
                   config: Optional[DiscoveryConfig] = None,
                   progress_callback: Optional[Callable[[DiscoveryProgress], None]] = None,
                   regime_mask: Optional[pd.Series] = None) -> dict:
    engine = DiscoveryEngine(config)
    return engine.run(df, feats, symbol, timeframe, progress_callback, regime_mask=regime_mask)
