from __future__ import annotations
import pandas as pd
from discovery.indicators import compute_indicator_frame
from discovery.genetic_discovery import GeneticDiscoveryEngine, GeneticConfig, Individual
from discovery.discovery_engine import _verify_individual, individual_to_definition, DiscoveryConfig


def prepare_features(df: pd.DataFrame, asset: str, timeframe: str) -> pd.DataFrame:
    feats = compute_indicator_frame(df)
    try:
        from config.settings import TIMEFRAMES
        from regime.calibration import regime_feature_frame
        ann_factor = TIMEFRAMES.get(timeframe, {}).get("ann_factor", 252)
        regime_feats, _ = regime_feature_frame(df, ann_factor, asset, timeframe)
        feats = pd.concat([feats, regime_feats], axis=1)
    except Exception:
        pass
    return feats


def run_single(
    df: pd.DataFrame, feats: pd.DataFrame, timeframe: str, objective: str, *,
    population: int, generations: int, min_trades: int, target_trades: int,
    n_folds: int, n_jobs: int, seed: int, initial_capital: float,
    commission: float, slippage: float, risk_per_trade: float,
    allow_long: bool, allow_short: bool, evolve_risk: bool,
    tournament_size: int = 4, elite_fraction: float = 0.10,
    crossover_rate: float = 0.65, mutation_rate: float = 0.30,
    random_immigrant_fraction: float = 0.10,
    min_clauses: int = 1, max_clauses: int = 2,
    min_conditions: int = 1, max_conditions: int = 3,
    hall_of_fame_size: int = 60,
    sl_atr_min: float = 0.8, sl_atr_max: float = 4.0,
    tp_atr_min: float = 1.0, tp_atr_max: float = 8.0,
    risk_mutation_sigma_frac: float = 0.20,
    use_oos_validation: bool = True,
    train_fraction: float = 0.75,
    min_trades_holdout: int | None = None,
    niche_similarity_threshold: float = 0.6,
    niche_penalty_weight: float = 0.06,
    hof_diversity_threshold: float = 0.6,
    verify_with_real_engine: bool = True,
    verify_top_n: int = 10,
    regime_mask: pd.Series | None = None,
    progress_callback=None,
) -> list[Individual]:
    """Ejecuta exactamente el mismo GeneticConfig del Descubridor Genético.

    `verify_with_real_engine` se conserva en la firma por compatibilidad con
    ResearchConfig/Orchestrator, pero la evolución no lo consume: la
    verificación real se hace bajo demanda sobre candidatos finales. Esto
    evita que cada individuo de la población dispare el motor completo.
    """
    cfg = GeneticConfig(
        population_size=population,
        n_generations=generations,
        tournament_size=tournament_size,
        elite_fraction=elite_fraction,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        random_immigrant_fraction=random_immigrant_fraction,
        min_clauses=min_clauses,
        max_clauses=max_clauses,
        min_conditions=min_conditions,
        max_conditions=max_conditions,
        min_trades=min_trades,
        target_trades=target_trades,
        fitness_objective=objective,
        allow_short=allow_short,
        allow_long=allow_long,
        n_jobs=n_jobs,
        seed=seed,
        initial_capital=initial_capital,
        hall_of_fame_size=hall_of_fame_size,
        evolve_risk=evolve_risk,
        sl_atr_min=sl_atr_min,
        sl_atr_max=sl_atr_max,
        tp_atr_min=tp_atr_min,
        tp_atr_max=tp_atr_max,
        risk_mutation_sigma_frac=risk_mutation_sigma_frac,
        commission=commission,
        slippage=slippage,
        risk_per_trade=risk_per_trade,
        use_oos_validation=use_oos_validation,
        n_folds=n_folds,
        train_fraction=train_fraction,
        min_trades_holdout=min_trades_holdout,
        niche_similarity_threshold=niche_similarity_threshold,
        niche_penalty_weight=niche_penalty_weight,
        hof_diversity_threshold=hof_diversity_threshold,
    )
    # El filtro de régimen se aplica dentro de la misma semántica del
    # Descubridor: solo restringe la muestra/señal; no recalcula indicadores.
    result = GeneticDiscoveryEngine(cfg).run(
        df, feats, timeframe=timeframe,
        progress_callback=progress_callback,
        regime_mask=regime_mask,
    )
    hof = result.hall_of_fame

    # Verificación final acotada: nunca volvemos a correr el motor real sobre
    # toda la población. Solo los mejores N del Hall of Fame pasan por el
    # mismo BacktestEngine usado por la plataforma. Esto conserva calidad de
    # confirmación sin convertir cada búsqueda multiobjetivo en decenas de
    # backtests completos.
    if verify_with_real_engine and hof:
        verify_cfg = DiscoveryConfig(
            verify_with_real_engine=True,
            verify_initial_capital=float(initial_capital),
            verify_commission=float(commission),
            verify_slippage=float(slippage),
            verify_risk_per_trade=float(risk_per_trade),
            verify_allow_long=bool(allow_long),
        )
        limit = max(1, min(int(verify_top_n), len(hof)))
        for ind in hof[:limit]:
            ind.metrics_fast = dict(ind.metrics or {})
            verified = _verify_individual(
                individual_to_definition(ind), df, feats, timeframe, verify_cfg,
                regime_mask=regime_mask,
            )
            if verified is not None and int(verified.get("n_trades", 0)) >= int(min_trades):
                ind.metrics = verified
                ind.research_verified = True
            else:
                ind.research_verified = False
    for ind in hof:
        if not hasattr(ind, "metrics_fast"):
            ind.metrics_fast = dict(ind.metrics or {})
        if not hasattr(ind, "research_verified"):
            ind.research_verified = False
    return hof
