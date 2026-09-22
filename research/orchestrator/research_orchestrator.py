from __future__ import annotations
import pandas as pd
from typing import Callable, Optional
from research.data_registry import load_clean_dataset
from research.config import ResearchConfig
from research.jobs.research_job import ResearchJob
from research.jobs.research_batch import ResearchBatch
from research.discovery.discovery_runner import prepare_features, run_single
from research.regimes.regime_filter import build_regime_mask
from research.results.result_store import new_batch_id, save_batch


class ResearchOrchestrator:
    """Coordina búsquedas independientes activo × régimen × objetivo.

    Los datasets proceden exclusivamente del registro de series LIMPIAS creado por Gráficos.
    Este orquestador no descarga ni solicita datos directamente a MT5.
    """

    def __init__(
        self,
        config: ResearchConfig,
        preloaded_data: Optional[dict[str, pd.DataFrame]] = None,
    ):
        config.validate()
        self.config = config
        self.batch = ResearchBatch(batch_id=new_batch_id())
        self.datasets: dict[str, pd.DataFrame] = {
            k: v.copy() for k, v in (preloaded_data or {}).items()
            if v is not None and not v.empty
        }
        self.features: dict[str, pd.DataFrame] = {}
        self.regime_masks: dict[tuple[str, str], pd.Series | None] = {}

        regimes = config.regime_filters or [None]
        n = 1
        for asset in config.assets:
            for regime in regimes:
                for objective in config.objectives:
                    job_id = f"{self.batch.batch_id}-{n:03d}"
                    self.batch.jobs.append(
                        ResearchJob(job_id, asset, config.timeframe, objective, regime)
                    )
                    n += 1

    def _load_asset(self, asset: str) -> pd.DataFrame:
        if asset not in self.datasets:
            df = load_clean_dataset(
                asset, self.config.timeframe,
                self.config.date_from, self.config.date_to,
            )
            if df is None or df.empty:
                raise ValueError(
                    f"No existe un dataset limpio registrado para {asset} {self.config.timeframe}. "
                    "Envíalo primero desde Gráficos."
                )
            self.datasets[asset] = df.sort_index()
        return self.datasets[asset]

    def _prepare_asset(self, asset: str) -> None:
        if asset in self.features:
            return
        df = self._load_asset(asset)
        self.features[asset] = prepare_features(df, asset, self.config.timeframe)
        if self.config.regime_filters:
            for regime in self.config.regime_filters:
                self.regime_masks[(asset, regime)] = build_regime_mask(
                    df, asset, self.config.timeframe, [regime]
                )

    def _persist(self) -> None:
        save_batch(self.batch.batch_id, {
            "batch_id": self.batch.batch_id,
            "status": self.batch.status,
            "config": self.config.__dict__,
            "jobs": [j.to_dict() for j in self.batch.jobs],
        })

    def run(
        self,
        progress: Callable[[ResearchBatch, ResearchJob], None] | None = None,
    ) -> ResearchBatch:
        self.batch.status = "running"
        for job in self.batch.jobs:
            try:
                self._prepare_asset(job.asset)
                df = self.datasets[job.asset]
                feats = self.features[job.asset]
                mask = (
                    self.regime_masks.get((job.asset, job.regime))
                    if job.regime else None
                )
                job.status = "running"
                job.progress = 0.0
                job.message = "Iniciando evolución genética..."
                if progress:
                    progress(self.batch, job)

                def _job_progress(stat):
                    # La barra global avanza dentro de cada investigación,
                    # no solo cuando termina una búsqueda completa.
                    job.progress = min(0.98, float(stat.generation + 1) / max(self.config.generations, 1))
                    job.message = (
                        f"Generación {stat.generation + 1}/{self.config.generations} · "
                        f"mejor fitness {stat.best_fitness:.3f} · "
                        f"{stat.n_valid} válidos · Hall of Fame {stat.hall_of_fame_size}"
                    )
                    # Snapshot ligero y serializable para que Research Lab
                    # muestre resultados reales durante la evolución.
                    job.live_candidates = list(getattr(stat, "top_candidates", []) or [])
                    if progress:
                        progress(self.batch, job)

                hof = run_single(
                    df, feats, self.config.timeframe, job.objective,
                    population=self.config.population_size,
                    generations=self.config.generations,
                    min_trades=self.config.min_trades,
                    target_trades=self.config.target_trades,
                    n_folds=self.config.n_folds,
                    n_jobs=self.config.n_jobs_per_search,
                    seed=self.config.seed + self.batch.jobs.index(job),
                    initial_capital=self.config.initial_capital,
                    commission=self.config.commission,
                    slippage=self.config.slippage,
                    risk_per_trade=self.config.risk_per_trade,
                    allow_long=self.config.allow_long,
                    allow_short=self.config.allow_short,
                    evolve_risk=self.config.evolve_risk,
                    tournament_size=self.config.tournament_size,
                    elite_fraction=self.config.elite_fraction,
                    crossover_rate=self.config.crossover_rate,
                    mutation_rate=self.config.mutation_rate,
                    random_immigrant_fraction=self.config.random_immigrant_fraction,
                    min_clauses=self.config.min_clauses,
                    max_clauses=self.config.max_clauses,
                    min_conditions=self.config.min_conditions,
                    max_conditions=self.config.max_conditions,
                    hall_of_fame_size=self.config.hall_of_fame_size,
                    sl_atr_min=self.config.sl_atr_min,
                    sl_atr_max=self.config.sl_atr_max,
                    tp_atr_min=self.config.tp_atr_min,
                    tp_atr_max=self.config.tp_atr_max,
                    risk_mutation_sigma_frac=self.config.risk_mutation_sigma_frac,
                    use_oos_validation=self.config.use_oos_validation,
                    train_fraction=self.config.train_fraction,
                    min_trades_holdout=self.config.min_trades_holdout,
                    niche_similarity_threshold=self.config.niche_similarity_threshold,
                    niche_penalty_weight=self.config.niche_penalty_weight,
                    hof_diversity_threshold=self.config.hof_diversity_threshold,
                    verify_with_real_engine=self.config.verify_with_real_engine,
                    verify_top_n=self.config.verify_top_n,
                    regime_mask=mask,
                    progress_callback=_job_progress,
                )

                from discovery.discovery_engine import individual_to_definition
                job.result = {
                    "candidates": [
                        {
                            "strategy_id": ind.individual_id,
                            "name": (
                                f"Research_{job.asset}_{self.config.timeframe}_"
                                f"{ind.individual_id}"
                            ),
                            "fitness": float(ind.fitness),
                            "metrics": ind.metrics or {},
                            "long_rule_text": str(ind.long_rule),
                            "short_rule_text": str(ind.short_rule),
                            "risk": ind.risk_params(),
                            "definition": individual_to_definition(ind),
                            "objective": job.objective,
                            "verified": bool(getattr(ind, "research_verified", False)),
                            "metrics_fast": getattr(ind, "metrics_fast", None),
                        }
                        for ind in hof
                    ]
                }
                job.progress = 1.0
                job.message = "Investigación completada."
                job.status = "completed"
            except Exception as exc:
                job.status = "failed"
                job.progress = 1.0
                job.message = "Error durante la investigación."
                job.error = str(exc)

            self._persist()
            if progress:
                progress(self.batch, job)

        self.batch.status = (
            "completed" if self.batch.failed == 0 else "completed_with_errors"
        )
        self._persist()
        return self.batch
