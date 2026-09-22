from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

DEFAULT_OBJECTIVES = [
    "calmar", "sharpe", "profit_factor", "win_rate", "cagr",
    "expectancy", "drawdown", "stability", "robustness",
]

REGIMES = ["Tendencia Alcista", "Tendencia Bajista", "Consolidación"]


@dataclass
class ResearchConfig:
    """Configuración del Research Lab.

    Mantiene deliberadamente el mismo nivel de control que el Descubridor
    Genético. El Research Lab solo orquesta múltiples investigaciones;
    no usa una configuración genética simplificada.
    """
    assets: list[str]
    timeframe: str
    objectives: list[str] = field(default_factory=lambda: DEFAULT_OBJECTIVES.copy())
    regime_filters: list[str] = field(default_factory=list)
    date_from: Optional[date] = None
    date_to: Optional[date] = None

    # --- Motor genético: mismos controles que GeneticConfig ---
    population_size: int = 80
    generations: int = 20
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
    n_jobs_per_search: int = -1
    seed: int = 42
    initial_capital: float = 10_000.0
    hall_of_fame_size: int = 60

    # Riesgo evolucionado
    evolve_risk: bool = True
    sl_atr_min: float = 0.8
    sl_atr_max: float = 4.0
    tp_atr_min: float = 1.0
    tp_atr_max: float = 8.0
    risk_mutation_sigma_frac: float = 0.20

    # Costos / sizing
    commission: float = 0.001
    slippage: float = 0.0005
    risk_per_trade: float = 0.02

    # Validación WFO
    use_oos_validation: bool = True
    n_folds: int = 3
    train_fraction: float = 0.75
    min_trades_holdout: Optional[int] = None

    # Diversidad
    niche_similarity_threshold: float = 0.6
    niche_penalty_weight: float = 0.06
    hof_diversity_threshold: float = 0.6

    # Verificación final
    verify_with_real_engine: bool = True
    verify_top_n: int = 10

    @property
    def use_full_history(self) -> bool:
        return self.date_from is None and self.date_to is None

    @property
    def regime_mode(self) -> str:
        return "filtered" if self.regime_filters else "full_history"

    def validate(self) -> None:
        if not self.assets or len(self.assets) > 10:
            raise ValueError("Research Lab permite entre 1 y 10 activos.")
        if not self.objectives:
            raise ValueError("Selecciona al menos un objetivo.")
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("La fecha inicial no puede ser posterior a la final.")
        unknown = set(self.objectives) - set(DEFAULT_OBJECTIVES)
        if unknown:
            raise ValueError(f"Objetivos no soportados: {sorted(unknown)}")
        unknown_regimes = set(self.regime_filters) - set(REGIMES)
        if unknown_regimes:
            raise ValueError(f"Regímenes no soportados: {sorted(unknown_regimes)}")
        if self.population_size < 20 or self.generations < 3:
            raise ValueError("La población debe ser >=20 y las generaciones >=3.")
        if self.max_clauses < self.min_clauses:
            raise ValueError("Máximo de cláusulas no puede ser menor que el mínimo.")
        if self.max_conditions < self.min_conditions:
            raise ValueError("Máximo de condiciones no puede ser menor que el mínimo.")
        if self.n_folds < 1:
            raise ValueError("Las ventanas WFO deben ser >=1.")
        if self.verify_top_n < 1 or self.verify_top_n > self.hall_of_fame_size:
            raise ValueError("verify_top_n debe estar entre 1 y el tamaño del Hall of Fame.")

    @property
    def job_count(self) -> int:
        return len(self.assets) * len(self.objectives) * max(1, len(self.regime_filters) or 1)
