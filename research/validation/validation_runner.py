from __future__ import annotations

def summarize_discovery(individual, objective: str, asset: str, regime: str | None) -> dict:
    metrics = individual.metrics or {}
    return {
        "asset": asset,
        "regime": regime or "Todo el histórico",
        "objective": objective,
        "strategy_id": individual.individual_id,
        "fitness": float(individual.fitness),
        "metrics": metrics,
        "long_rule": individual.long_rule,
        "short_rule": individual.short_rule,
        "risk": individual.risk_params(),
    }
