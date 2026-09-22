from dataclasses import dataclass

@dataclass(frozen=True)
class ObjectiveSpec:
    key: str
    label: str
    description: str

OBJECTIVES = {
    "calmar": ObjectiveSpec("calmar", "Calmar", "Retorno anualizado por unidad de drawdown."),
    "sharpe": ObjectiveSpec("sharpe", "Sharpe", "Retorno ajustado por volatilidad."),
    "profit_factor": ObjectiveSpec("profit_factor", "Profit Factor", "Ganancia bruta dividida por pérdida bruta."),
    "win_rate": ObjectiveSpec("win_rate", "Win Rate", "Proporción de operaciones ganadoras."),
    "cagr": ObjectiveSpec("cagr", "CAGR", "Tasa de crecimiento anual compuesto."),
    "expectancy": ObjectiveSpec("expectancy", "Expectancy", "Resultado esperado por operación."),
    "drawdown": ObjectiveSpec("drawdown", "Drawdown", "Busca el drawdown máximo más cercano a cero."),
    "stability": ObjectiveSpec("stability", "Stability", "Consistencia entre ventanas temporales."),
    "robustness": ObjectiveSpec("robustness", "Robustness", "Suelo de desempeño entre ventanas WFO."),
}

def get_objective(key: str) -> ObjectiveSpec:
    return OBJECTIVES[key]
