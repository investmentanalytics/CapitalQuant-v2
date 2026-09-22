# CapitalQuant 3.0

**Plataforma profesional de investigación cuantitativa y construcción de portafolios multi-activo.**

---

## Aletheia Discovery Engine, ahora integrado

El **Descubridor Genético** (antes un programa aparte, "Aletheia Discovery Engine") vive ahora
dentro de CapitalQuant como una página más — mismo proceso, mismo `streamlit run ui/app.py`,
mismo tema visual. Qué cambió exactamente:

- **Se eliminó todo lo que ya hacía CapitalQuant**: su propia fuente/caché de datos (Synthetic / MT5 / CSV + base sqlite), su backtester de resultados, su walk-forward, su Monte Carlo,
  su bootstrap y su clasificador de regímenes. Todo eso ahora se hace con los módulos de
  CapitalQuant (`market_data/`, `engine/backtester.py`, `optimization/`), sin duplicar nada.
- **Se conservó únicamente lo que CapitalQuant no tenía**: el motor genético que descubre
  combinaciones nuevas de indicadores (`discovery/genetic_discovery.py`, `discovery/rule_engine.py`,
  `discovery/indicators.py`) y el generador de código que traduce una regla descubierta en una
  clase `BaseStrategy` real (`discovery/codegen.py`). El backtester interno que queda en
  `discovery/backtest_engine.py` es solo el evaluador rápido usado por el algoritmo genético para
  puntuar miles de combinaciones por corrida — no es una funcionalidad aparte, nunca se expone al
  usuario como "backtesting".
- **Página nueva "Descubridor Genético"**: configura y corre la búsqueda evolutiva sobre el
  activo/temporalidad que tengas seleccionados, y con un clic ("Enviar a Backtesting") compila la
  regla elegida como una estrategia real —el mismo mecanismo de archivo que usa el Constructor
  (`strategies.save_user_strategy_file` + `refresh_registry`)— y queda disponible al instante en
  Backtesting, Optimización y Portfolio, igual que cualquier otra estrategia.
- **Backtesting ya graficaba indicadores + señales de entrada/salida + niveles de SL/TP** sobre
  el precio (`visualization/charts.py::candlestick_chart`); las estrategias descubiertas ahora
  también exponen esas columnas (`stop_loss`/`take_profit`, calculadas en múltiplos de ATR igual
  que evalúa el genético), así que aparecen graficadas automáticamente sin cambios adicionales.
- **Página nueva "Mercado en Vivo"**: elige cualquier estrategia registrada y muestra un gráfico
  estilo TradingView con sus indicadores y la última señal (con SL/TP) sobre datos en vivo (MT5 si
  hay sesión activa; si no, el último dato disponible en el caché local). Es **solo visualización**
  — no envía órdenes; la ejecución la haces tú manualmente en tu bróker/terminal.

---

## Novedades de esta revisión

**Nuevo:**
- **Librería de indicadores ampliada a 58** (`strategies/base.py` — `Indicators`): además de los 11
  originales, se añadieron WMA, DEMA, TEMA, HMA, KAMA, VWMA, ADX/+DI/-DI, Aroon, Parabolic SAR,
  SuperTrend, Ichimoku, TRIX, Vortex, Mass Index, DPO, regresión lineal, Choppiness Index, Coppock
  Curve, CCI, Williams %R, ROC, Awesome/Ultimate Oscillator, Fisher Transform, TSI, KST, PPO,
  Stochastic RSI, Keltner Channels, desviación estándar, volatilidad histórica, NATR, %B y ancho de
  Bollinger, OBV, MFI, CMF, línea A/D, Chaikin Oscillator, Force Index, Ease of Movement, PVT,
  oscilador de volumen, pivot points, z-score y correlación móvil. Todos disponibles vía
  `Indicators.<nombre>(...)` tanto en estrategias escritas a mano como en el Constructor.
- **Constructor de Estrategias** (página "Constructor"): editor de código dentro de la app para
  crear estrategias con cualquier combinación de indicadores, solo largos / solo cortos / ambos,
  SL/TP a tu gusto. Valida, backtestea y optimiza sin salir de la página, y al guardar queda
  disponible al instante en Backtesting, Optimización, Walk-Forward, Monte Carlo y Sensibilidad
  (strategies/code_compiler.py, ui/views/strategy_builder.py).
- Rediseño visual completo: paleta negro/dorado, tipografía Fraunces + Inter + IBM Plex Mono,
  sin emojis en ningún punto de la interfaz.
- Control explícito de "Largos y cortos / Solo largos / Solo cortos" en Backtesting y en el
  Constructor (`BacktestConfig.allow_long`, nuevo — antes solo existía `allow_short`).

**Corregido:**
- `MetaTrader5` en `requirements.txt` rompía `pip install` fuera de Windows (sin wheel para
  Linux/Mac). Ahora tiene marcador de entorno `; sys_platform == "win32"`.
- `BacktestConfig.max_positions` y `trade_on_close` existían pero el motor nunca los leía —
  se retiraron para no inducir a error (el motor soporta una sola posición a la vez).
- `Trade.duration_bars` era un placeholder que siempre devolvía 0; ahora el motor lo calcula
  y lo guarda en cada operación.
- `use_container_width` (deprecado desde el 31/12/2025) reemplazado por `width` en toda la app.
- Documentado en el motor el comportamiento de reentrada en la misma vela tras un stop
  loss/take profit intrabar (no cambia el comportamiento, pero antes no estaba advertido).

---

## Ejecutar

```bash
# Instalar dependencias
pip install -r requirements.txt

# Lanzar la aplicación
streamlit run ui/app.py
```

---

## Arquitectura

```
capitalquant/
│
├── config/           # Configuración centralizada (activos, timeframes, defaults, tema visual)
├── core/             # Lógica pura reutilizable: tipos, métricas, validadores
├── engine/           # Motor de backtesting vectorizado
├── strategies/       # Plug-and-play: añadir .py = nueva estrategia disponible
│   └── code_compiler.py  # Compilador seguro para el Constructor de Estrategias
├── discovery/        # Descubridor Genético (ex-Aletheia): solo la búsqueda evolutiva + codegen
│   ├── rule_engine.py       # Librería de condiciones atómicas por familia de indicador
│   ├── indicators.py        # Banco de indicadores precomputado para el genético
│   ├── genetic_discovery.py # Algoritmo genético (reglas DNF, SL/TP evolucionado)
│   ├── backtest_engine.py   # Backtester rápido INTERNO, solo para puntuar individuos
│   ├── discovery_engine.py  # Orquesta la corrida y arma el leaderboard/hall of fame
│   └── codegen.py           # Traduce una regla descubierta a una clase BaseStrategy real
├── portfolio/        # Portfolio manager multi-activo y multi-estrategia
├── optimization/     # Bayesiana, Walk-Forward, Monte Carlo, Sensibilidad
├── market_data/      # Proveedores de datos desacoplados (MT5 + local — única fuente: MT5)
├── visualization/    # Gráficos Plotly puros (sin Streamlit)
├── ui/               # Interfaz Streamlit — solo presentación
│   ├── app.py        # Entry point
│   ├── views/         # Una página por módulo:
│   │                  #   charts, strategy_builder, discovery, backtesting,
│   │                  #   live (Mercado en Vivo), portfolio, optimization
│   └── components/   # Sidebar, metrics grid, etc.
├── tests/            # Tests del core y estrategias
└── data/             # Datos locales (parquet, excluidos del repo)
```

---

## Añadir una estrategia nueva

1. Crear archivo `strategies/mi_estrategia.py`
2. Definir clase que hereda `BaseStrategy`:

```python
from strategies.base import BaseStrategy, Indicators

class MiEstrategia(BaseStrategy):
    name = "Mi Estrategia"
    description = "Descripción breve"

    def __init__(self, periodo: int = 20, **kwargs):
        super().__init__(periodo=periodo, **kwargs)
        self.periodo = periodo

    def generate_signals(self, data):
        df = data.copy()
        # ... tu lógica aquí
        df["signal"] = 0
        return df

    def get_param_space(self):
        return {"periodo": ("int", 5, 50)}

    def get_indicator_columns(self):
        return []
```

3. ✅ La estrategia aparece automáticamente en toda la plataforma.

---

## Módulos clave

### `core/types.py`
Todos los dataclasses del sistema: `BacktestConfig`, `Trade`, `BacktestResults`.

### `core/metrics.py`
Funciones puras de métricas financieras: Sharpe, Sortino, Calmar, CAGR, Drawdown.

### `portfolio/manager.py`
`PortfolioManager` multi-activo: combina N activos × M estrategias con pesos flexibles.

### `optimization/walk_forward.py`
Walk-Forward Optimization con Efficiency Ratio para detectar overfitting.

### `optimization/monte_carlo.py`
Simulación de 5000+ trayectorias por bootstrap de retornos empíricos.

### `ui/state.py`
`AppState` — toda la gestión de session_state en un único lugar.

---

## Añadir un proveedor de datos nuevo

```python
# market_data/mi_fuente.py
from market_data.base import BaseDataProvider

class MiFuente(BaseDataProvider):
    name = "Mi Fuente"

    def get_data(self, symbol, timeframe, *, force_download=False):
        # implementar
        ...

    def is_available(self, symbol, timeframe):
        # implementar
        ...
```

---

## Tests

```bash
pytest tests/ -v
```

---

## Preparación para API móvil

Todo el código en `core/`, `engine/`, `strategies/`, `portfolio/` y `optimization/`
es **completamente independiente de Streamlit**.

Para exponer como API REST con FastAPI:

```python
# api/routes/backtest.py
from fastapi import APIRouter
from engine.backtester import BacktestEngine
from core.types import BacktestConfig

router = APIRouter()

@router.post("/backtest")
def run_backtest(payload: dict):
    config = BacktestConfig(**payload["config"])
    engine = BacktestEngine(config)
    # ...
```

La app móvil iOS/Android consume los mismos módulos sin reescribir código.

---

## Bugs corregidos respecto a v2.x

| Bug | Estado |
|---|---|
| `NameError: key` no definida antes del if | ✅ Corregido |
| `st.button()` anidado en página Datos | ✅ Eliminado (módulo eliminado) |
| `ann_factor = 252` fijo en portfolio ignora timeframe | ✅ Corregido |
| SL/TP via `__dict__` dinámico | ✅ Campos tipados en `Trade` |
| 5 `st.write()` de DEBUG en producción | ✅ Eliminados |
| `PortfolioConfig` declarado pero nunca usado | ✅ Eliminado |

---

*CapitalQuant 3.0 — Arquitectura refactorizada 2026*


## Research Lab — nueva investigación multi-activo

Esta revisión añade `research/` como una capa de orquestación sobre los motores existentes. No reemplaza el
Descubridor Genético, Régimen de Mercado, Backtesting, Portfolio ni Mercado en Vivo.

El **Research Lab** permite seleccionar hasta 10 símbolos ofrecidos por el terminal FBS MetaTrader 5,
elegir timeframe y ventana temporal, y ejecutar desde un solo botón búsquedas genéticas independientes:

- Calmar
- Sharpe
- Profit Factor
- Win Rate
- CAGR
- Expectancy
- Drawdown
- Stability
- Robustness

La unidad de investigación es:

`activo × régimen (opcional) × objetivo`

Por ejemplo, 10 activos y 9 objetivos generan 90 búsquedas independientes. Si se seleccionan 2
regímenes para investigar por separado, generan 180 búsquedas. El histórico de precios se carga una
sola vez por activo y se reutiliza entre sus búsquedas; el filtro de régimen se calcula con el mismo
servicio causal de `regime/`.

Los resultados de cada batch se guardan en `data/research/*.json`, por lo que la investigación puede
consultarse después de terminar.

### Ejecución

En Windows con FBS MetaTrader 5 instalado y abierto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run ui/app.py
```

También puedes usar `run_capitalquant.bat`.

### Nota metodológica

Cada objetivo representa una búsqueda genética independiente. Las métricas restantes se conservan
en el resultado para permitir el análisis posterior. `Stability` y `Robustness` se calculan sobre las
ventanas WFO disponibles durante la búsqueda; no significan garantía de desempeño futuro.

