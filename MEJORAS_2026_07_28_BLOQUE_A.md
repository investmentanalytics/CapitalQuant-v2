# CapitalQuant — Continuación: Bloque A del roadmap (sesión siguiente a 28 jul 2026)

Este archivo documenta el trabajo hecho a partir de `PROMPT_CONTINUACION.md`,
bloque **A ("Cerrar el cableado de lo ya construido")**. Ver
`MEJORAS_2026_07_28.md` para lo hecho en la sesión anterior (purga/embargo,
costos realistas como motor, generalización como API, sandbox AST).

## ⚠️ Nota sobre el entorno de esta sesión

Este bloque se hizo en un entorno **sin acceso a red**, por lo que no fue
posible `pip install` `pytest`, `streamlit`, `optuna`, `loguru`, `plotly` ni
`pyarrow`. Para poder verificar el trabajo de todas formas:

- Se armaron shims mínimos offline de `loguru`, `optuna`, `pytest` (solo
  `approx`/`raises`) y `streamlit`/`plotly` (solo lo necesario para poder
  *importar* los módulos y detectar errores de sintaxis/nombres — **no**
  ejercitan la interactividad real de los widgets Streamlit).
- Se usó un runner de tests casero (no pytest real) que sí entiende
  clases `Test*` + `setup_method`, que es lo único "avanzado" que usan
  los tests existentes.
- Con esto se confirmó el baseline de 44/44 tests antes de tocar nada, y
  44 → 50 después de este bloque (los 6 nuevos son de este bloque).

**Recomendación:** antes de dar por buena esta entrega, correr
`pip install -r requirements.txt && python3 -m pytest tests/ -v` en una
máquina con red (idealmente Windows, para probar también la rama real de
`MetaTrader5`) para confirmar con las dependencias reales — especialmente
la UI de Streamlit, que aquí solo se verificó por import, no interacción.

## A.1 — Costos realistas cableados a la UI (Backtesting + Optimización)

- `market_data/mt5_provider.py::get_symbol_info`: se agregó la clave
  `contract_size` (desde `trade_contract_size` de MT5), que antes faltaba
  — sin ella, `InstrumentCostProfile.from_symbol_info` siempre caía al
  default de 100 000 aunque el símbolo fuera un CFD de acciones/oro/etc.
  con contrato distinto.
- Nuevo `ui/components/cost_toggle.py`: widget reutilizable
  `render_cost_toggle(symbol, key_prefix)` que:
  - Si hay conexión MT5 activa: arma el `InstrumentCostProfile` real vía
    `MT5Provider.get_symbol_info(symbol)` + comisión por lote que ingresa
    el usuario.
  - Si NO hay conexión: **deshabilita el toggle y explica por qué**
    (nunca cae en silencio al modelo en % sin decirlo).
  - Si el símbolo no responde estando conectado: igual, error explícito
    y fallback claro al modelo en %.
- Conectado en:
  - `ui/views/backtesting.py` — configuración + ejecución + mensaje de
    resultado indicando qué modelo de costos se usó.
  - `ui/views/optimization.py` — las 3 pestañas que corren backtests con
    parámetros de usuario: Optimización Estándar (+ su re-backtest con
    mejores parámetros), Walk-Forward, Sensibilidad.
- **Nota de alcance:** no se tocaron los defaults fijos de Monte Carlo
  (que re-samplea sobre resultados ya corridos, no vuelve a correr el
  motor con un símbolo/config nuevos) — no aplica ahí.

## A.2 — Botón "Validar generalización" en Discovery

- Nueva sección en `ui/views/discovery.py`, dentro del flujo de "Enviar
  una estrategia a Backtesting": **🧪 Validar generalización**.
  - Multiselect de símbolos (8-15 recomendado), poblado desde
    `MT5Provider.get_available_symbols()` si hay conexión activa, o
    desde `config.settings.ASSETS` si no la hay (nunca desde una fuente
    externa a MT5, consistente con el resto de la plataforma).
  - Al presionar el botón: compila la definición elegida del Hall of
    Fame a una clase `BaseStrategy` real, descarga/carga datos de los
    símbolos elegidos vía el `MarketDataProvider` existente
    (`market_data.get_data`, mismo camino que el resto de la UI), corre
    `run_cross_symbol_generalization` + `run_regime_and_seasonality_test`
    (`discovery/generalization.py`, ya existía como API desde la sesión
    anterior) y muestra el reporte completo: tasa de generalización +
    veredicto, tabla por símbolo, estabilidad mensual y por régimen ADX.
  - El resultado queda en `st.session_state`, no en disco — es una
    validación de la sesión actual, no algo que deba persistir si el
    usuario no envía la estrategia.
- **"Antes de exportar como validada"**: el manifiesto que guarda
  `record_discovery_manifest` ahora incluye
  `generalization_validated` (bool) + `generalization_rate` +
  `generalization_verdict` cuando existen. El botón "📤 Enviar a
  Backtesting" **no se bloquea** si no se validó (para no romper el
  flujo existente ni forzar 8-15 descargas de símbolos en cada envío),
  pero:
  - Si se envía sin validar, se lo dice explícitamente y sugiere
    correr la validación antes de confiar en la estrategia para operar.
  - El panel de "Estrategias del Descubridor ya guardadas" ahora
    muestra si cada una fue validada o no, y con qué veredicto.
- **Bug real encontrado y corregido en el camino:** el primer intento
  de este punto usó `strategies/code_compiler.py::compile_strategy_code`
  (el sandbox regex+AST del Constructor) para compilar la definición
  elegida — y falló siempre, porque el código que produce
  `discovery/codegen.py::generate_strategy_code` trae `import pandas as
  pd` / `import numpy as np` de cabecera, y el sandbox del Constructor
  (correctamente) rechaza *cualquier* `import`, porque ese sandbox es
  para texto libre tecleado por un humano. Son dos niveles de confianza
  distintos: el código de `codegen.py` es generado programáticamente
  por la plataforma (mismo nivel de confianza que ya usan
  `save_user_strategy_file` + `refresh_registry()` para persistirlo a
  disco), no texto arbitrario de un usuario. Se agregó
  `discovery/codegen.py::compile_definition_to_class(definition,
  class_name, symbol, timeframe)`, que compila en memoria por el camino
  confiable correcto (sin sandbox, sin tocar disco). Esto es reutilizable
  para cualquier otro punto del roadmap que necesite instanciar una
  definición del Hall of Fame sin pasar por el Constructor.

## Tests nuevos

`tests/test_discovery_generalization_pipeline.py` (6 tests) — cubre el
camino completo codegen → compilación en memoria → generalización cruzada
+ estabilidad de régimen, con una definición de regla real (RSI(14) de
`CONDITION_LIBRARY`, no inventada), incluyendo el caso de un símbolo sin
datos (debe marcarse `ok=False` en esa fila, no tumbar el reporte
completo). Esto cierra parte de B.5 (cobertura de tests en `discovery/`,
que antes solo se probaba con una `BaseStrategy` escrita a mano en
`test_hardening.py`).

```bash
cd capitalquant
pip install -r requirements.txt
python3 -m pytest tests/ -v
# 50 passed (44 previos + 6 nuevos)
```

## Lo que falta de A (deliberadamente no tocado en este bloque)

- No se agregó UI para configurar `purge_pct`/`embargo_pct`/
  `max_lookback_bars` del Walk-Forward (quedan en sus defaults seguros
  0.02/0.02) — no estaba en el alcance explícito de A.1/A.2, que era
  costos + generalización.
- La validación de generalización no se corre automáticamente ni se
  exige antes de enviar — es una decisión de UX deliberada (ver arriba);
  si se prefiere hacerla obligatoria, es un cambio de una línea (bloquear
  el botón de envío si `gen_state is None`).

## Siguiente en la cola (B, C, D del prompt de continuación)

Sin empezar todavía: FDR (Benjamini-Hochberg), CPCV completo, ampliar
tests de `discovery/`/`portfolio/` más allá de lo agregado aquí,
optimización de portafolio real (Fase 2), panel de correlación (Fase 2),
y toda la Fase 3 (ejecución) — deliberadamente la última y la que exige
más cuidado.
