# CapitalQuant — Bloques B, C, D (2026-07-30)

Continuación de `MEJORAS_2026_07_28_BLOQUE_A.md`. Cierra las 3 fases
pendientes del roadmap.

## Cómo se verificó este entrega (léelo antes de dudar de un resultado)

Este sandbox **no tiene red** (no hay `pip install` posible) y **no
tiene instalados** `streamlit`, `plotly`, `optuna` ni `loguru` — las
mismas dependencias que en la sesión anterior. Se reconstruyeron shims
mínimos (no son los tuyos, viven fuera del zip) para poder:

1. Correr los **100 tests** reales del proyecto (69 nuevos + 31 de antes... en
   realidad 50 preexistentes + 50 nuevos = 100) con un runner casero
   (no es pytest real, pero ejecuta la misma lógica `assert` de siempre).
2. Import-checkear los 12 módulos de `ui/` y `visualization/` tocados o
   nuevos (detecta errores de sintaxis, nombres no definidos, imports
   rotos — **no** simula clics ni interacción real).
3. `python3 -m py_compile` sobre **todos** los `.py` del repo.

Lo que esto **no** reemplaza: correrlo de verdad en tu máquina con
Streamlit, un backtest real de varios minutos, o —sobre todo— probar
Fase 3 con MT5 conectado a una cuenta demo real. Antes de operar con
esto, aunque sea en demo, **corré tú mismo `pytest tests/ -v`** en tu
entorno con las dependencias reales instaladas.

---

## Bloque B — Cierre de Fase 1

### B.1 — FDR (Benjamini-Hochberg)

**Nuevo:** `core/multiple_testing.py`

El **DSR** (Deflated Sharpe Ratio, ya existía) corrige el Sharpe de UNA
estrategia por cuántas combinaciones probó el genético para llegar a
ella. El **FDR** resuelve un problema distinto: mirando el Hall of Fame
**completo** (decenas de candidatas a la vez), ¿cuántas de ellas son
en realidad ruido? Se aplica Benjamini-Hochberg (1995) sobre
`p = 1 - DSR` de cada estrategia del leaderboard.

- Wireado en `discovery/discovery_engine.py`: cada entrada del
  leaderboard ahora tiene `fdr_significant` y `fdr_adjusted_p`.
- Nuevo parámetro `DiscoveryConfig.fdr_alpha` (default 0.10).
- UI (`ui/views/discovery.py`): columna "fdr" en la tabla, métrica en
  el detalle de cada estrategia, y se persiste en el manifiesto guardado.

### B.2 — CPCV completo (no solo purga/embargo simple)

**Nuevo:** `optimization/cpcv.py`

El `WalkForwardOptimizer` existente ya hacía purga/embargo, pero sobre
un único camino secuencial. Esto implementa **Combinatorial Purged
Cross-Validation** (López de Prado, cap. 11-12): parte el histórico en
N grupos, evalúa **todas** las combinaciones C(N,k) de grupos de test
(con purga+embargo en cada frontera, en ambos lados — a diferencia de
WFO secuencial, en CPCV un bloque de train puede quedar antes O
después de un bloque de test), y evalúa un **pool fijo** de
configuraciones candidatas en cada partición para poder comparar
resultados entre particiones.

De ahí se calcula la **Probabilidad de Sobreajuste de Backtest (PBO)**
— Bailey, Borwein, López de Prado & Zhu (2016), método logit: en cada
partición, ¿el candidato que mejor rindió in-sample quedó por encima o
por debajo de la mediana out-of-sample? PBO = fracción de particiones
donde quedó por debajo.

- UI: nueva pestaña **"CPCV (Sobreajuste)"** en Optimización, con
  gráfico IS-vs-OOS de todos los candidatos + histograma, y comparación
  lado a lado entre "mejor candidato in-sample" vs "mejor out-of-sample".
- Es **computacionalmente más caro** que WFO: N=6,k=2 son 15
  particiones × n_candidatos backtests × 2 (train+test). La UI avisa si
  la combinación pedida supera 45 particiones.

### B.3 — Tests ampliados

**Nuevo:** `tests/test_cpcv_fdr.py` (19 tests), `tests/test_portfolio_optimizer.py`
(16 tests), `tests/test_execution.py` (15 tests, ver Bloque D). Cubren FDR,
construcción de particiones CPCV (sin fuga de datos, purga/embargo
correctos), motor CPCV end-to-end y PBO.

---

## Bloque C — Fase 2: Portafolio

### Hallazgo importante antes de empezar

`PortfolioManager` (correlaciones, descomposición de riesgo, métricas
consolidadas) **ya existía y funcionaba**, pero **no estaba conectado a
ninguna pantalla** — cero líneas en `ui/` lo usaban. `ui/views/portfolio.py`
era solo un visor de gráficos independientes, sin ningún concepto de
"portafolio" real. Este bloque construye esa UI desde cero sobre el
backend ya existente, sin tocar `PortfolioManager`.

### C.1 — Optimización real de pesos

**Nuevo:** `portfolio/optimizer.py`

- **Risk Parity**: cada componente (activo::estrategia) contribuye lo
  mismo a la volatilidad total. No requiere estimar retornos esperados
  (mucho más ruidosos que la volatilidad) — solo la covarianza.
  Maillard, Roncalli & Teiletche (2010).
- **Mean-Variance (Markowitz)**: máximo Sharpe, mínima varianza, o
  retorno objetivo, sobre la frontera eficiente clásica. Sin cortos por
  default (`min_weight=0.0`).
- `decompose_component_weights()`: convierte pesos planos por
  componente en `(asset_weights, strategies_by_asset)` para alimentar
  `PortfolioManager.compute()` **sin modificar esa clase** — cada
  "activo" reconstruye exactamente su peso plano original vía
  `asset_weight × peso_normalizado_dentro_del_activo`.

### C.2 — Constructor de Portafolio (UI nueva)

`ui/views/portfolio.py` ahora tiene dos pestañas:
- **Visor Multi-Activo** (el contenido original, intacto).
- **Constructor de Portafolio** (nuevo): agregá componentes
  (activo+temporalidad+estrategia — pensado para juntar varias
  estrategias del Hall of Fame del Descubridor Genético), corré sus
  backtests, elegí método de ponderación (igual peso / paridad de
  riesgo / media-varianza máx. Sharpe / media-varianza mín. varianza /
  manual), y obtené: equity consolidada, métricas institucionales,
  **panel de correlación entre componentes** (`correlation_heatmap`,
  ya existía pero sin uso), descomposición de riesgo y tabla resumen.

---

## Bloque D — Fase 3: Ejecución hacia MT5 Demo

**La fase más delicada** — se construyó siguiendo EXACTAMENTE la
secuencia que pediste: solo lectura (ya existía) → orden manual → una
estrategia con risk overlay. **Deliberadamente no se construyó nada
más allá de eso** — ni multi-estrategia automática, ni cuentas reales,
ni loop autónomo sin supervisión. Eso es para "más adelante", como dijiste.

### Guardas de seguridad — no opcionales, en este orden, en TODA orden

1. **Cuenta debe ser DEMO.** `execution/mt5_execution.py` verifica
   `account_info().trade_mode == 0` (constante MT5 para DEMO) ANTES de
   tocar `order_send`. Si la cuenta es real, se rechaza sin excepción
   y sin bypass posible — ni siquiera hay un parámetro para saltarlo.
2. **Risk overlay obligatorio** (`execution/risk_overlay.py`):
   - Kill switch de pérdida diaria (default -5% del equity del día).
   - Límite de posiciones abiertas simultáneas (default 1 — "una
     estrategia a la vez", tal como pediste).
   - Stop-loss obligatorio en toda orden (sin SL, no hay orden).
   - Dimensionamiento por riesgo idéntico en filosofía al
     `_calculate_size` del backtester: `volumen = (equity × riesgo%) /
     distancia_al_stop`, con tope duro de % de equity y redondeo al
     step de lote del símbolo.
3. **Confirmación humana explícita.** `send_market_order(..., confirm=True)`
   — el parámetro `confirm` existe para que un re-render de Streamlit
   nunca pueda disparar una orden por accidente; `True` solo se origina
   en el manejador de un botón que el usuario acaba de apretar. En la
   UI, cada orden pasa por un segundo botón de "¿confirmás?" antes de
   enviarse — nunca hay envío en un solo clic.

### Qué se agregó en la UI (`ui/views/live.py`)

Nueva sección **"🎯 Ejecución (Demo)"**, visible solo si MT5 está
conectado, con 3 pestañas:
- **Señal de la estrategia**: si la estrategia (reproducida por el
  mismo motor de backtest de siempre) tiene una posición abierta ahora,
  muestra una orden sugerida (dirección, SL/TP tomados de la propia
  posición) ya dimensionada por el risk overlay — con confirmación de
  2 pasos para enviarla.
- **Ticket manual**: orden de mercado libre (dirección, SL, TP a mano),
  también dimensionada por el risk overlay, también con confirmación
  de 2 pasos.
- **Posiciones abiertas**: tabla en vivo + botón de cierre individual y
  botón de pánico "Cerrar todas las posiciones" (con confirmación).

Si la cuenta conectada NO es demo, esta sección se reemplaza por un
error claro y **todos** los controles de envío desaparecen — solo
queda la visualización de siempre.

### Qué NO se construyó (a propósito)

- Envío automático sin clic humano (ni para "una estrategia" ni para
  ninguna). El roadmap decía "recién después escalar" — esto se
  respetó al pie de la letra.
- Multi-estrategia simultánea en ejecución real (el risk overlay limita
  a 1 posición abierta por diseño, coincidiendo con "una estrategia").
- Cuentas reales: bloqueadas sin excepción en esta entrega.
- Fase 4 (multiusuario, FastAPI): fuera de alcance, como dijiste.

---

## Archivos nuevos

```
core/multiple_testing.py
optimization/cpcv.py
portfolio/optimizer.py
execution/__init__.py
execution/risk_overlay.py
execution/mt5_execution.py
tests/test_cpcv_fdr.py
tests/test_portfolio_optimizer.py
tests/test_execution.py
```

## Archivos modificados

```
discovery/discovery_engine.py    (FDR wireado tras el DSR)
ui/views/discovery.py            (columna/metric/manifiesto FDR)
ui/views/optimization.py         (pestaña CPCV)
ui/views/portfolio.py            (pestaña Constructor de Portafolio)
ui/views/live.py                 (sección Ejecución Demo)
ui/state.py                      (save_cpcv/get_cpcv)
market_data/mt5_provider.py      (account_info(), get_raw_mt5())
visualization/optimization_charts.py  (cpcv_results_chart)
```

## Antes de operar en demo — checklist recomendado

1. `pip install -r requirements.txt` en tu entorno real (Windows, para MT5).
2. `pytest tests/ -v` — deberían pasar 100/100.
3. Abrí la app, conectá MT5 a una cuenta **demo**, confirmá en pantalla
   que el badge dice "(DEMO)" antes de tocar cualquier botón de envío.
4. Probá primero con volumen mínimo y un símbolo líquido — la primera
   orden real (aunque sea demo) siempre conviene verificarla a mano
   contra el terminal de MT5 directamente.
