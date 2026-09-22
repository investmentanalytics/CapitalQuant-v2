# CapitalQuant — Trading Algorítmico (reemplazo de Ejecución Demo manual)
### 30-31 jul 2026

## Qué se pidió
Reemplazar el panel "Ejecución (Demo)" (orden manual + confirmación por
clic, solo cuentas DEMO) por un módulo de **Trading Algorítmico**: login
directo a FBS (demo o real) con credenciales, selección de estrategia y
sus parámetros (incluidos periodos de indicadores), ajuste de riesgo, y
ejecución automática — el motor compra/vende solo según la señal de la
estrategia, sin confirmación por operación. Panel de monitoreo en vivo
estilo TradingView (velas + indicadores + señales + retículo + herramientas
de dibujo).

## Cambios

- **`market_data/mt5_provider.py`** — `connect()` ahora acepta
  `login/password/server` opcionales para iniciar sesión directa en una
  cuenta específica (antes solo se conectaba a la sesión ya abierta a mano
  en el terminal).
- **`execution/risk_overlay.py`** — `check_order()` ya no bloquea cuentas
  reales de forma absoluta: agrega `allow_real_account: bool = False`. Con
  `False` (default) se comporta igual que antes; con `True` permite operar
  en cuenta real. El resto de las guardas (SL obligatorio, kill switch
  diario, límite de posiciones, tamaño acotado) no cambian.
- **`execution/mt5_execution.py`** — `_require_demo()` →
  `_verify_account_allowed()`; `MT5ExecutionClient` acepta
  `allow_real_account` en el constructor. `DemoAccountRequiredError` ahora
  se lanza solo si la cuenta es real y no se confirmó explícitamente.
- **`execution/live_trader.py`** (nuevo) — `AlgoTradingEngine`: motor de
  una estrategia en ejecución automática, corre en un hilo de fondo,
  revalida el kill switch y todas las guardas en cada ciclo, y opera sin
  pedir confirmación por operación una vez iniciada la sesión.
- **`ui/views/algo_trading.py`** (nuevo, página "Trading Algorítmico") —
  login a FBS, selector de estrategia + parámetros dinámicos, sliders de
  riesgo/kill-switch/frecuencia, checkbox de confirmación explícita para
  cuenta real, botón Iniciar/Detener, gráfico en vivo con retículo y
  herramientas de dibujo, log de operaciones, panel de posición actual con
  cierre de emergencia manual.
- **`ui/views/live.py`** — se eliminó por completo el panel de ejecución
  manual (`_render_execution_panel`, `_render_order_confirmation`,
  `_get_execution_client`); ahora es solo visualización, con un aviso que
  apunta a la nueva página.
- **`visualization/charts.py`** — `candlestick_chart()` agrega el parámetro
  `crosshair: bool = False` (retículo con spikes en ambos ejes + estilo de
  `newshape` para las herramientas de dibujo); retrocompatible, default
  apagado.
- **`config/settings.py`** — `FBS_SERVER_PRESETS` para el selector de
  servidor en el formulario de login.
- **`ui/app.py` / `ui/components/sidebar.py`** — nueva entrada de
  navegación "Trading Algorítmico".
- **`tests/test_execution.py`** — actualizado para el nuevo modelo
  demo/real con opt-in explícito (antes probaba un bloqueo absoluto).

## Verificación hecha en esta sesión

- `python3 -m py_compile` sobre todo el árbol: sin errores de sintaxis.
- Import real de todos los módulos tocados (incluida `ui.views.algo_trading`
  con `streamlit` real instalado): sin `ImportError` ni `NameError`.
- **`pytest tests/ -v` real (no el runner casero de sesiones anteriores):
  102/102 tests pasando.**
- Prueba funcional end-to-end del motor (`AlgoTradingEngine`) con un
  proveedor MT5 simulado: abre posición cuando la señal cambia, no re-envía
  nada si la señal no cambió, cierra y abre la contraria si la señal se da
  vuelta, y el kill switch corta la operativa cuando la pérdida del día
  supera el límite configurado — los cuatro casos verificados manualmente
  con datos y una estrategia de prueba deterministas.

## Pendiente / próximos pasos sugeridos

- Si se quiere una experiencia 1:1 con TradingView (no solo "parecida"),
  migrar el gráfico de Plotly a la librería `lightweight-charts` de
  TradingView embebida vía HTML — es un cambio más grande, no incluido
  aquí.
- Correr esto en Windows con el terminal MT5 y una cuenta demo real de FBS
  antes de considerar la cuenta real — todo lo anterior se validó con un
  proveedor MT5 simulado (no hay MetaTrader5 disponible en este entorno
  Linux).
- Sigue pendiente el punto ya reportado en la auditoría anterior: la
  diversidad del Descubridor Genético no se tocó en esta sesión.
