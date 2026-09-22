# CapitalQuant Research Lab — actualización 1.10+
## Cambios principales
- Fuente histórica ORIGINAL publicada desde Gráficos, conservando exactamente las filas recibidas de MT5.
- Fuente CLEAN independiente; ninguna reemplaza a la otra.
- Research Lab permite investigar tanto originales publicados como datasets limpios.
- Research Lab incorpora filtros de selección automática por trades, win rate, profit factor y drawdown, con orden por Fitness/Calmar/Sharpe/PF/Win Rate/CAGR/Expectancy.
- Exportación de estrategias seleccionadas en ZIP con JSON + Python + Pine Script v6.
- Las estrategias guardadas desde Research Lab persisten su `definition`.
- Optimización crea una variante independiente `<Estrategia> Optimizada` y no modifica la original.
- Pine Script Generator integrado, usando Pine Script v6, la versión vigente documentada por TradingView.
- Pine v6 exportado conserva la estructura DNF (OR entre cláusulas y AND dentro de cada cláusula), SL/TP ATR y ejecución de entrada al siguiente bar mediante `process_orders_on_close=false`.
## Validación
- `pytest -q`: 106 tests passed.
- `compileall`: OK.
## Importante
La equivalencia Python ↔ TradingView no es matemática en todos los casos: el broker emulator de TradingView, costes, margen y fills intrabar pueden diferir. El código exportado incluye esta advertencia y debe revalidarse en TradingView.
