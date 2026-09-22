# CapitalQuant Research Lab 1.7 — Correcciones y persistencia

## Correcciones críticas
- `visualization/charts.py`: `indicators_chart()` ya no intenta calcular cuantiles sobre series booleanas. Las series se filtran/convierten a numéricas y se eliminan infinitos, evitando el error `numpy boolean subtract`.
- `ui/views/discovery.py`: `_symbol_universe()` recibe correctamente la temporalidad desde `render()`, evitando `NameError: timeframe is not defined`.
- Research Lab: se eliminó la incompatibilidad que enviaba `verify_with_real_engine` a `GeneticConfig`.

## Verificación genética controlada
- La evolución sigue usando el motor genético para no multiplicar el coste computacional.
- Cuando está activada la verificación real, solo los mejores `N` candidatos del Hall of Fame pasan por el `BacktestEngine` real.
- El número de candidatos se controla desde Research Lab.
- Un candidato solo se marca como verificado si el motor real obtiene al menos el mínimo de operaciones configurado.

## Research Lab
- Los datasets siguen llegando exclusivamente desde `Gráficos → datos limpios`.
- El backtest rápido de una estrategia seleccionada usa el motor real.
- El preview muestra métricas, tabla estadística completa, precio con entradas/salidas/SL/TP, indicadores, curva de equity y tabla de operaciones.
- Si la estrategia fue investigada por régimen, el preview reaplica el filtro de entradas de ese régimen.
- Guardar una estrategia primero valida que el código generado pueda compilarse; después la registra en `strategies/` y en el manifiesto persistente.
- Tras guardar, se ofrecen atajos a Backtesting, Optimización y Mercado en Vivo.
- El estado de guardado queda reflejado en el lote JSON.

## Persistencia
- El último lote de Research Lab se registra en `data/research/_latest.json` y se recupera al volver a la página.
- Los resultados del backtest de las páginas se mantienen en `AppState` durante la sesión; cambiar de página no los borra.
- Los IDs de Research Lab incorporan un sufijo único para evitar colisiones si se lanzan dos investigaciones en el mismo segundo.
- Se añadieron campos de progreso/mensaje por investigación para que la interfaz pueda mostrar avance por generación, no únicamente al terminar un job.

## Principio metodológico
La verificación real es una etapa final, no parte del loop de millones de candidatos. El `BacktestEngine` continúa siendo la referencia para entradas/salidas y estadísticas finales; el motor genético sigue siendo el mecanismo de búsqueda.
