# CapitalQuant Research Lab 1.9 — resultados progresivos

Esta revisión cambia la experiencia de ejecución para que los módulos de cálculo publiquen resultados parciales durante el proceso, en lugar de reservar toda la visualización para el final.

## Backtesting

`engine/backtester.py` mantiene exactamente la semántica de ejecución existente y añade un callback opcional de progreso. La interfaz de Backtesting muestra durante el recorrido:

- vela actual / total;
- capital parcial;
- operaciones cerradas hasta ese momento;
- curva de equity parcial;
- gráfico de precio con las señales y operaciones que ya ocurrieron;
- tabla de operaciones acumulada.

Los snapshots se emiten cada 50 velas para evitar que el coste de renderizado de Streamlit se convierta en el nuevo cuello de botella. El último snapshot siempre se fuerza al terminar.

## Descubridor Genético

Después de cada generación se actualizan:

- mejor fitness;
- mejor estrategia encontrada;
- curva de evolución del fitness;
- Sharpe de la mejor estrategia;
- tabla del Hall of Fame vivo.

La visualización no cambia la selección, mutación, crossover ni función de fitness.

## Research Lab

Durante cada investigación se actualizan:

- estado de todos los trabajos del lote;
- progreso de cada trabajo;
- tabla de candidatos encontrados;
- curva de fitness/Sharpe del trabajo activo.

Los datasets siguen entrando desde el registro de datos limpios de Gráficos; Research Lab no descarga directamente de MT5.

## Optimización

### Optimización estándar
Optuna publica el historial de trials, TOP de configuraciones y evolución del score mientras avanza.

### Walk-Forward
Se muestran las ventanas ya completadas y las curvas OOS disponibles mientras se procesan las siguientes.

### CPCV
Se muestra la matriz incremental de partición/candidato con scores IS/OOS a medida que se completan las particiones.

### Monte Carlo
Las simulaciones se procesan por bloques y se actualizan las bandas P5/P25/P50/P75/P95 y la distribución del capital final conforme se completan los bloques.

### Sensibilidad
Los resultados 1D se van llenando punto a punto; en 2D se actualiza la matriz conforme se evalúan combinaciones.

## Principio metodológico

Los cambios son de observabilidad y streaming de resultados. No se sustituyó el motor genético por un algoritmo simplificado para acelerar artificialmente la búsqueda.
