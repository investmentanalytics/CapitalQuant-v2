# CapitalQuant 1.11

Base: CapitalQuant 1.10 UPGRADED_READY.

## Cambios de esta versión
- Importación de CSV propios desde Gráficos.
- Registro persistente de CSV para reutilización en módulos históricos.
- Activos importados aparecen en el selector global aunque no existan en MT5.
- Fuente histórica `CSV importados` disponible en Backtesting/Optimización/Research Lab/Descubridor.
- Corrección del panel Pine Script: elimina el import roto de `_load_registry`.
- Pine Script v6 con validación previa de condiciones soportadas, DNF, conflicto LONG/SHORT y SL/TP con ATR congelado en la entrada.

## Intencionalmente NO cambiado
- Arquitectura de Research Lab/Genetic Discoverer respecto a la versión base.
- Ciclo de renderizado de Streamlit.
- Motor de backtesting.
- Optimización, WFO, CPCV, Monte Carlo y métricas existentes.
