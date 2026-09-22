# Research Lab — configuración y datos

- La temporalidad se hereda del selector global de CapitalQuant; no se duplica en Research Lab.
- El Research Lab selecciona hasta 10 activos y, al iniciar, carga cada `(activo, temporalidad)` una sola vez en `AppState`.
- El mismo DataFrame cargado se entrega al `ResearchOrchestrator`; el orquestador no vuelve a pedir el mismo activo a MT5.
- El bloque `Motor genético` expone los mismos controles de `GeneticConfig` que `Descubridor Genético`: población, generaciones, torneo, élite, crossover, mutación, inmigrantes, estructura de reglas, riesgo, WFO, diversidad y verificación.
- El backtest de una estrategia es bajo demanda y usa el motor real de CapitalQuant. No forma parte de la búsqueda genética.
