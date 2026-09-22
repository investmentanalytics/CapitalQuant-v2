# CapitalQuant Research Lab 1.8

## Cambios

- Descubridor Genético: el selector de objetivo de fitness ahora incluye Calmar, Sharpe, Profit Factor, Win Rate, CAGR, Expectancy, Drawdown, Stability y Robustness, además de los objetivos históricos Composite, High Win Rate + Frequency y High Profitability.
- Evolución en vivo: cada generación expone un snapshot serializable del Hall of Fame para que la UI actualice una tabla de candidatos mientras el genético sigue trabajando.
- Research Lab: la preparación de datos limpios se muestra como tabla de estados por activo, y la investigación genética muestra los mejores candidatos encontrados en vivo por generación.
- No se cambia la lógica de fitness, selección, backtesting ni la semántica de ejecución; el snapshot es solo visualización.
- Los datos siguen viniendo exclusivamente del registro de datasets limpios enviados desde Gráficos.
