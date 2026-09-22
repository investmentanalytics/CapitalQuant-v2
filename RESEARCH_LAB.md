# CapitalQuant Research Lab — cambios

- Nuevo módulo `research/` para orquestar investigaciones multi-activo.
- Máximo 10 activos por batch.
- Cada activo se investiga de forma independiente.
- Cada objetivo tiene una búsqueda genética independiente.
- Soporte: Calmar, Sharpe, Profit Factor, Win Rate, CAGR, Expectancy, Drawdown, Stability y Robustness.
- Soporte de histórico completo o rango de fechas.
- Soporte de investigaciones separadas por uno o varios regímenes causales.
- Los datos de cada activo se cargan una vez y se reutilizan en todos sus jobs.
- Resultados persistentes en `data/research/`.
- Nueva página `Research Lab` integrada al router.
- Se conserva Mercado en Vivo y Régimen de Mercado.
- No se agregaron dependencias nuevas.
