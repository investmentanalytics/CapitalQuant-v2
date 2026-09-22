# CapitalQuant — GitHub / CSV local

Esta edición está preparada para ejecutarse sin MetaTrader 5.

## 1. Datos

Coloca los CSV en `data/csv_datasets/` siguiendo la convención `ACTIVO_TEMPORALIDAD.csv`.
Ejemplos: `XAUUSD_4h.csv`, `ETHUSD_1h.csv`, `US100_15m.csv`.

## 2. Streamlit

La aplicación de entrada es `ui/app.py`.

Instalación:

```bash
pip install -r requirements.txt
streamlit run ui/app.py
```

## 3. GitHub / Streamlit Cloud

Sube el repositorio completo. El archivo `requirements.txt` ya no contiene `MetaTrader5`.
Los CSV versionados se descubren automáticamente; no hace falta generar un registro JSON antes del despliegue.

También puedes usar `Gráficos → Importar CSV propio` durante una sesión. En un despliegue efímero esa subida vive en el almacenamiento temporal de la instancia, por lo que para que un dataset sea permanente debe quedar versionado dentro del repositorio o en un almacenamiento externo que se añada posteriormente.

## 4. Limitación intencional

`Mercado en Vivo` y cualquier ejecución real que dependa de un terminal MT5 no están disponibles en esta edición. El objetivo de esta variante es investigación/backtesting reproducible con datos locales.
