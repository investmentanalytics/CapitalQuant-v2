# Compatibilidad CSV MT5 — CapitalQuant 1.11

Se corrigió el importador para el formato real de exportación de MetaTrader 5.

El error anterior:

`No encontré una columna temporal...`

ocurría porque algunos CSV de MT5 llegan separados por TAB y con encabezados entre `< >`. En determinadas condiciones pandas los interpretaba como una sola columna.

El parser ahora detecta explícitamente el delimitador antes de leer el archivo y normaliza:

`<DATE>` → `date`
`<TIME>` → `time`
`<OPEN>` → `open`
`<HIGH>` → `high`
`<LOW>` → `low`
`<CLOSE>` → `close`
`<TICKVOL>` → `volume`
`<VOL>` → `real_volume`
`<SPREAD>` → `spread`

Además interpreta correctamente fechas como `2020.01.01` y horas como `00:00:00`.

Archivo probado:

`ETHUSD_H4_202001010000_202609131200.csv`

Resultado de la prueba:

- 14.671 velas
- 2020-01-01 00:00 UTC → 2026-09-13 12:00 UTC
- OHLCV validado correctamente
- compatible con el motor de datos local

Suite del proyecto:

`106 passed`
