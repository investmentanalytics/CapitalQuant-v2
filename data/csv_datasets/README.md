# CSV locales de CapitalQuant

CapitalQuant acepta directamente los CSV exportados desde MetaTrader 5, además de CSV convencionales.

## Formato MT5 soportado

Encabezado típico:

```text
<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
```

Ejemplo:

```text
2020.01.01\t00:00:00\t128.18\t130.19\t127.50\t130.18\t5312\t0\t1
```

El importador reconoce automáticamente:

- TAB (`\\t`) de MetaTrader 5
- coma `,`
- punto y coma `;`
- `datetime`
- `date + time`
- fechas `YYYY.MM.DD`
- `TICKVOL` como `volume`
- `VOL` como `real_volume` cuando ambos están presentes
- `SPREAD` como columna informativa

## Nombres recomendados

Para que el repositorio detecte automáticamente activo y temporalidad:

```text
ETHUSD_H4_202001010000_202609131200.csv
XAUUSD_H4_202001010000_202609131200.csv
BTCUSD_H1_202001010000_202609131200.csv
AAPL_D1_202001010000_202609131200.csv
```

También funcionan nombres sencillos como `XAUUSD_4h.csv`.

Los datos se normalizan a un índice temporal UTC y a OHLCV antes de entrar al motor de CapitalQuant. No se rellenan artificialmente los huecos de mercado.
