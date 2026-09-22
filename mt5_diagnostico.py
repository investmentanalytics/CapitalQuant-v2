"""
mt5_diagnostico.py
Script independiente (NO depende de Streamlit ni del resto de CapitalQuant)
para ver exactamente qué responde la API de MetaTrader 5 al pedir historial,
símbolo por símbolo. Sirve para diferenciar:

  (a) un bug en la app  →  este script también fallaría igual, o
  (b) un límite real del terminal/bróker  →  este script mostrará
      claramente en qué fecha se corta el historial disponible.

CÓMO USARLO
-----------
1. Abre MetaTrader 5 en Windows y asegúrate de estar logueado (demo o real).
2. En una terminal, en la carpeta donde tengas Python:
       pip install MetaTrader5 pandas
3. Ejecuta:
       python mt5_diagnostico.py EURUSD 1d
   (cambia EURUSD y 1d por el símbolo/temporalidad que quieras probar;
   si no pasas nada, prueba con EURUSD 1d y BTCUSD 4h por defecto)
4. Copia y pégame TODA la salida que imprima — con eso puedo saber si es
   la app o el bróker/terminal.
"""
import sys
from datetime import datetime, timedelta

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: el paquete 'MetaTrader5' no está instalado o no estás en Windows.")
    print("Instálalo con:  pip install MetaTrader5")
    sys.exit(1)

TF_MAP = {
    "1m": mt5.TIMEFRAME_M1, "5m": mt5.TIMEFRAME_M5, "15m": mt5.TIMEFRAME_M15,
    "30m": mt5.TIMEFRAME_M30, "1h": mt5.TIMEFRAME_H1, "4h": mt5.TIMEFRAME_H4,
    "1d": mt5.TIMEFRAME_D1, "1w": mt5.TIMEFRAME_W1, "1M": mt5.TIMEFRAME_MN1,
}


def diagnosticar(symbol: str, tf_label: str) -> None:
    tf = TF_MAP.get(tf_label)
    if tf is None:
        print(f"  Temporalidad '{tf_label}' no reconocida en este script (usa: {list(TF_MAP)})")
        return

    print(f"\n{'='*70}\nSÍMBOLO: {symbol}   TEMPORALIDAD: {tf_label}\n{'='*70}")

    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"  ❌ symbol_info('{symbol}') = None → el bróker NO reconoce este símbolo "
              f"tal cual. Revisa mayúsculas/sufijos (ej. '{symbol}m', '{symbol}.a').")
        # Sugerir símbolos parecidos
        todos = mt5.symbols_get()
        if todos:
            parecidos = [s.name for s in todos if symbol.upper() in s.name.upper()][:10]
            if parecidos:
                print(f"  Símbolos del bróker que contienen '{symbol}': {parecidos}")
        return

    print(f"  symbol_info: visible={info.visible}  path={info.path}")
    if not info.visible:
        print("  → No estaba visible en Market Watch. Seleccionándolo con symbol_select()...")
        mt5.symbol_select(symbol, True)
        import time
        time.sleep(1.5)

    # 1) Últimas 10 velas — confirma que al menos hay datos recientes
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, 10)
    n = 0 if rates is None else len(rates)
    print(f"  copy_rates_from_pos(0, 10)  → {n} vela(s)."
          + ("" if n else f"  last_error={mt5.last_error()}"))

    # 2) Todo lo que el terminal tenga ya cacheado, de una sola vez
    rates_all = mt5.copy_rates_from_pos(symbol, tf, 0, 999_999)
    n_all = 0 if rates_all is None else len(rates_all)
    if n_all:
        primero = datetime.utcfromtimestamp(rates_all[0]["time"])
        ultimo = datetime.utcfromtimestamp(rates_all[-1]["time"])
        print(f"  copy_rates_from_pos(0, 999999) → {n_all} vela(s), de {primero} a {ultimo}")
    else:
        print(f"  copy_rates_from_pos(0, 999999) → 0 velas.  last_error={mt5.last_error()}")

    # 3) Recorrido hacia atrás por tramos de 1 año, hasta 15 años,
    #    igual que hace la app en modo "Todo el histórico" — para ver
    #    en qué año exacto el bróker deja de tener datos.
    print("  Recorrido año por año hacia atrás (copy_rates_range):")
    fin = datetime.now()
    for i in range(15):
        inicio = fin - timedelta(days=365)
        r = mt5.copy_rates_range(symbol, tf, inicio, fin)
        n_r = 0 if r is None else len(r)
        print(f"    [{inicio.date()} → {fin.date()}]  {n_r} vela(s)"
              + ("" if n_r else f"   last_error={mt5.last_error()}"))
        fin = inicio


if __name__ == "__main__":
    if not mt5.initialize():
        print(f"❌ No se pudo inicializar MT5: {mt5.last_error()}")
        print("Verifica que el terminal esté abierto y logueado en esta misma máquina.")
        sys.exit(1)

    term = mt5.terminal_info()
    acc = mt5.account_info()
    print(f"Terminal: {term.name} — build {term.build}")
    if acc:
        print(f"Cuenta: {acc.login} @ {acc.server}  ({'DEMO' if acc.trade_mode == 0 else 'REAL'})")

    args = sys.argv[1:]
    if len(args) >= 2:
        pares = [(args[0], args[1])]
    else:
        pares = [("EURUSD", "1d"), ("BTCUSD", "4h")]

    for symbol, tf_label in pares:
        diagnosticar(symbol, tf_label)

    mt5.shutdown()
