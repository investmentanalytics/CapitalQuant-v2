"""
config/settings.py
Configuración centralizada de CapitalQuant.
Fuente de datos principal: MetaTrader 5.
"""
from pathlib import Path

ROOT_DIR    = Path(__file__).parent.parent
DATA_DIR    = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
OPT_DIR     = REPORTS_DIR / "optimizations"
STRATEGIES_DIR = ROOT_DIR / "strategies"

for d in [DATA_DIR, OPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Timeframes — referenciados desde MT5_LABELS en mt5_provider.py
# Se mantiene este dict para compatibilidad con módulos que lo usan
# ---------------------------------------------------------------------------
TIMEFRAMES: dict[str, dict] = {
    "1m":  {"label": "1 Minuto",   "folder": "1m",   "ann_factor": 252 * 1440},
    "2m":  {"label": "2 Minutos",  "folder": "2m",   "ann_factor": 252 * 720},
    "3m":  {"label": "3 Minutos",  "folder": "3m",   "ann_factor": 252 * 480},
    "5m":  {"label": "5 Minutos",  "folder": "5m",   "ann_factor": 252 * 288},
    "10m": {"label": "10 Minutos", "folder": "10m",  "ann_factor": 252 * 144},
    "15m": {"label": "15 Minutos", "folder": "15m",  "ann_factor": 252 * 96},
    "30m": {"label": "30 Minutos", "folder": "30m",  "ann_factor": 252 * 48},
    "1h":  {"label": "1 Hora",     "folder": "1h",   "ann_factor": 252 * 24},
    "2h":  {"label": "2 Horas",    "folder": "2h",   "ann_factor": 252 * 12},
    "3h":  {"label": "3 Horas",    "folder": "3h",   "ann_factor": 252 * 8},
    "4h":  {"label": "4 Horas",    "folder": "4h",   "ann_factor": 252 * 6},
    "6h":  {"label": "6 Horas",    "folder": "6h",   "ann_factor": 252 * 4},
    "8h":  {"label": "8 Horas",    "folder": "8h",   "ann_factor": 252 * 3},
    "12h": {"label": "12 Horas",   "folder": "12h",  "ann_factor": 252 * 2},
    "1d":  {"label": "Diario",     "folder": "daily","ann_factor": 252},
    "1w":  {"label": "Semanal",    "folder": "1w",   "ann_factor": 52},
    "1M":  {"label": "Mensual",    "folder": "1M",   "ann_factor": 12},
}

BACKTEST_DEFAULTS = {
    "initial_capital": 100_000.0,
    "commission":      0.001,
    "slippage":        0.0005,
    "risk_per_trade":  0.02,
    "allow_short":     True,
}

OPTIMIZER_DEFAULTS = {
    "n_trials":   100,
    "objective":  "sharpe",
    "n_jobs":     1,
    "min_trades": 10,
}

# Tema visual: blanco + amarillo, texto negro. Existe también una variante
# oscura para cuando el usuario activa el modo oscuro en la barra superior
# (ver ui/app.py y visualization/theme.get_active_theme()).
THEME = {
    "bg":      "#ffffff", "surface": "#ffffff", "border": "#e3e1d3",
    "text":    "#14140c", "accent":  "#b8860b", "green":  "#1a9850",
    "red":     "#d1403d", "yellow":  "#f2c200", "grid":   "#efeee2",
}

THEME_DARK = {
    "bg":      "#0f0e0a", "surface": "#181611", "border": "#332f22",
    "text":    "#f5f3e9", "accent":  "#f2c200", "green":  "#2ecc71",
    "red":     "#e5544f", "yellow":  "#ffd633", "grid":   "#26221a",
}

STRATEGY_COLORS = [
    "#b8860b", "#14140c", "#d1403d", "#1a9850", "#8a6d00",
    "#6f6c62", "#c99a00", "#4c4a44", "#a67c00", "#2f6f4e",
]

# ---------------------------------------------------------------------------
# Activos predeterminados (usados como fallback cuando MT5 no está conectado)
# Con MT5 activo, la lista de símbolos viene del broker dinámicamente.
# ---------------------------------------------------------------------------
ASSETS: dict[str, dict] = {
    # --- Cripto ---
    "BTCUSD":  {"name": "Bitcoin",   "type": "crypto"},
    "ETHUSD":  {"name": "Ethereum",   "type": "crypto"},
    "XRPUSD":  {"name": "Ripple",   "type": "crypto"},
    "SOLUSD":  {"name": "Solana",   "type": "crypto"},
    "BNBUSD":  {"name": "BNB",   "type": "crypto"},
    "ADAUSD":  {"name": "Cardano",   "type": "crypto"},
    "DOGEUSD": {"name": "Dogecoin",  "type": "crypto"},
    "AVAXUSD": {"name": "Avalanche",  "type": "crypto"},
    "DOTUSD":  {"name": "Polkadot",   "type": "crypto"},
    "LTCUSD":  {"name": "Litecoin",   "type": "crypto"},
    "LINKUSD": {"name": "Chainlink",  "type": "crypto"},

    # --- Comodities (futuros) ---
    "XAUUSD":  {"name": "Oro",      "type": "commodity"},
    "XAGUSD":  {"name": "Plata",      "type": "commodity"},
    "PALL":    {"name": "Paladio",      "type": "commodity"},
    "PLAT":    {"name": "Platino",      "type": "commodity"},
    "WTI":     {"name": "Petróleo WTI",      "type": "commodity"},
    "BRENT":   {"name": "Petróleo Brent",      "type": "commodity"},
    "NATGAS":  {"name": "Gas Natural",      "type": "commodity"},
    "COPPER":  {"name": "Cobre",      "type": "commodity"},
    "CORN":    {"name": "Maíz",      "type": "commodity"},
    "WHEAT":   {"name": "Trigo",      "type": "commodity"},
    "SOYBEAN": {"name": "Soja",      "type": "commodity"},
    "COFFEE":  {"name": "Café",      "type": "commodity"},
    "SUGAR":   {"name": "Azúcar",      "type": "commodity"},
    "COCOA":   {"name": "Cacao",      "type": "commodity"},
    "COTTON":  {"name": "Algodón",      "type": "commodity"},

    # --- Divisas (Forex) ---
    "EURUSD":  {"name": "Euro / Dólar", "type": "forex"},
    "GBPUSD":  {"name": "Libra / Dólar", "type": "forex"},
    "USDJPY":  {"name": "Dólar / Yen", "type": "forex"},
    "USDCHF":  {"name": "Dólar / Franco", "type": "forex"},
    "AUDUSD":  {"name": "Dólar Aus. / USD", "type": "forex"},
    "USDCAD":  {"name": "Dólar / Dólar Can.", "type": "forex"},
    "NZDUSD":  {"name": "Dólar NZ / USD", "type": "forex"},
    "EURGBP":  {"name": "Euro / Libra", "type": "forex"},
    "EURJPY":  {"name": "Euro / Yen", "type": "forex"},
    "USDMXN":  {"name": "Dólar / Peso Mex.", "type": "forex"},
    "USDCOP":  {"name": "Dólar / Peso Col.", "type": "forex"},
    "USDBRL":  {"name": "Dólar / Real", "type": "forex"},

    # --- Índices ---
    "SP500":   {"name": "S&P 500",  "type": "index"},
    "NASDAQ":  {"name": "NASDAQ 100",   "type": "index"},
    "DOWJONES":{"name": "Dow Jones",   "type": "index"},
    "RUSSELL2000": {"name": "Russell 2000",   "type": "index"},
    "VIX":     {"name": "VIX (Volatilidad)",  "type": "index"},
    "DAX":     {"name": "DAX (Alemania)", "type": "index"},
    "FTSE100": {"name": "FTSE 100 (RU)",  "type": "index"},
    "CAC40":   {"name": "CAC 40 (Francia)",  "type": "index"},
    "IBEX35":  {"name": "IBEX 35 (España)",  "type": "index"},
    "NIKKEI225": {"name": "Nikkei 225 (Japón)", "type": "index"},
    "HANGSENG": {"name": "Hang Seng (HK)",   "type": "index"},
    "BOVESPA": {"name": "Bovespa (Brasil)",  "type": "index"},

    # --- ETFs ---
    "SPY":  {"name": "SPDR S&P 500 ETF",  "type": "etf"},
    "QQQ":  {"name": "Invesco QQQ (Nasdaq)",  "type": "etf"},
    "DIA":  {"name": "SPDR Dow Jones ETF",  "type": "etf"},
    "IWM":  {"name": "iShares Russell 2000",  "type": "etf"},
    "GLD":  {"name": "SPDR Gold Shares",  "type": "etf"},
    "SLV":  {"name": "iShares Silver Trust",  "type": "etf"},
    "USO":  {"name": "United States Oil Fund",  "type": "etf"},
    "TLT":  {"name": "iShares 20+ Year Treasury", "type": "etf"},
    "HYG":  {"name": "iShares High Yield Bond",  "type": "etf"},
    "XLF":  {"name": "Financial Select ETF",  "type": "etf"},
    "XLE":  {"name": "Energy Select ETF",  "type": "etf"},
    "XLK":  {"name": "Technology Select ETF",  "type": "etf"},
    "ARKK": {"name": "ARK Innovation ETF", "type": "etf"},
    "EEM":  {"name": "iShares Emerging Mkts",  "type": "etf"},
    "VNQ":  {"name": "Vanguard Real Estate",  "type": "etf"},

    # --- Acciones: Tecnología ---
    "AAPL":  {"name": "Apple",  "type": "equity"},
    "MSFT":  {"name": "Microsoft",  "type": "equity"},
    "GOOGL": {"name": "Alphabet (Google)", "type": "equity"},
    "AMZN":  {"name": "Amazon",  "type": "equity"},
    "META":  {"name": "Meta Platforms",  "type": "equity"},
    "NVDA":  {"name": "NVIDIA",  "type": "equity"},
    "TSLA":  {"name": "Tesla",  "type": "equity"},
    "AMD":   {"name": "AMD",   "type": "equity"},
    "INTC":  {"name": "Intel",  "type": "equity"},
    "NFLX":  {"name": "Netflix",  "type": "equity"},
    "ORCL":  {"name": "Oracle",  "type": "equity"},
    "CRM":   {"name": "Salesforce",   "type": "equity"},
    "ADBE":  {"name": "Adobe",  "type": "equity"},
    "IBM":   {"name": "IBM",   "type": "equity"},
    "CSCO":  {"name": "Cisco",  "type": "equity"},
    "QCOM":  {"name": "Qualcomm",  "type": "equity"},
    "AVGO":  {"name": "Broadcom",  "type": "equity"},
    "SHOP":  {"name": "Shopify",  "type": "equity"},
    "UBER":  {"name": "Uber",  "type": "equity"},
    "PLTR":  {"name": "Palantir",  "type": "equity"},

    # --- Acciones: Finanzas ---
    "JPM":  {"name": "JPMorgan Chase",  "type": "equity"},
    "BAC":  {"name": "Bank of America",  "type": "equity"},
    "WFC":  {"name": "Wells Fargo",  "type": "equity"},
    "GS":   {"name": "Goldman Sachs",   "type": "equity"},
    "MS":   {"name": "Morgan Stanley",   "type": "equity"},
    "V":    {"name": "Visa",    "type": "equity"},
    "MA":   {"name": "Mastercard",   "type": "equity"},
    "PYPL": {"name": "PayPal", "type": "equity"},
    "AXP":  {"name": "American Express",  "type": "equity"},
    "BRK-B":{"name": "Berkshire Hathaway B", "type": "equity"},

    # --- Acciones: Consumo / Salud / Industria ---
    "WMT":   {"name": "Walmart",  "type": "equity"},
    "KO":    {"name": "Coca-Cola",   "type": "equity"},
    "PEP":   {"name": "PepsiCo",  "type": "equity"},
    "MCD":   {"name": "McDonald's",  "type": "equity"},
    "SBUX":  {"name": "Starbucks", "type": "equity"},
    "NKE":   {"name": "Nike",  "type": "equity"},
    "DIS":   {"name": "Disney",  "type": "equity"},
    "JNJ":   {"name": "Johnson & Johnson", "type": "equity"},
    "PFE":   {"name": "Pfizer",  "type": "equity"},
    "UNH":   {"name": "UnitedHealth",  "type": "equity"},
    "MRK":   {"name": "Merck",  "type": "equity"},
    "ABBV":  {"name": "AbbVie", "type": "equity"},
    "XOM":   {"name": "Exxon Mobil",  "type": "equity"},
    "CVX":   {"name": "Chevron",  "type": "equity"},
    "BA":    {"name": "Boeing",   "type": "equity"},
    "CAT":   {"name": "Caterpillar",  "type": "equity"},
    "GE":    {"name": "General Electric",   "type": "equity"},
    "F":     {"name": "Ford",    "type": "equity"},
    "GM":    {"name": "General Motors",   "type": "equity"},

    # --- Acciones: Latinoamérica / ADRs ---
    "MELI":  {"name": "MercadoLibre", "type": "equity"},
    "VALE":  {"name": "Vale S.A.", "type": "equity"},
    "PBR":   {"name": "Petrobras",  "type": "equity"},
    "ITUB":  {"name": "Itaú Unibanco", "type": "equity"},
    "ECOPETROL": {"name": "Ecopetrol",   "type": "equity"},
}

# Nota: la lista de arriba es solo un catálogo de referencia (nombres
# amigables por categoría: cripto, comodities, forex, índices, ETFs,
# acciones) usado como fallback cuando MT5 no está conectado. Con MT5
# conectado, el selector de activos de la barra superior no se limita a
# esta lista: muestra TODOS los símbolos que el broker ofrece (todo su
# Market Watch), vía `MT5Provider.get_available_symbols()`.

# ---------------------------------------------------------------------------
# Servidores de FBS conocidos, para el selector de "Trading Algorítmico".
# El usuario puede escribir cualquier otro servidor a mano si el suyo no
# está en esta lista corta (FBS tiene decenas de servidores regionales).
# ---------------------------------------------------------------------------
FBS_SERVER_PRESETS = ["FBS-Demo", "FBS-Real", "FBS-Real-2", "FBS-Real-3", "Otro (escribir manualmente)"]
