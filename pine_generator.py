from __future__ import annotations
import re

PINE_VERSION = 6

# Condiciones soportadas por el motor genético. La función genera expresiones
# Pine equivalentes y evita depender de funciones privadas del registro.

def _condition_expr(cid: str) -> str:
    m = re.fullmatch(r"rsi(7|14|21)_below_(20|25|30)", cid)
    if m: return f"rsi_{m.group(1)} < {m.group(2)}"
    m = re.fullmatch(r"rsi(7|14|21)_above_(70|75|80)", cid)
    if m: return f"rsi_{m.group(1)} > {m.group(2)}"
    m = re.fullmatch(r"rsi(7|14|21)_cross_(up|down)_50", cid)
    if m: return f"ta.{'crossover' if m.group(2)=='up' else 'crossunder'}(rsi_{m.group(1)}, 50)"
    fixed = {
        "macd_cross_up_signal":"ta.crossover(macdLine, macdSignal)","macd_cross_down_signal":"ta.crossunder(macdLine, macdSignal)",
        "macd_hist_positive":"macdHist > 0","macd_hist_negative":"macdHist < 0","macd_above_zero":"macdLine > 0","macd_below_zero":"macdLine < 0",
        "plus_di_above_minus_di":"plusDI > minusDI","minus_di_above_plus_di":"minusDI > plusDI",
        "plus_di_cross_up_minus_di":"ta.crossover(plusDI, minusDI)","minus_di_cross_up_plus_di":"ta.crossover(minusDI, plusDI)",
        "bb_pctb_oversold":"bbPctB < 0.05","bb_pctb_overbought":"bbPctB > 0.95","bb_breakout_up":"ta.crossover(close, bbUpper)","bb_breakout_down":"ta.crossunder(close, bbLower)",
        "bb_squeeze":"bbWidth < ta.percentile_nearest_rank(bbWidth, 100, 20)",
        "donchian_breakout_up":"close > donchianUpper[1]","donchian_breakout_down":"close < donchianLower[1]",
        "stoch_oversold":"stochK < 20","stoch_overbought":"stochK > 80","stoch_cross_up":"ta.crossover(stochK, stochD)","stoch_cross_down":"ta.crossunder(stochK, stochD)",
        "willr_oversold":"willr < -80","willr_overbought":"willr > -20","obv_rising":"obvValue > obvValue[10]","obv_falling":"obvValue < obvValue[10]",
        "volume_spike":"volume > ta.sma(volume, 20) * 1.5",
        "regime_is_bullish":"regimeBull","regime_is_bearish":"regimeBear","regime_is_consolidation":"regimeFlat","regime_confidence_high":"regimeConfidence > 0.6",
        "regime_strength_bullish":"regimeStrength > 0.3","regime_strength_bearish":"regimeStrength < -0.3",
        "keltner_breakout_up":"ta.crossover(close, keltnerUpper)","keltner_breakout_down":"ta.crossunder(close, keltnerLower)","keltner_inside":"close <= keltnerUpper and close >= keltnerLower",
        "supertrend_bullish":"supertrendDir > 0","supertrend_bearish":"supertrendDir < 0","supertrend_flip_bullish":"supertrendDir[1] <= 0 and supertrendDir > 0","supertrend_flip_bearish":"supertrendDir[1] >= 0 and supertrendDir < 0",
        "ichimoku_tenkan_cross_up_kijun":"ta.crossover(tenkan, kijun)","ichimoku_tenkan_cross_down_kijun":"ta.crossunder(tenkan, kijun)","ichimoku_price_above_cloud":"close > math.max(senkouA, senkouB)","ichimoku_price_below_cloud":"close < math.min(senkouA, senkouB)",
        "psar_bullish":"psar < close","psar_bearish":"psar > close","psar_flip_bullish":"psar[1] >= close[1] and psar < close","psar_flip_bearish":"psar[1] <= close[1] and psar > close",
        "vol_percentile_low":"atrPercentile < 0.20","vol_percentile_high":"atrPercentile > 0.80",
        "bullish_engulfing":"close[1] < open[1] and close > open and close >= open[1] and open <= close[1]",
        "bearish_engulfing":"close[1] > open[1] and close < open and open >= close[1] and close <= open[1]",
        "pin_bar_bullish":"low < low[1] and (math.min(open, close)-low) > math.abs(close-open)*2",
        "pin_bar_bearish":"high > high[1] and (high-math.max(open, close)) > math.abs(close-open)*2",
    }
    if cid in fixed: return fixed[cid]
    m = re.fullmatch(r"(sma|ema)_(5|10|20|50|200)_above_(sma|ema)_(5|10|20|50|200)", cid)
    if m:
        return f"{m.group(1)}_{m.group(2)} > {m.group(3)}_{m.group(4)}"
    m = re.fullmatch(r"(sma|ema)_(5|10|20|50|200)_below_(sma|ema)_(5|10|20|50|200)", cid)
    if m:
        return f"{m.group(1)}_{m.group(2)} < {m.group(3)}_{m.group(4)}"
    m = re.fullmatch(r"(sma|ema)_(5|10|20|50|200)_cross_(up|down)_(sma|ema)_(5|10|20|50|200)", cid)
    if m:
        return f"ta.{'crossover' if m.group(3)=='up' else 'crossunder'}({m.group(1)}_{m.group(2)}, {m.group(4)}_{m.group(5)})"
    m = re.fullmatch(r"close_(above|below)_sma(20|50|100|200)", cid)
    if m: return f"close {'>' if m.group(1)=='above' else '<'} sma_{m.group(2)}"
    m = re.fullmatch(r"adx_(above|below)_(20|25|30)", cid)
    if m: return f"adxValue {'>' if m.group(1)=='above' else '<'} {m.group(2)}"
    m = re.fullmatch(r"cci_(below_neg|above_)(100|150|200)", cid)
    if m:
        op="<" if m.group(1).startswith("below") else ">"; val=f"-{m.group(2)}" if op=="<" else m.group(2)
        return f"cciValue {op} {val}"
    m = re.fullmatch(r"mom(10|20)_(positive|negative|cross_up_zero|cross_down_zero)", cid)
    if m:
        p=m.group(1); typ=m.group(2)
        if typ=="positive": return f"mom_{p} > 0"
        if typ=="negative": return f"mom_{p} < 0"
        return f"ta.{'crossover' if typ=='cross_up_zero' else 'crossunder'}(mom_{p},0)"
    m = re.fullmatch(r"zscore_(below_neg|above_)(1\.5|2\.0|2\.5)", cid)
    if m:
        val=m.group(2); return f"zscore20 {'<' if m.group(1).startswith('below') else '>'} {'-'+val if m.group(1).startswith('below') else val}"
    raise ValueError(f"Condición no soportada en Pine v6: {cid}")


def _clause_expr(clause):
    return " and ".join(f"({_condition_expr(cid)})" for cid in clause) if clause else "true"


def _rule_expr(rule):
    return " or ".join(f"({_clause_expr(c)})" for c in rule) if rule else "false"


def validate_definition_for_pine(definition: dict) -> list[str]:
    unsupported=[]
    for side in ("long_rule","short_rule"):
        for clause in definition.get(side) or []:
            for cid in clause:
                try: _condition_expr(cid)
                except ValueError: unsupported.append(cid)
    return sorted(set(unsupported))


def generate_pine_script(definition: dict, strategy_name: str="", asset: str="", timeframe: str="") -> str:
    unsupported = validate_definition_for_pine(definition)
    if unsupported:
        raise ValueError("Condiciones no soportadas en Pine: " + ", ".join(unsupported))
    sl=float(definition.get("stop_loss_atr",2.0)); tp=float(definition.get("take_profit_atr",3.0)); atr_p=int(definition.get("atr_period",14))
    name=str(strategy_name or "CapitalQuant Research Strategy").replace('"', "'")
    long_expr=_rule_expr(definition.get("long_rule") or []); short_expr=_rule_expr(definition.get("short_rule") or [])
    return f'''//@version=6
// CapitalQuant 1.11 -> TradingView Pine Script v6
// Origin asset: {asset} | timeframe: {timeframe}
// Rule semantics: DNF (OR of clauses; AND inside each clause).
// Execution convention: signal confirmed at bar close; order is filled by TradingView's broker emulator on the next available tick.
strategy("{name}", overlay=true, process_orders_on_close=false, pyramiding=0, initial_capital=100000, margin_long=100, margin_short=100)

// ----- Moving averages / RSI -----
sma_5=ta.sma(close,5)
sma_10=ta.sma(close,10)
sma_20=ta.sma(close,20)
sma_50=ta.sma(close,50)
sma_100=ta.sma(close,100)
sma_200=ta.sma(close,200)
ema_5=ta.ema(close,5)
ema_10=ta.ema(close,10)
ema_20=ta.ema(close,20)
ema_50=ta.ema(close,50)
ema_200=ta.ema(close,200)
rsi_7=ta.rsi(close,7)
rsi_14=ta.rsi(close,14)
rsi_21=ta.rsi(close,21)

// ----- MACD / ADX -----
[macdLine,macdSignal,macdHist]=ta.macd(close,12,26,9)
[plusDI,minusDI,adxValue]=ta.dmi(14,14)

// ----- Bollinger / Donchian -----
[bbBasis,bbUpper,bbLower]=ta.bb(close,20,2.0)
bbWidth=bbBasis!=0 ? (bbUpper-bbLower)/math.abs(bbBasis) : 0.0
bbPctB=bbUpper!=bbLower ? (close-bbLower)/(bbUpper-bbLower) : 0.5
donchianUpper=ta.highest(high,20)
donchianLower=ta.lowest(low,20)

// ----- Stochastic / CCI / Williams / Momentum / Z-score -----
rawStoch=ta.stoch(close,high,low,14)
stochK=ta.sma(rawStoch,3)
stochD=ta.sma(stochK,3)
willr=ta.wpr(14)
cciValue=ta.cci(hlc3,20)
mom_10=ta.mom(close,10)
mom_20=ta.mom(close,20)
mean20=ta.sma(close,20)
std20=ta.stdev(close,20)
zscore20=std20!=0 ? (close-mean20)/std20 : na

// ----- OBV / volume -----
obvValue=ta.cum(math.sign(ta.change(close))*volume)

// ----- Keltner / SuperTrend / Ichimoku / PSAR -----
keltnerBasis=ta.ema(close,20)
keltnerUpper=keltnerBasis+2.0*ta.atr(20)
keltnerLower=keltnerBasis-2.0*ta.atr(20)
[supertrendValue,supertrendDir]=ta.supertrend(3.0,10)
tenkan=(ta.highest(high,9)+ta.lowest(low,9))/2.0
kijun=(ta.highest(high,26)+ta.lowest(low,26))/2.0
senkouA=(tenkan+kijun)/2.0
senkouB=(ta.highest(high,52)+ta.lowest(low,52))/2.0
psar=ta.sar(0.02,0.02,0.2)

// ----- Volatility / regime -----
atr=ta.atr({atr_p})
atrPercentile=ta.percentrank(atr,100)/100.0
regimeFast=ta.ema(close,20)
regimeSlow=ta.ema(close,50)
regimeStrength=regimeSlow!=0 ? (regimeFast-regimeSlow)/regimeSlow : 0.0
regimeBull=regimeStrength>0.003
regimeBear=regimeStrength<-0.003
regimeFlat=not regimeBull and not regimeBear
regimeConfidence=math.min(1.0,math.abs(regimeStrength)/0.02)

longCondition={long_expr}
shortCondition={short_expr}
// CapitalQuant convention: if both sides trigger on the same bar, cancel both.
conflict=longCondition and shortCondition
longSignal=longCondition and not conflict
shortSignal=shortCondition and not conflict

if longSignal and strategy.position_size<=0
    strategy.entry("CQ Long",strategy.long)
if shortSignal and strategy.position_size>=0
    strategy.entry("CQ Short",strategy.short)

// Freeze SL/TP from the SIGNAL bar, matching CapitalQuant's next-bar execution model.
var float pendingLongSL=na
var float pendingLongTP=na
var float pendingShortSL=na
var float pendingShortTP=na
if longSignal
    pendingLongSL:=close-atr*{sl}
    pendingLongTP:=close+atr*{tp}
if shortSignal
    pendingShortSL:=close+atr*{sl}
    pendingShortTP:=close-atr*{tp}

var float activeSL=na
var float activeTP=na
if strategy.position_size!=strategy.position_size[1]
    if strategy.position_size>0
        activeSL:=pendingLongSL
        activeTP:=pendingLongTP
    else if strategy.position_size<0
        activeSL:=pendingShortSL
        activeTP:=pendingShortTP
    else
        activeSL:=na
        activeTP:=na
if strategy.position_size>0 and not na(activeSL)
    strategy.exit("CQ Long Exit","CQ Long",stop=activeSL,limit=activeTP)
if strategy.position_size<0 and not na(activeSL)
    strategy.exit("CQ Short Exit","CQ Short",stop=activeSL,limit=activeTP)

plotshape(longSignal,title="CapitalQuant LONG",style=shape.triangleup,location=location.belowbar,size=size.tiny,text="L")
plotshape(shortSignal,title="CapitalQuant SHORT",style=shape.triangledown,location=location.abovebar,size=size.tiny,text="S")
plot(activeSL,title="Active Stop",color=color.new(color.red,70),display=display.none)
plot(activeTP,title="Active Target",color=color.new(color.green,70),display=display.none)
'''
