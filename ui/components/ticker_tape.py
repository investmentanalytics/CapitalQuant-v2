"""Cinta superior usando únicamente los CSV locales."""
import html as _html
import streamlit as st
from research.csv_registry import list_csv_datasets, load_csv_dataset

@st.cache_data(ttl=30, show_spinner=False)
def _local_quotes():
    metas=list_csv_datasets()
    preferred=["NASDAQ","SP500","XAUUSD","ETHUSD","BTCUSD","XRPUSD","EURUSD"]
    available={m.get("asset") for m in metas}
    symbols=[s for s in preferred if s in available] + sorted(available-set(preferred))[:8]
    out={}
    for s in symbols:
        tf=next((m.get("timeframe") for m in metas if m.get("asset")==s),None)
        if not tf: continue
        df=load_csv_dataset(s,tf)
        if df is not None and len(df)>=2:
            a,b=float(df["close"].iloc[-2]),float(df["close"].iloc[-1])
            out[s]=(b,((b-a)/a*100) if a else 0.0)
        elif df is not None and len(df)==1:
            out[s]=(float(df["close"].iloc[-1]),0.0)
    return out

def render_ticker_tape():
    quotes=_local_quotes()
    if not quotes:
        st.caption("CSV LOCAL · sin datasets cargados")
        return
    items=[]
    for s,(price,pct) in quotes.items():
        cls="cq-tk-up" if pct>0 else ("cq-tk-down" if pct<0 else "cq-tk-flat")
        items.append(f"<span class='cq-tk-item'><span class='cq-tk-label'>{_html.escape(s)}</span><span class='cq-tk-price'>{price:,.4f}</span><span class='{cls}'>{pct:+.2f}%</span></span>")
    track="".join(items)*2
    st.markdown(f"""<div class='cq-tk-wrap'><div class='cq-tk-track'>{track}</div></div>""", unsafe_allow_html=True)
