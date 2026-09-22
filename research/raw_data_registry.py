"""
Registro persistente de datasets originales publicados desde Gráficos.

La fuente ORIGINAL conserva las velas recibidas de MT5 antes de cualquier
depuración. No elimina duplicados, huecos ni filas: solo normaliza el índice
DatetimeIndex para que el resto de CapitalQuant pueda consumirla de forma
segura y reproducible.

La fuente CLEAN sigue siendo independiente.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "research_lab"
DATA_STORE = STORE / "raw_datasets"
REGISTRY_PATH = STORE / "raw_dataset_registry.json"
STORE.mkdir(parents=True, exist_ok=True)
DATA_STORE.mkdir(parents=True, exist_ok=True)


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_registry(data: dict) -> None:
    REGISTRY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _stem(asset: str, timeframe: str) -> str:
    a = re.sub(r"[^A-Za-z0-9_.-]", "_", str(asset)).strip("_") or "asset"
    t = re.sub(r"[^A-Za-z0-9_.-]", "_", str(timeframe)).strip("_") or "tf"
    return f"{a}__{t}".lower()


def register_raw_dataset(df: pd.DataFrame, asset: str, timeframe: str, *, source="MT5") -> dict:
    if df is None or df.empty:
        raise ValueError("No se puede publicar un dataset original vacío.")
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="raise", utc=True)
    elif out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    # IMPORTANTE: no deduplicamos, no eliminamos huecos y no filtramos filas.
    path = DATA_STORE / f"{_stem(asset, timeframe)}.parquet"
    try:
        out.to_parquet(path, index=True)
        fmt = "parquet"
    except (ImportError, ValueError):
        path = DATA_STORE / f"{_stem(asset, timeframe)}.pkl"
        out.to_pickle(path)
        fmt = "pickle"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {
        "asset": str(asset), "timeframe": str(timeframe),
        "path": str(path.relative_to(ROOT)), "storage_format": fmt,
        "source": source, "status": "raw_original",
        "bars": int(len(out)),
        "start": str(out.index.min()), "end": str(out.index.max()),
        "registered_at": now, "updated_at": now,
        "preserves_rows_exactly": True,
    }
    reg = _load_registry()
    key = f"{asset}|{timeframe}"
    if key in reg:
        meta["registered_at"] = reg[key].get("registered_at", now)
    reg[key] = meta
    _save_registry(reg)
    return meta


def list_raw_datasets(timeframe: str | None = None) -> list[dict]:
    reg = _load_registry()
    out = []
    for meta in reg.values():
        if meta.get("status") != "raw_original":
            continue
        if timeframe and meta.get("timeframe") != timeframe:
            continue
        path = ROOT / meta.get("path", "")
        if path.exists():
            out.append(dict(meta))
    return sorted(out, key=lambda x: (x.get("asset", ""), x.get("timeframe", "")))


def load_raw_dataset(asset: str, timeframe: str, date_from=None, date_to=None) -> pd.DataFrame | None:
    reg = _load_registry()
    meta = reg.get(f"{asset}|{timeframe}")
    if not meta:
        return None
    path = ROOT / meta.get("path", "")
    if not path.exists():
        return None
    df = pd.read_pickle(path) if path.suffix == ".pkl" else pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="raise", utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    if date_from is not None:
        df = df.loc[df.index >= pd.Timestamp(date_from, tz="UTC")]
    if date_to is not None:
        df = df.loc[df.index <= pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)]
    return df
