"""
research/data_registry.py
Registro de datasets LIMPIOS que fueron preparados explícitamente desde
la página Gráficos y enviados al Research Lab.

Research Lab NO descarga datos de MT5. Solo puede investigar series que
aparezcan aquí. Cada registro apunta a un parquet persistente y conserva
metadatos de calidad/limpieza para que la procedencia sea auditable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "research_lab"
DATA_STORE = STORE / "clean_datasets"
REGISTRY_PATH = STORE / "clean_dataset_registry.json"
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
    REGISTRY_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _stem(asset: str, timeframe: str) -> str:
    safe_a = re.sub(r"[^A-Za-z0-9_.-]", "_", str(asset)).strip("_") or "asset"
    safe_t = re.sub(r"[^A-Za-z0-9_.-]", "_", str(timeframe)).strip("_") or "tf"
    return f"{safe_a}__{safe_t}".lower()


def register_clean_dataset(
    df: pd.DataFrame,
    asset: str,
    timeframe: str,
    *,
    cleaning_info: dict | None = None,
    source: str = "Gráficos",
) -> dict:
    """Persistir una serie ya verificada y registrarla como fuente permitida."""
    if df is None or df.empty:
        raise ValueError("No se puede registrar un dataset vacío.")
    out = df.copy().sort_index()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    elif out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out = out[~out.index.duplicated(keep="first")]
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Faltan columnas OHLC: {missing}")

    stem = _stem(asset, timeframe)
    # Parquet es preferido cuando pyarrow/fastparquet está instalado;
    # pickle permite que el registro siga funcionando incluso antes de
    # instalar dependencias opcionales. El formato queda declarado en el
    # manifiesto y nunca se mezcla con la caché MT5.
    try:
        path = DATA_STORE / f"{stem}.parquet"
        out.to_parquet(path, index=True)
        storage_format = "parquet"
    except (ImportError, ValueError):
        path = DATA_STORE / f"{stem}.pkl"
        out.to_pickle(path)
        storage_format = "pickle"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    info = dict(cleaning_info or {})
    meta = {
        "asset": str(asset),
        "timeframe": str(timeframe),
        "path": str(path.relative_to(ROOT)),
        "storage_format": storage_format,
        "source": source,
        "status": "clean",
        "bars": int(len(out)),
        "start": str(out.index.min()),
        "end": str(out.index.max()),
        "cleaning_info": info,
        "registered_at": now,
        "updated_at": now,
    }
    registry = _load_registry()
    key = f"{asset}|{timeframe}"
    if key in registry:
        meta["registered_at"] = registry[key].get("registered_at", now)
    registry[key] = meta
    _save_registry(registry)
    return meta


def list_clean_datasets(timeframe: str | None = None) -> list[dict]:
    registry = _load_registry()
    out=[]
    for meta in registry.values():
        if meta.get("status") != "clean":
            continue
        if timeframe and meta.get("timeframe") != timeframe:
            continue
        path = ROOT / meta.get("path", "")
        if not path.exists():
            continue
        out.append(dict(meta))
    out.sort(key=lambda x: (x.get("asset", ""), x.get("timeframe", "")))
    return out


def available_clean_assets(timeframe: str) -> list[str]:
    return [x["asset"] for x in list_clean_datasets(timeframe)]


def load_clean_dataset(asset: str, timeframe: str, date_from=None, date_to=None) -> pd.DataFrame | None:
    registry = _load_registry()
    meta = registry.get(f"{asset}|{timeframe}")
    if not meta or meta.get("status") != "clean":
        return None
    path = ROOT / meta.get("path", "")
    if not path.exists():
        return None
    if meta.get("storage_format") == "pickle" or path.suffix == ".pkl":
        df = pd.read_pickle(path)
    else:
        try:
            df = pd.read_parquet(path)
        except (ImportError, ValueError):
            # Un registro creado como parquet no se puede abrir sin el motor;
            # no se intenta tocar MT5 ni generar otra fuente silenciosamente.
            return None
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    if date_from is not None:
        start = pd.Timestamp(date_from, tz="UTC")
        df = df[df.index >= start]
    if date_to is not None:
        end = pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        df = df[df.index <= end]
    return df.sort_index()


def remove_clean_dataset(asset: str, timeframe: str) -> bool:
    registry = _load_registry()
    key=f"{asset}|{timeframe}"
    meta=registry.pop(key, None)
    if not meta:
        return False
    path=ROOT / meta.get("path", "")
    try:
        if path.exists(): path.unlink()
    except OSError:
        pass
    _save_registry(registry)
    return True
