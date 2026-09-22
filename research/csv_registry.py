"""Registro persistente de datasets importados por el usuario desde CSV.

Los CSV importados se normalizan a OHLCV y se almacenan como parquet/pickle
para que puedan reutilizarse desde Gráficos, Backtesting, Optimización,
Research Lab y Descubridor sin volver a leer el archivo original.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.validators import normalize_ohlcv, validate_ohlcv, DataValidationError

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "csv_datasets"
REGISTRY_PATH = STORE / "csv_dataset_registry.json"
STORE.mkdir(parents=True, exist_ok=True)


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_registry(data: dict) -> None:
    REGISTRY_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _safe_stem(asset: str, timeframe: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{asset}__{timeframe}").strip("_").lower()


def _parse_csv(file_or_path) -> pd.DataFrame:
    """Carga CSV/TSV de mercado con compatibilidad explícita con exportaciones MT5.

    Formato MT5 típico:
        <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
        2020.01.01\t00:00:00\t128.18\t130.19\t127.50\t130.18\t5312\t0\t1

    También acepta CSV separados por coma/; y esquemas con datetime o date+time.
    """
    if hasattr(file_or_path, "seek"):
        file_or_path.seek(0)

    # Detectamos el delimitador a partir de la primera línea. El exportador
    # de MT5 usa TAB y, en algunos casos, pandas con sep=None no lo detecta
    # de forma fiable cuando los encabezados llevan <...>.
    sample = ""
    try:
        if hasattr(file_or_path, "read"):
            pos = file_or_path.tell() if hasattr(file_or_path, "tell") else 0
            raw = file_or_path.read(4096)
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8-sig", errors="replace")
            sample = str(raw)
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(pos)
        else:
            with open(file_or_path, "r", encoding="utf-8-sig", errors="replace") as fh:
                sample = fh.read(4096)
    except Exception:
        sample = ""

    first_line = sample.splitlines()[0] if sample else ""
    if "\t" in first_line:
        sep = "\t"
    elif ";" in first_line:
        sep = ";"
    elif "," in first_line:
        sep = ","
    else:
        sep = None

    if hasattr(file_or_path, "seek"):
        file_or_path.seek(0)
    if sep is None:
        df = pd.read_csv(file_or_path, sep=None, engine="python")
    else:
        df = pd.read_csv(file_or_path, sep=sep, engine="python")

    if df.empty:
        raise DataValidationError("El CSV está vacío.")

    # Encabezados MT5: <DATE>, <TIME>, <OPEN>, ...
    # Quitamos < >, BOM, espacios y símbolos para obtener nombres canónicos.
    def _canon_col(c):
        c = str(c).replace("\ufeff", "").strip().lower()
        c = c.replace("<", "").replace(">", "")
        c = re.sub(r"[\s\-]+", "_", c)
        return c

    df.columns = [_canon_col(c) for c in df.columns]
    aliases = {
        "tick_volume": "volume", "tickvol": "volume",
        "real_volume": "real_volume", "vol": "vol",
        "datetime": "datetime", "date_time": "datetime", "timestamp": "datetime",
        "date": "date", "time": "time",
        "fecha": "date", "hora": "time",
        "open_price": "open", "high_price": "high", "low_price": "low",
        "close_price": "close", "adj_close": "close", "adj_close_price": "close",
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # En MT5 suelen venir TICKVOL y VOL simultáneamente. Para el motor
    # cuantitativo usamos TICKVOL como volume y conservamos VOL aparte.
    # Esto evita crear dos columnas llamadas "volume".
    if "volume" in df.columns and "vol" in df.columns:
        df.rename(columns={"vol": "real_volume"}, inplace=True)
    elif "volume" not in df.columns and "vol" in df.columns:
        df.rename(columns={"vol": "volume"}, inplace=True)

    # Si el encabezado MT5 llegó como una sola columna (por archivos con
    # codificación/separador inusual), intentamos dividir las filas por TAB.
    if len(df.columns) == 1 and "\t" in str(df.columns[0]):
        raw_col = str(df.columns[0])
        df = df.iloc[:, 0].astype(str).str.split("\t", expand=True)
        df.columns = [_canon_col(c) for c in raw_col.split("\t")]

    if "datetime" not in df.columns:
        if "date" in df.columns and "time" in df.columns:
            df["datetime"] = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
        elif "date" in df.columns:
            df["datetime"] = df["date"]
        elif "time" in df.columns:
            df["datetime"] = df["time"]
        elif len(df.columns) and str(df.columns[0]).startswith("unnamed"):
            df["datetime"] = df.iloc[:, 0]
        else:
            raise DataValidationError(
                "No encontré una columna temporal. Se admiten exportaciones MT5 "
                "(<DATE> + <TIME>), 'datetime', 'date'+'time' o una primera columna de índice."
            )

    # MT5 usa fechas YYYY.MM.DD; pandas las reconoce, pero hacemos explícita
    # la conversión flexible y eliminamos únicamente filas temporalmente inválidas.
    dt_raw = df.pop("datetime").astype(str).str.strip()
    dt = pd.to_datetime(dt_raw, errors="coerce", utc=True, format="mixed")
    if dt.notna().sum() == 0:
        # Fallback para versiones antiguas de pandas / formatos mixtos.
        dt = pd.to_datetime(dt_raw, errors="coerce", utc=True)
    df.index = dt
    df.index.name = "datetime"
    df = df[~df.index.isna()]

    # SPREAD es informativo para el histórico; no debe impedir el análisis.
    # volume puede provenir de TICKVOL o VOL (se prioriza TICKVOL mediante alias).
    df = normalize_ohlcv(df)
    validate_ohlcv(df, "CSV importado")
    return df


def register_csv(file_or_path, asset: str, timeframe: str, original_name: str = "") -> dict:
    asset = str(asset).strip().upper()
    timeframe = str(timeframe).strip()
    if not asset:
        raise DataValidationError("El activo no puede estar vacío.")
    if not timeframe:
        raise DataValidationError("La temporalidad no puede estar vacía.")

    df = _parse_csv(file_or_path)
    stem = _safe_stem(asset, timeframe)
    path = STORE / f"{stem}.parquet"
    try:
        df.to_parquet(path, index=True)
        fmt = "parquet"
    except (ImportError, ValueError):
        path = STORE / f"{stem}.pkl"
        df.to_pickle(path)
        fmt = "pickle"

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    reg = _load_registry()
    key = f"{asset}|{timeframe}"
    registered_at = reg.get(key, {}).get("registered_at", now)
    reg[key] = {
        "asset": asset,
        "timeframe": timeframe,
        "path": str(path.relative_to(ROOT)),
        "storage_format": fmt,
        "source": "CSV del usuario",
        "original_name": original_name or "",
        "status": "csv_imported",
        "bars": int(len(df)),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "registered_at": registered_at,
        "updated_at": now,
    }
    _save_registry(reg)
    return reg[key]


def _infer_bundled_csv_meta(path: Path) -> dict | None:
    """Descubre CSV que vienen versionados directamente dentro del repo.

    Convención recomendada: ASSET_TIMEFRAME.csv o ASSET-TIMEFRAME.csv,
    por ejemplo XAUUSD_4h.csv, BTCUSD_1d.csv o NASDAQ-15m.csv.
    """
    stem = path.stem
    parts = re.split(r"[_\- ]+", stem)
    if len(parts) < 2:
        return None
    aliases = {
        "m1":"1m","m5":"5m","m15":"15m","m30":"30m",
        "h1":"1h","h2":"2h","h4":"4h","h6":"6h","h12":"12h",
        "d1":"1d","w1":"1w","mn1":"1M",
    }
    tf = None; tf_part = None
    for part in parts:
        key = part.strip()
        low = key.lower()
        if key in {"1m","5m","15m","30m","1h","2h","4h","6h","12h","1d","1w","1M"}:
            tf, tf_part = key, key
            break
        if low in aliases:
            tf, tf_part = aliases[low], key
            break
    if not tf:
        return None
    idx = parts.index(tf_part)
    asset = "_".join(parts[:idx]).strip().upper()
    if not asset:
        return None
    return {
        "asset": asset, "timeframe": tf,
        "path": str(path.relative_to(ROOT)), "storage_format": "csv",
        "source": "CSV versionado en repositorio", "original_name": path.name,
        "status": "csv_imported",
    }


def list_csv_datasets(timeframe: str | None = None) -> list[dict]:
    reg = _load_registry()
    out = []
    seen = set()
    for meta in reg.values():
        if meta.get("status") != "csv_imported":
            continue
        if timeframe and meta.get("timeframe") != timeframe:
            continue
        path = ROOT / meta.get("path", "")
        if path.exists():
            item = dict(meta)
            seen.add(str(path.resolve()))
            out.append(item)

    # En GitHub no existe necesariamente el registro JSON generado por una
    # subida web. Por eso también descubrimos CSV que ya estén en el repo.
    STORE.mkdir(parents=True, exist_ok=True)
    for path in STORE.glob("*.csv"):
        if str(path.resolve()) in seen:
            continue
        meta = _infer_bundled_csv_meta(path)
        if not meta or (timeframe and meta["timeframe"] != timeframe):
            continue
        try:
            df = _parse_csv(path)
            meta.update({"bars": int(len(df)), "start": str(df.index.min()), "end": str(df.index.max())})
        except Exception:
            continue
        out.append(meta)
    return sorted(out, key=lambda x: (x.get("asset", ""), x.get("timeframe", "")))


def load_csv_dataset(asset: str, timeframe: str, date_from=None, date_to=None) -> pd.DataFrame | None:
    reg = _load_registry()
    meta = reg.get(f"{asset}|{timeframe}")
    path = None
    if meta:
        path = ROOT / meta.get("path", "")
    else:
        for candidate in STORE.glob("*.csv"):
            inferred = _infer_bundled_csv_meta(candidate)
            if inferred and inferred.get("asset") == str(asset).upper() and inferred.get("timeframe") == timeframe:
                meta = inferred
                path = candidate
                break
    if not meta or path is None:
        return None
    if not path.exists():
        return None
    if path.suffix == ".csv":
        df = _parse_csv(path)
    else:
        df = pd.read_pickle(path) if path.suffix == ".pkl" else pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, errors="raise", utc=True)
    if date_from is not None:
        df = df.loc[df.index >= pd.Timestamp(date_from, tz="UTC")]
    if date_to is not None:
        df = df.loc[df.index <= pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)]
    return df
