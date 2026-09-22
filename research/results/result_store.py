from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import uuid

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "data" / "research"
STORE.mkdir(parents=True, exist_ok=True)
LATEST_PATH = STORE / "_latest.json"

def save_batch(batch_id: str, payload: dict) -> Path:
    path = STORE / f"{batch_id}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    # Persistimos cuál fue la última investigación terminada/iniciada para
    # que navegar entre páginas no borre el contexto y, además, una nueva
    # sesión pueda recuperar el último lote sin ejecutar otra búsqueda.
    LATEST_PATH.write_text(
        json.dumps({"batch_id": batch_id, "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path

def load_batch(batch_id: str) -> dict | None:
    path = STORE / f"{batch_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def load_latest_batch() -> dict | None:
    if not LATEST_PATH.exists():
        return None
    try:
        meta = json.loads(LATEST_PATH.read_text(encoding="utf-8"))
        batch_id = meta.get("batch_id")
        return load_batch(batch_id) if batch_id else None
    except Exception:
        return None

def new_batch_id() -> str:
    # El timestamp solo a segundos podía colisionar si se lanzaban dos lotes
    # rápidamente. El sufijo corto mantiene el ID legible y único.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"RQ-{stamp}-{uuid.uuid4().hex[:6].upper()}"
