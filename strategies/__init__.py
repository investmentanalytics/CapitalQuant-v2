"""
strategies/__init__.py
Auto-descubrimiento de estrategias.

Para añadir una estrategia nueva: crea un archivo .py en strategies/
con una clase que herede de BaseStrategy. Aparece automáticamente
registrada en toda la plataforma, sin tocar ningún otro archivo.
"""
import importlib
import inspect
import sys
from pathlib import Path
from typing import Type, Dict

from .base import BaseStrategy, Indicators

_STRATEGIES_DIR = Path(__file__).parent


def discover_strategies() -> Dict[str, Type[BaseStrategy]]:
    registry: Dict[str, Type[BaseStrategy]] = {}
    for py_file in sorted(_STRATEGIES_DIR.glob("*.py")):
        if py_file.name.startswith("_") or py_file.name == "base.py":
            continue
        _load_from_file(py_file, f"strategies.{py_file.stem}", registry)
    return registry


def _load_from_file(py_file: Path, module_name: str, registry: dict) -> None:
    try:
        module = importlib.import_module(module_name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (issubclass(obj, BaseStrategy) and obj is not BaseStrategy
                    and hasattr(obj, "name")):
                registry[obj.name] = obj
    except Exception as e:
        print(f"[WARNING] No se pudo cargar {py_file.name}: {e}")


def refresh_registry() -> Dict[str, Type[BaseStrategy]]:
    """
    Reescanea strategies/*.py y actualiza STRATEGY_REGISTRY EN EL MISMO
    objeto dict (nunca lo reasigna) para que los `from strategies import
    STRATEGY_REGISTRY` ya hechos en otros módulos vean los cambios.

    Se llama después de guardar una estrategia nueva desde el Constructor
    de Estrategias, para que aparezca al instante en Backtesting,
    Optimización, etc. sin reiniciar la app.
    """
    fresh: Dict[str, Type[BaseStrategy]] = {}
    for py_file in sorted(_STRATEGIES_DIR.glob("*.py")):
        if py_file.name.startswith("_") or py_file.name == "base.py":
            continue
        module_name = f"strategies.{py_file.stem}"
        if module_name in sys.modules:
            try:
                importlib.reload(sys.modules[module_name])
            except Exception as e:
                print(f"[WARNING] No se pudo recargar {py_file.name}: {e}")
                continue
        _load_from_file(py_file, module_name, fresh)
    STRATEGY_REGISTRY.clear()
    STRATEGY_REGISTRY.update(fresh)
    return STRATEGY_REGISTRY


STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = discover_strategies()


def save_user_strategy_file(filename_stem: str, code: str) -> Path:
    """
    Guarda el código de una estrategia creada en el Constructor como un
    archivo .py real dentro de strategies/. A partir de ahí es una
    estrategia normal: aparece en Backtesting, Optimización, Walk-Forward,
    Monte Carlo y Sensibilidad como cualquier otra, sin código especial
    en esas páginas.
    """
    import re
    safe_stem = re.sub(r"[^a-zA-Z0-9_]", "_", filename_stem).strip("_").lower() or "mi_estrategia"
    if safe_stem in ("base", "__init__"):
        safe_stem = f"custom_{safe_stem}"
    path = _STRATEGIES_DIR / f"{safe_stem}.py"
    header = (
        f'"""\nstrategies/{safe_stem}.py\n'
        "Estrategia creada con el Constructor de Estrategias (editor de código).\n"
        '"""\n'
        "import pandas as pd\nimport numpy as np\n"
        "from strategies.base import BaseStrategy, Indicators\n\n\n"
    )
    path.write_text(header + code.strip() + "\n", encoding="utf-8")
    return path


def list_user_strategy_files() -> list:
    """Lista los archivos guardados desde el Constructor (heurística: no son los 5 originales)."""
    builtin = {"rsi_mean_reversion", "ema_trend_following", "donchian_breakout",
               "momentum", "range_reversion"}
    non_strategy = {"base", "code_compiler"}
    return sorted(
        p.stem for p in _STRATEGIES_DIR.glob("*.py")
        if p.stem not in builtin and p.stem not in non_strategy and not p.stem.startswith("_")
    )


def delete_user_strategy_file(filename_stem: str) -> bool:
    """
    Borra un archivo de estrategia guardado (Constructor o Descubridor
    Genético) del directorio strategies/ y refresca el registro en caliente.
    Los 5 archivos originales del sistema están protegidos y nunca se borran.
    """
    builtin = {"rsi_mean_reversion", "ema_trend_following", "donchian_breakout",
               "momentum", "range_reversion", "base", "code_compiler", "__init__"}
    stem = filename_stem.strip().lower()
    if stem in builtin:
        return False
    path = _STRATEGIES_DIR / f"{stem}.py"
    if not path.exists():
        return False
    path.unlink()
    _remove_discovery_manifest_entry(stem)
    refresh_registry()
    return True


# ---------------------------------------------------------------------------
# Manifiesto de estrategias descubiertas — metadatos persistentes en disco
# (mismo directorio strategies/, junto al .py real) para que el Descubridor
# Genético pueda mostrar, tras reiniciar el programa, qué estrategias ya
# están guardadas y listas para usarse en Backtesting/Optimización/Portfolio,
# sin depender de la memoria de sesión de Streamlit (que se pierde al
# reiniciar).
# ---------------------------------------------------------------------------
_MANIFEST_PATH = _STRATEGIES_DIR / "_discovery_manifest.json"


def _load_discovery_manifest() -> dict:
    import json
    if not _MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_discovery_manifest(data: dict) -> None:
    import json
    _MANIFEST_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def record_discovery_manifest(filename_stem: str, meta: dict) -> None:
    """Registra los metadatos (símbolo, temporalidad, familias, regla, fecha)
    de una estrategia enviada desde el Descubridor Genético, para poder
    listarla luego sin depender del estado de sesión."""
    data = _load_discovery_manifest()
    data[filename_stem.strip().lower()] = meta
    _save_discovery_manifest(data)


def _remove_discovery_manifest_entry(filename_stem: str) -> None:
    data = _load_discovery_manifest()
    data.pop(filename_stem.strip().lower(), None)
    _save_discovery_manifest(data)


def list_discovered_strategies() -> list[dict]:
    """
    Lista, directamente desde disco, todas las estrategias que el
    Descubridor Genético ha guardado alguna vez — persiste entre
    reinicios del programa porque lee el archivo .py (que sigue existiendo
    en strategies/) y el manifiesto de metadatos, no el estado de sesión.
    Cada entrada incluye si la clase sigue activa en STRATEGY_REGISTRY
    en este momento (por si el archivo se borró o falló al cargar).
    """
    manifest = _load_discovery_manifest()
    out = []
    for stem, meta in sorted(manifest.items()):
        path = _STRATEGIES_DIR / f"{stem}.py"
        class_name = meta.get("class_name", stem)
        out.append({
            "stem": stem,
            "file_exists": path.exists(),
            "active": class_name in STRATEGY_REGISTRY,
            **meta,
        })
    return out


def get_strategy(name: str, **params) -> BaseStrategy:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Estrategia '{name}' no encontrada. Disponibles: {sorted(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name](**params)


def list_strategies() -> list:
    return sorted(STRATEGY_REGISTRY.keys())


def strategy_catalog() -> list[dict]:
    catalog = []
    for name, cls in STRATEGY_REGISTRY.items():
        try:
            instance = cls()
            space = instance.get_param_space()
            catalog.append({
                "name": name, "description": getattr(cls, "description", "—"),
                "version": getattr(cls, "version", "—"), "n_params": len(space),
                "params": list(space.keys()),
            })
        except Exception:
            catalog.append({"name": name, "error": "No se pudo instanciar"})
    return catalog


def save_optimized_variant(
    strategy_name: str,
    best_params: dict,
    *,
    objective: str = "",
    asset: str = "",
    timeframe: str = "",
) -> str:
    """Save an optimized subclass without overwriting the original strategy."""
    import json as _json
    import re as _re

    original = STRATEGY_REGISTRY.get(strategy_name)
    if original is None:
        raise ValueError(f"No existe la estrategia original: {strategy_name}")
    module = getattr(original, "__module__", "")
    original_class = getattr(original, "__name__", "")
    if not module.startswith("strategies."):
        raise ValueError(f"La estrategia {strategy_name} no pertenece a strategies/")
    safe_base = _re.sub(r"[^A-Za-z0-9_]", "_", original_class).strip("_") or "Strategy"
    variant_class = f"{safe_base}Optimizada"
    variant_stem = _re.sub(r"[^A-Za-z0-9_]", "_", variant_class).strip("_").lower()
    if variant_stem in {"base", "__init__"}:
        variant_stem = "optimized_" + variant_stem

    params_json = _json.dumps(best_params or {}, ensure_ascii=False, indent=4, default=str)
    code = (
        '"""Variante optimizada automáticamente por CapitalQuant.\\n'
        f'Original: {original_class}\\nObjetivo: {objective}\\n'
        f'Activo: {asset}\\nTemporalidad: {timeframe}\\n"""\n'
        f"from {module} import {original_class}\n\n"
        f"class {variant_class}({original_class}):\n"
        f'    name = "{strategy_name} Optimizada"\n\n'
        "    def __init__(self, **kwargs):\n"
        f"        optimized = {params_json}\n"
        "        optimized.update(kwargs)\n"
        "        super().__init__(**optimized)\n"
    )
    path = _STRATEGIES_DIR / f"{variant_stem}.py"
    path.write_text(code, encoding="utf-8")
    manifest = {
        "class_name": variant_class,
        "strategy_name": f"{strategy_name} Optimizada",
        "original_strategy": strategy_name,
        "source": "Optimization",
        "optimized": True,
        "objective": objective,
        "asset": asset,
        "timeframe": timeframe,
        "best_params": best_params or {},
    }
    try:
        _remove_discovery_manifest_entry(variant_stem)
    except Exception:
        pass
    record_discovery_manifest(variant_stem, manifest)
    refresh_registry()
    return f"{strategy_name} Optimizada"
