"""
strategies/code_compiler.py
Compilador de estrategias escritas por el usuario en el Constructor de
Estrategias (editor de código dentro de la app).

Contrato: el usuario escribe una clase Python que hereda de BaseStrategy,
exactamente el mismo contrato que ya usan las estrategias precargadas
(strategies/rsi_mean_reversion.py, etc.). Esto es intencional: así una
estrategia creada en el editor queda 100% compatible con el motor de
backtesting, el optimizador, walk-forward, Monte Carlo y sensibilidad
sin ningún adaptador adicional — es el mismo objeto BaseStrategy de
siempre, solo que se escribió y compiló desde la UI en vez de a mano
en un archivo.

Sandbox: se ejecuta el código con un __builtins__ reducido (sin
import/open/eval/exec/os/sys/subprocess) para evitar que una estrategia
pegada desde cualquier lado pueda tocar el sistema de archivos o la red.
pandas, numpy e Indicators ya están inyectados en el namespace, así que
no hace falta `import` para el 99% de los casos de uso.
"""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from typing import Optional, Type

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, Indicators
from core.validators import validate_ohlcv
from strategies.ast_sandbox import validate_ast_safety, ForbiddenConstructError

# Builtins seguros — sin __import__, open, eval, exec, compile, input, etc.
_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
    "isinstance", "issubclass", "len", "list", "max", "min", "print",
    "property", "range", "round", "set", "sorted", "str", "sum", "tuple",
    "zip", "super", "staticmethod", "classmethod", "True", "False", "None",
    "type", "getattr", "setattr", "hasattr", "Exception", "ValueError",
    "TypeError", "KeyError", "IndexError", "object", "__build_class__",
    "__name__",
]
import builtins as _builtins
SAFE_BUILTINS = {name: getattr(_builtins, name) for name in _SAFE_BUILTIN_NAMES if hasattr(_builtins, name)}

_FORBIDDEN_PATTERNS = [
    (r"\bimport\b", "import"),
    (r"__import__", "__import__"),
    (r"\bopen\s*\(", "open("),
    (r"\beval\s*\(", "eval("),
    (r"\bexec\s*\(", "exec("),
    (r"\bcompile\s*\(", "compile("),
    (r"\bos\.", "os."),
    (r"\bsys\.", "sys."),
    (r"\bsubprocess\b", "subprocess"),
    (r"\bsocket\b", "socket"),
    (r"\bshutil\b", "shutil"),
    (r"\bpathlib\b", "pathlib"),
    (r"__builtins__", "__builtins__"),
    (r"__globals__", "__globals__"),
    (r"__class__\.__bases__", "__class__.__bases__"),
    (r"\binput\s*\(", "input("),
]


@dataclass
class CompileResult:
    ok: bool
    strategy_class: Optional[Type[BaseStrategy]] = None
    error: str = ""
    class_name: str = ""


def _static_safety_check(code: str) -> Optional[str]:
    """
    Chequeo rápido por patrones peligrosos antes de ejecutar nada.
    Usa límites de palabra (\\b) para no confundir texto natural en
    español (ej: 'cortos.', 'largos.', que contienen 'os.' como
    substring) con un acceso real a os./sys./etc.
    """
    for pattern, label in _FORBIDDEN_PATTERNS:
        if re.search(pattern, code):
            return (
                f"Por seguridad, el editor no permite usar '{label}'. "
                "pandas (pd), numpy (np) e Indicators ya están disponibles — "
                "no deberías necesitar imports para una estrategia."
            )
    return None


def compile_strategy_code(code: str) -> CompileResult:
    """
    Compila el código del usuario y devuelve la clase BaseStrategy resultante.
    No la instancia todavía — eso lo hace validate_strategy_class.
    """
    safety_err = _static_safety_check(code)
    if safety_err:
        return CompileResult(ok=False, error=safety_err)

    # Segunda capa de defensa, a nivel de AST: detecta patrones de evasión
    # estructurales (getattr encadenado, __class__.__mro__, etc.) que el
    # regex de arriba —al operar sobre texto plano— no puede ver. Ver
    # strategies/ast_sandbox.py y Auditoría sección 2.6.
    try:
        validate_ast_safety(textwrap.dedent(code))
    except ForbiddenConstructError as e:
        return CompileResult(ok=False, error=str(e))

    namespace = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "BaseStrategy": BaseStrategy,
        "Indicators": Indicators,
    }

    try:
        compiled = compile(textwrap.dedent(code), "<constructor_estrategias>", "exec")
        exec(compiled, namespace)
    except SyntaxError as e:
        return CompileResult(ok=False, error=f"Error de sintaxis en la línea {e.lineno}: {e.msg}")
    except Exception as e:
        return CompileResult(ok=False, error=f"Error al ejecutar el código: {e}")

    candidates = [
        obj for obj in namespace.values()
        if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy
    ]
    if not candidates:
        return CompileResult(
            ok=False,
            error="No se encontró ninguna clase que herede de BaseStrategy. "
                  "Tu código debe definir: class MiEstrategia(BaseStrategy): ...",
        )
    if len(candidates) > 1:
        return CompileResult(
            ok=False,
            error=f"Se encontraron {len(candidates)} clases de estrategia en el mismo bloque "
                  f"({', '.join(c.__name__ for c in candidates)}). Define solo una por editor.",
        )

    strategy_class = candidates[0]
    return CompileResult(ok=True, strategy_class=strategy_class, class_name=strategy_class.__name__)


def validate_strategy_class(strategy_class: Type[BaseStrategy], sample_df: pd.DataFrame) -> CompileResult:
    """
    Instancia la estrategia con parámetros por defecto y corre generate_signals
    sobre datos de muestra, verificando que respete el contrato del motor.
    """
    try:
        instance = strategy_class()
    except Exception as e:
        return CompileResult(ok=False, error=f"No se pudo instanciar la estrategia con parámetros por defecto: {e}")

    if not getattr(strategy_class, "name", None) or strategy_class.name == "Base Strategy":
        return CompileResult(ok=False, error="Define un atributo `name` distinto para tu estrategia (ej: name = \"Mi Estrategia\").")

    try:
        out = instance.generate_signals(sample_df.copy())
    except Exception as e:
        return CompileResult(ok=False, error=f"generate_signals() lanzó un error: {e}")

    if not isinstance(out, pd.DataFrame):
        return CompileResult(ok=False, error="generate_signals() debe devolver un DataFrame.")
    if "signal" not in out.columns:
        return CompileResult(ok=False, error="El DataFrame devuelto debe tener una columna 'signal' (1=largo, -1=corto, 0=fuera).")

    bad_values = set(out["signal"].dropna().unique()) - {-1, 0, 1}
    if bad_values:
        return CompileResult(ok=False, error=f"La columna 'signal' solo admite -1, 0 o 1. Se encontraron valores: {bad_values}")

    try:
        space = instance.get_param_space()
        if not isinstance(space, dict):
            return CompileResult(ok=False, error="get_param_space() debe devolver un dict.")
    except Exception as e:
        return CompileResult(ok=False, error=f"get_param_space() lanzó un error: {e}")

    return CompileResult(ok=True, strategy_class=strategy_class, class_name=strategy_class.__name__)


def suggest_class_name(display_name: str) -> str:
    """Convierte 'Mi Estrategia Rara' -> 'MiEstrategiaRara' para nombres de archivo/clase."""
    words = re.findall(r"[A-Za-z0-9]+", display_name)
    return "".join(w.capitalize() for w in words) or "MiEstrategia"
