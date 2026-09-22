"""
strategies/ast_sandbox.py
Validación de seguridad del Constructor de Estrategias basada en el árbol
de sintaxis (AST), no en texto/regex.

Motivación (Auditoría 2.6): un filtro de solo-regex sobre el texto fuente
es evadible por construcción — no entiende sintaxis, así que cualquier
forma de escribir lo mismo que el regex no anticipó (unicode escapes en
nombres, `getattr` encadenado, `__class__.__mro__`, comprensiones que
acceden a atributos dunder, etc.) puede colarse. Este módulo camina el
AST ya parseado por Python (que normaliza toda esa ofuscación de texto)
y rechaza nodos estructuralmente peligrosos, en vez de patrones de texto.

Esto es DEFENSA EN PROFUNDIDAD, no un sandbox de aislamiento real de
proceso — sigue siendo aceptable solo para un único usuario en su propia
máquina local (ver Auditoría 2.6). Si el roadmap avanza hacia
multiusuario/API, esto debe reemplazarse por aislamiento real de proceso
(subprocess con recursos limitados, contenedor efímero, o un intérprete
restringido auditado como RestrictedPython) — ítem explícito de la Fase 4
del roadmap.
"""
from __future__ import annotations

import ast
from typing import Optional

# Nombres dunder que, si aparecen como atributo en CUALQUIER posición del
# árbol, permiten escapar del sandbox reducido de __builtins__ (acceder a
# clases base, subclases, globals de función, etc. — la cadena clásica
# `().__class__.__mro__[1].__subclasses__()` para llegar a `os` sin
# jamás escribir la palabra "import" ni "os." literalmente).
_FORBIDDEN_ATTR_NAMES = {
    "__globals__", "__subclasses__", "__bases__", "__mro__", "__base__",
    "__builtins__", "__loader__", "__spec__", "__code__", "__closure__",
    "__getattribute__", "__reduce__", "__reduce_ex__", "__init_subclass__",
    "__import__",
}

# Nombres que, si se referencian directamente como identificador, se
# consideran peligrosos independientemente de si `__builtins__` los
# expone o no (defensa en profundidad si alguna versión futura del
# namespace seguro los agregara por error).
_FORBIDDEN_NAMES = {
    "exec", "eval", "compile", "open", "input", "__import__",
    "vars", "globals", "locals", "memoryview", "breakpoint",
}


class ForbiddenConstructError(Exception):
    """Se lanza cuando el AST contiene una construcción no permitida."""
    def __init__(self, message: str, lineno: Optional[int] = None):
        self.lineno = lineno
        super().__init__(message)


class _SandboxVisitor(ast.NodeVisitor):
    def __init__(self):
        self.errors: list[str] = []

    def _flag(self, node: ast.AST, msg: str):
        lineno = getattr(node, "lineno", "?")
        self.errors.append(f"Línea {lineno}: {msg}")

    def visit_Import(self, node):
        self._flag(node, "no se permite 'import' dentro del editor de estrategias.")

    def visit_ImportFrom(self, node):
        self._flag(node, "no se permite 'from ... import ...' dentro del editor de estrategias.")

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in _FORBIDDEN_ATTR_NAMES:
            self._flag(
                node,
                f"acceso a atributo restringido '{node.attr}' — este patrón se usa "
                "típicamente para escapar del sandbox accediendo a clases internas "
                "de Python (ej. __class__.__mro__)."
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id in _FORBIDDEN_NAMES:
            self._flag(node, f"uso de '{node.id}' no permitido en el editor de estrategias.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # getattr(obj, "__globals__") / setattr(...) con nombres dunder como
        # literal de string es la variante más común de evasión de un
        # filtro de texto ingenuo — aquí se detecta aunque el nombre venga
        # como argumento de string en vez de como atributo directo.
        if isinstance(node.func, ast.Name) and node.func.id in ("getattr", "setattr", "delattr"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value in _FORBIDDEN_ATTR_NAMES or arg.value.startswith("__"):
                        self._flag(
                            node,
                            f"'{node.func.id}' con nombre de atributo dunder ('{arg.value}') "
                            "no está permitido — es un patrón típico de evasión de sandbox."
                        )
        self.generic_visit(node)


def validate_ast_safety(source_code: str) -> None:
    """
    Parsea `source_code` y recorre el AST buscando construcciones
    prohibidas. Lanza `ForbiddenConstructError` con todos los problemas
    encontrados si hay al menos uno; no lanza nada si el código es seguro
    según estas reglas (complementa, no reemplaza, la validación por
    regex existente en code_compiler.py).
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        # El error de sintaxis ya se reporta por separado en compile_strategy_code;
        # aquí simplemente no hay nada más que validar.
        return

    visitor = _SandboxVisitor()
    visitor.visit(tree)
    if visitor.errors:
        raise ForbiddenConstructError(
            "Por seguridad, el editor rechazó este código:\n" + "\n".join(visitor.errors)
        )
