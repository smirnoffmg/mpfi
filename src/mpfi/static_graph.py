"""A call graph built from names: imports, scopes and definitions.

No points-to analysis. A call is resolved when the name in front of it can be
traced to an import or a definition, which covers how feature-engineering code
is normally written, and fails on aliasing — `f = transform; f(df)` stays
unresolved, deliberately and visibly.

What this buys over a points-to analysis is that the answer is the same every
run and arrives in one pass over the syntax.
"""

import ast
from pathlib import Path

from mpfi.patterns import ml_edges


def call_graph(package: Path) -> dict[str, set[str]]:
    """Merge the graphs of every module in a package."""
    graph: dict[str, set[str]] = {}
    for path in sorted(package.rglob("*.py")):
        try:
            source = path.read_text(errors="replace")
        except OSError:
            continue
        module = _module_name(path, package)
        for caller, callees in call_graph_for_source(source, module).items():
            graph.setdefault(caller, set()).update(callees)
    return graph


def call_graph_for_source(source: str, module: str) -> dict[str, set[str]]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}

    graph: dict[str, set[str]] = {}
    for scope, node in _scopes(tree, module):
        callees = set()
        for call in _calls_directly_in(node):
            target = _resolve(call.func, scope, tree, module)
            if target:
                callees.add(target)
        if callees:
            graph.setdefault(scope, set()).update(callees)

    for caller, callees in ml_edges(source, module).items():
        graph.setdefault(caller, set()).update(callees)
    return graph


def _module_name(path: Path, package: Path) -> str:
    parts = path.relative_to(package).with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _scopes(tree: ast.Module, module: str) -> list[tuple[str, ast.AST]]:
    """Every function, named by the chain of definitions that encloses it."""
    found: list[tuple[str, ast.AST]] = [(module, tree)]

    def walk(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = f"{scope}.{child.name}"
                if not isinstance(child, ast.ClassDef):
                    found.append((name, child))
                walk(child, name)
            else:
                walk(child, scope)

    walk(tree, module)
    return found


def _calls_directly_in(node: ast.AST) -> list[ast.Call]:
    """Calls belonging to this scope, not to a function nested inside it."""
    found: list[ast.Call] = []

    def walk(current: ast.AST) -> None:
        for child in ast.iter_child_nodes(current):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            if isinstance(child, ast.Call):
                found.append(child)
            walk(child)

    walk(node)
    return found


def _resolve(func: ast.expr, scope: str, tree: ast.Module, module: str) -> str | None:
    if isinstance(func, ast.Name):
        return _lookup(func.id, scope, tree, module)
    if isinstance(func, ast.Attribute):
        base = _resolve_value(func.value, scope, tree, module)
        return f"{base}.{func.attr}" if base else None
    return None


def _resolve_value(
    node: ast.expr, scope: str, tree: ast.Module, module: str
) -> str | None:
    if isinstance(node, ast.Name):
        return _lookup(node.id, scope, tree, module)
    if isinstance(node, ast.Attribute):
        base = _resolve_value(node.value, scope, tree, module)
        return f"{base}.{node.attr}" if base else None
    return None


def _lookup(name: str, scope: str, tree: ast.Module, module: str) -> str | None:
    """Innermost enclosing definition wins; then imports; then nothing."""
    imports = _imports(tree)
    defined = _definitions(tree, module)

    parts = scope.split(".")
    while parts:
        candidate = ".".join([*parts, name])
        if candidate in defined:
            return candidate
        parts.pop()
    if name in imports:
        return imports[name]
    return None


def _imports(tree: ast.Module) -> dict[str, str]:
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return names


def _definitions(tree: ast.Module, module: str) -> set[str]:
    found = set()

    def walk(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = f"{scope}.{child.name}"
                found.add(name)
                walk(child, name)
            else:
                walk(child, scope)

    walk(tree, module)
    return found
