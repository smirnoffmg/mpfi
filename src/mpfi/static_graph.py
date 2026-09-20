"""A call graph built from names: imports, scopes, classes and definitions.

No points-to analysis and no fixpoint. A call resolves when the name in front of
it traces to an import, a definition, the class the method belongs to, or a
variable that was handed a fresh instance in the same function. Anything else is
left unresolved on purpose: two candidates mean guessing.

Measured against a points-to analysis on the same repositories, this adds no
false edges; what it used to lose were the calls a method makes through `self`.
"""

import ast
from dataclasses import dataclass
from pathlib import Path

from mpfi.patterns import ml_edges

MAX_BASE_DEPTH = 10


@dataclass(frozen=True)
class _Names:
    """Everything a module lets a name mean."""

    imports: dict[str, str]
    defined: set[str]
    bases: dict[str, list[str]]
    attributes: dict[str, dict[str, str]]


def call_graph(package: Path) -> dict[str, set[str]]:
    """Resolve calls against the whole package, not one module at a time.

    A base class or an instantiated class usually lives in a sibling module, so
    definitions are collected from every module before any call is resolved.
    """
    modules = []
    for path in sorted(package.rglob("*.py")):
        try:
            source = path.read_text(errors="replace")
            tree = ast.parse(source)
        except (OSError, SyntaxError, ValueError):
            continue
        modules.append((_module_name(path, package), source, tree))

    imports = {module: _imports(tree, module) for module, _, tree in modules}
    defined: set[str] = set()
    for module, _, tree in modules:
        defined |= _definitions(tree, module)

    bases: dict[str, list[str]] = {}
    for module, _, tree in modules:
        lookup = _Names(
            imports=imports[module], defined=defined, bases={}, attributes={}
        )
        bases.update(_bases(tree, module, lookup))

    attributes: dict[str, dict[str, str]] = {}
    for module, _, tree in modules:
        known = _Names(
            imports=imports[module], defined=defined, bases=bases, attributes={}
        )
        attributes.update(_attributes(tree, module, known))

    graph: dict[str, set[str]] = {}
    for module, source, tree in modules:
        names = _Names(
            imports=imports[module],
            defined=defined,
            bases=bases,
            attributes=attributes,
        )
        for caller, callees in _graph_of(tree, source, module, names).items():
            graph.setdefault(caller, set()).update(callees)
    return graph


def call_graph_for_source(source: str, module: str) -> dict[str, set[str]]:
    """One module on its own; a package is resolved by call_graph."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    return _graph_of(tree, source, module, _names_of(tree, module))


def _graph_of(
    tree: ast.Module, source: str, module: str, names: _Names
) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for scope, node in _scopes(tree, module):
        instances = _local_instances(node, scope, names)
        callees = set()
        for call in _calls_directly_in(node):
            target = _resolve(call.func, scope, names, instances)
            if target:
                callees.add(target)
        if callees:
            graph.setdefault(scope, set()).update(callees)

    for caller, callees in ml_edges(source, module).items():
        graph.setdefault(caller, set()).update(callees)
    return graph


def _names_of(tree: ast.Module, module: str) -> _Names:
    imports = _imports(tree, module)
    defined = _definitions(tree, module)
    lookup = _Names(imports=imports, defined=defined, bases={}, attributes={})
    bases = _bases(tree, module, lookup)
    known = _Names(imports=imports, defined=defined, bases=bases, attributes={})
    return _Names(
        imports=imports,
        defined=defined,
        bases=bases,
        attributes=_attributes(tree, module, known),
    )


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


def _local_instances(node: ast.AST, scope: str, names: _Names) -> dict[str, str]:
    """Variables handed a fresh instance: `scaler = Scaler()`.

    A name assigned more than once, or assigned anything else, is dropped —
    picking one of two candidates would be a guess.
    """
    seen: dict[str, str | None] = {}
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign) or len(child.targets) != 1:
            continue
        target = child.targets[0]
        if not isinstance(target, ast.Name):
            continue
        cls = None
        if isinstance(child.value, ast.Call):
            resolved = _resolve_name(child.value.func, scope, names)
            if resolved and resolved in names.bases:
                cls = resolved
        seen[target.id] = None if target.id in seen else cls
    return {name: cls for name, cls in seen.items() if cls}


def _resolve(
    func: ast.expr, scope: str, names: _Names, instances: dict[str, str]
) -> str | None:
    if isinstance(func, ast.Name):
        return _lookup(func.id, scope, names)
    if not isinstance(func, ast.Attribute):
        return None

    receiver = func.value
    if (
        isinstance(receiver, ast.Attribute)
        and isinstance(receiver.value, ast.Name)
        and receiver.value.id == "self"
    ):
        owner = _enclosing_class(scope, names)
        held = names.attributes.get(owner or "", {}).get(receiver.attr)
        return _class_attribute(held, func.attr, names) if held else None
    if _is_super_call(receiver):
        owner = _enclosing_class(scope, names)
        return _base_attribute(owner, func.attr, names) if owner else None
    if isinstance(receiver, ast.Name):
        if receiver.id == "self":
            owner = _enclosing_class(scope, names)
            return _class_attribute(owner, func.attr, names) if owner else None
        if receiver.id in instances:
            return _class_attribute(instances[receiver.id], func.attr, names)
    base = _resolve_name(receiver, scope, names)
    return f"{base}.{func.attr}" if base else None


def _resolve_name(node: ast.expr, scope: str, names: _Names) -> str | None:
    if isinstance(node, ast.Name):
        return _lookup(node.id, scope, names)
    if isinstance(node, ast.Attribute):
        base = _resolve_name(node.value, scope, names)
        return f"{base}.{node.attr}" if base else None
    return None


def _attributes(
    tree: ast.Module, module: str, known: _Names
) -> dict[str, dict[str, str]]:
    """Attributes handed a fresh instance: `self.scaler_ = StandardScaler()`."""
    held: dict[str, dict[str, str]] = {}

    def walk(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = f"{scope}.{child.name}"
                if isinstance(child, ast.ClassDef):
                    held[name] = _self_assignments(child, name, known)
                walk(child, name)
            else:
                walk(child, scope)

    walk(tree, module)
    return held


def _self_assignments(node: ast.ClassDef, cls: str, known: _Names) -> dict[str, str]:
    found: dict[str, str | None] = {}
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign) or len(child.targets) != 1:
            continue
        target = child.targets[0]
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            continue
        held = None
        if isinstance(child.value, ast.Call):
            resolved = _resolve_name(child.value.func, cls, known)
            if resolved and resolved in known.bases:
                held = resolved
        found[target.attr] = None if target.attr in found else held
    return {name: cls_ for name, cls_ in found.items() if cls_}


def _is_super_call(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "super"
    )


def _base_attribute(cls: str, attr: str, names: _Names) -> str | None:
    """What `super().attr` means: the attribute as a base class defines it."""
    for base in names.bases.get(cls, []):
        if base in names.bases:
            found = _class_attribute(base, attr, names)
            if found:
                return found
        else:
            return f"{base}.{attr}"
    return None


def _enclosing_class(scope: str, names: _Names) -> str | None:
    owner = scope.rsplit(".", 1)[0]
    return owner if owner in names.bases else None


def _class_attribute(cls: str, attr: str, names: _Names, depth: int = 0) -> str | None:
    """The method as this class sees it, following base classes outward."""
    candidate = f"{cls}.{attr}"
    if candidate in names.defined:
        return candidate
    if depth >= MAX_BASE_DEPTH:
        return None
    for base in names.bases.get(cls, []):
        if base in names.bases:
            found = _class_attribute(base, attr, names, depth + 1)
            if found:
                return found
        elif base not in names.defined:
            # A class from outside the package: name the method on it and stop.
            return f"{base}.{attr}"
    return None


def _lookup(name: str, scope: str, names: _Names) -> str | None:
    """Innermost enclosing definition wins; then imports; then nothing."""
    parts = scope.split(".")
    while parts:
        candidate = ".".join([*parts, name])
        if candidate in names.defined:
            return candidate
        parts.pop()
    return names.imports.get(name)


def _imports(tree: ast.Module, module: str = "") -> dict[str, str]:
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            origin = _import_origin(node, module)
            if origin is None:
                continue
            for alias in node.names:
                full = f"{origin}.{alias.name}" if origin else alias.name
                names[alias.asname or alias.name] = full
    return names


def _import_origin(node: ast.ImportFrom, module: str) -> str | None:
    """`from .base import X` inside `a.b` comes from `a.base`."""
    if not node.level:
        return node.module
    parts = module.split(".") if module else []
    if node.level > len(parts):
        return None
    prefix = ".".join(parts[: len(parts) - node.level])
    if node.module:
        return f"{prefix}.{node.module}" if prefix else node.module
    return prefix


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


def _bases(tree: ast.Module, module: str, lookup: _Names) -> dict[str, list[str]]:
    """Every class in the module, with its base classes named in full."""
    classes: dict[str, list[str]] = {}

    def walk(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = f"{scope}.{child.name}"
                if isinstance(child, ast.ClassDef):
                    resolved = [
                        _resolve_name(base, scope, lookup) for base in child.bases
                    ]
                    classes[name] = [base for base in resolved if base]
                walk(child, name)
            else:
                walk(child, scope)

    walk(tree, module)
    return classes
