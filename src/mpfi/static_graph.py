"""A call graph built from names: imports, scopes, classes and definitions.

No points-to analysis and no fixpoint. A call resolves when the name in front of
it traces to an import, a definition, the class the method belongs to, or a
variable that was handed a fresh instance in the same function. Anything else is
left unresolved on purpose: two candidates mean guessing.

Measured against a points-to analysis on the same repositories, this adds no
false edges; what it used to lose were the calls a method makes through `self`.

Beside the calls, the graph carries hand-offs: an edge from `a` to `b` when a
value `a` returned is what `b` receives, as in `b(a(df))`, `df.pipe(a).pipe(b)`
or `df = a(df)` followed by `b(df)`. A chain of transformations is written in any
of these styles, and nested calls are only one of them.
"""

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from mpfi.patterns import PIPE_METHODS, ml_edges

MAX_BASE_DEPTH = 10

# Where a table goes through an estimator, in the order a pipeline would call it.
STEP_METHODS = ("transform", "fit_transform", "fit")

# A test exercises the pipeline without being part of it, so it says nothing
# about how the pipeline itself is wired.
TEST_DIRECTORIES = frozenset({"test", "tests", "testing"})


def is_test_path(path: Path) -> bool:
    name = path.name
    return (
        any(part in TEST_DIRECTORIES for part in path.parts)
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def source_files(package: Path) -> list[Path]:
    return [p for p in sorted(package.rglob("*.py")) if not is_test_path(p)]


@dataclass(frozen=True)
class _Names:
    """Everything a module lets a name mean."""

    imports: dict[str, str]
    defined: set[str]
    bases: dict[str, list[str]]
    attributes: dict[str, dict[str, set[str]]]
    returns: dict[str, set[str]]


def call_graph(package: Path) -> dict[str, set[str]]:
    return pipeline_graph(package)[0]


def pipeline_graph(package: Path) -> tuple[dict[str, set[str]], set[str]]:
    """The graph, and the package's functions that ML patterns name as steps.

    Calls resolve against the whole package, not one module at a time.

    A base class or an instantiated class usually lives in a sibling module, so
    definitions are collected from every module before any call is resolved.
    """
    modules = []
    for path in source_files(package):
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
            imports=imports[module],
            defined=defined,
            bases={},
            attributes={},
            returns={},
        )
        bases.update(_bases(tree, module, lookup))

    attributes: dict[str, dict[str, set[str]]] = {}
    for module, _, tree in modules:
        known = _Names(
            imports=imports[module],
            defined=defined,
            bases=bases,
            attributes={},
            returns={},
        )
        attributes.update(_attributes(tree, module, known))

    returns: dict[str, set[str]] = {}
    for module, _, tree in modules:
        known = _Names(
            imports=imports[module],
            defined=defined,
            bases=bases,
            attributes={},
            returns={},
        )
        returns.update(_returns(tree, module, known))

    graph: dict[str, set[str]] = {}
    steps: set[str] = set()
    for module, _, tree in modules:
        names = _Names(
            imports=imports[module],
            defined=defined,
            bases=bases,
            attributes=attributes,
            returns=returns,
        )
        edges, found = _graph_of(tree, module, names)
        for caller, callees in edges.items():
            graph.setdefault(caller, set()).update(callees)
        steps |= found
    return graph, steps


def call_graph_for_source(source: str, module: str) -> dict[str, set[str]]:
    """One module on its own; a package is resolved by call_graph."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    return _graph_of(tree, module, _names_of(tree, module))[0]


def _graph_of(
    tree: ast.Module, module: str, names: _Names
) -> tuple[dict[str, set[str]], set[str]]:
    graph: dict[str, set[str]] = {}
    for scope, node in _scopes(tree, module):
        instances = _local_instances(node, scope, names)
        callees = set()
        for call in _calls_directly_in(node):
            callees |= _resolve(call.func, scope, names, instances)
        if callees:
            graph.setdefault(scope, set()).update(callees)
        for producer, consumer in _hand_offs(node, scope, names, instances):
            graph.setdefault(producer, set()).add(consumer)

    steps: set[str] = set()
    patterns = ml_edges(tree, module, lambda name: _step_node(name, names))
    for caller, callees in patterns.items():
        graph.setdefault(caller, set()).update(callees)
        steps |= callees & names.defined
    return graph, steps


def _step_node(name: str, names: _Names) -> str | None:
    """The package's own function a pipeline step runs the table through."""
    if name in names.bases:
        for method in STEP_METHODS:
            found = _class_attribute(name, method, names)
            if found in names.defined:
                return found
        return None
    return name if name in names.defined else None


def _hand_offs(
    node: ast.AST, scope: str, names: _Names, instances: dict[str, set[str]]
) -> set[tuple[str, str]]:
    """Which function's result each function of the package is handed.

    Statements are read in source order and a variable remembers what produced
    its value last; branches and loops are not told apart. A call into a library
    or an unresolved method passes on what it was given, so `a(df).fillna(0)`
    still carries `a` to whatever comes next.
    """
    edges: set[tuple[str, str]] = set()
    held: dict[str, set[str]] = {}

    def steps(func: ast.expr) -> set[str]:
        resolved = _resolve(func, scope, names, instances)
        return {c for c in resolved if c in names.defined and c not in names.bases}

    def produced(expr: ast.AST) -> set[str]:
        if isinstance(expr, ast.Name):
            return set(held.get(expr.id, ()))
        if not isinstance(expr, ast.Call):
            return _union(produced(child) for child in ast.iter_child_nodes(expr))
        func = expr.func
        if isinstance(func, ast.Attribute) and func.attr in PIPE_METHODS and expr.args:
            targets = steps(expr.args[0])
            inputs = produced(func.value) | _union(
                produced(child) for child in [*expr.args[1:], *expr.keywords]
            )
        else:
            targets = steps(func)
            inputs = _union(produced(child) for child in ast.iter_child_nodes(expr))
        if not targets:
            return inputs
        edges.update((p, t) for p in inputs for t in targets if p != t)
        return targets

    def bind(target: ast.expr, value: set[str]) -> None:
        if isinstance(target, ast.Name):
            held[target.id] = value
        elif isinstance(target, ast.Tuple | ast.List):
            for element in target.elts:
                bind(element, value)
        elif isinstance(target, ast.Subscript):
            base = target.value
            while isinstance(base, ast.Subscript | ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                held[base.id] = held.get(base.id, set()) | value

    def visit(current: ast.AST) -> None:
        if isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            return
        if isinstance(current, ast.Assign):
            value = produced(current.value)
            for target in current.targets:
                bind(target, value)
        elif isinstance(current, ast.AnnAssign | ast.AugAssign):
            if current.value is not None:
                extra = (
                    produced(current.target)
                    if isinstance(current, ast.AugAssign)
                    else set()
                )
                bind(current.target, produced(current.value) | extra)
        elif isinstance(current, ast.For | ast.AsyncFor):
            bind(current.target, produced(current.iter))
            for statement in [*current.body, *current.orelse]:
                visit(statement)
        elif isinstance(current, ast.expr):
            produced(current)
        else:
            for child in ast.iter_child_nodes(current):
                visit(child)

    for child in ast.iter_child_nodes(node):
        visit(child)
    return edges


def _union(sets: Iterable[set[str]]) -> set[str]:
    found: set[str] = set()
    for part in sets:
        found |= part
    return found


def _names_of(tree: ast.Module, module: str) -> _Names:
    imports = _imports(tree, module)
    defined = _definitions(tree, module)
    lookup = _Names(
        imports=imports, defined=defined, bases={}, attributes={}, returns={}
    )
    bases = _bases(tree, module, lookup)
    known = _Names(
        imports=imports, defined=defined, bases=bases, attributes={}, returns={}
    )
    return _Names(
        imports=imports,
        defined=defined,
        bases=bases,
        attributes=_attributes(tree, module, known),
        returns=_returns(tree, module, known),
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


def _local_instances(node: ast.AST, scope: str, names: _Names) -> dict[str, set[str]]:
    """Variables handed an instance: `scaler = Scaler()`, `enc = get_encoder(n)`.

    A name assigned twice holds either, and a factory holds any class it can
    return; the graph keeps every candidate rather than lose an edge.
    """
    held: dict[str, set[str]] = {}
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign) or len(child.targets) != 1:
            continue
        target = child.targets[0]
        if isinstance(target, ast.Name):
            held.setdefault(target.id, set()).update(
                _candidates(child.value, scope, names)
            )
    return {name: classes for name, classes in held.items() if classes}


def _candidates(value: ast.expr, scope: str, names: _Names) -> set[str]:
    """Which classes an expression can hold: a constructor, or a factory's returns."""
    if not isinstance(value, ast.Call):
        return set()
    resolved = _resolve_name(value.func, scope, names)
    if not resolved:
        return set()
    if resolved in names.bases:
        return {resolved}
    return set(names.returns.get(resolved, set()))


def _resolve(
    func: ast.expr, scope: str, names: _Names, instances: dict[str, set[str]]
) -> set[str]:
    if isinstance(func, ast.Name):
        found = _lookup(func.id, scope, names)
        return {found} if found else set()
    if not isinstance(func, ast.Attribute):
        return set()

    receiver = func.value
    if (
        isinstance(receiver, ast.Attribute)
        and isinstance(receiver.value, ast.Name)
        and receiver.value.id == "self"
    ):
        owner = _enclosing_class(scope, names)
        held = names.attributes.get(owner or "", {}).get(receiver.attr, set())
        return _methods_of(held, func.attr, names)
    if _is_super_call(receiver):
        owner = _enclosing_class(scope, names)
        found = _base_attribute(owner, func.attr, names) if owner else None
        return {found} if found else set()
    if isinstance(receiver, ast.Name):
        if receiver.id == "self":
            owner = _enclosing_class(scope, names)
            found = _class_attribute(owner, func.attr, names) if owner else None
            return {found} if found else set()
        if receiver.id in instances:
            return _methods_of(instances[receiver.id], func.attr, names)
    base = _resolve_name(receiver, scope, names)
    return {f"{base}.{func.attr}"} if base else set()


def _methods_of(classes: set[str], attr: str, names: _Names) -> set[str]:
    found = {_class_attribute(cls, attr, names) for cls in classes}
    return {name for name in found if name}


def _resolve_name(node: ast.expr, scope: str, names: _Names) -> str | None:
    if isinstance(node, ast.Name):
        return _lookup(node.id, scope, names)
    if isinstance(node, ast.Attribute):
        base = _resolve_name(node.value, scope, names)
        return f"{base}.{node.attr}" if base else None
    return None


def _attributes(
    tree: ast.Module, module: str, known: _Names
) -> dict[str, dict[str, set[str]]]:
    """Attributes handed an instance: `self.scaler_ = StandardScaler()`."""
    held: dict[str, dict[str, set[str]]] = {}

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


def _self_assignments(
    node: ast.ClassDef, cls: str, known: _Names
) -> dict[str, set[str]]:
    held: dict[str, set[str]] = {}
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign) or len(child.targets) != 1:
            continue
        target = child.targets[0]
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            held.setdefault(target.attr, set()).update(
                _candidates(child.value, cls, known)
            )
    return {name: classes for name, classes in held.items() if classes}


def _returns(tree: ast.Module, module: str, known: _Names) -> dict[str, set[str]]:
    """Which classes a function hands back, for one level of indirection.

    A factory returning another factory's result is not followed: the classes
    of the inner one are not known while this map is being built.
    """
    found: dict[str, set[str]] = {}

    def walk(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = f"{scope}.{child.name}"
                if not isinstance(child, ast.ClassDef):
                    # A factory usually picks a class into a local and returns
                    # the local, so returned names are looked up there.
                    locals_ = _local_instances(child, name, known)
                    classes: set[str] = set()
                    for inner in ast.walk(child):
                        if not isinstance(inner, ast.Return) or inner.value is None:
                            continue
                        if isinstance(inner.value, ast.Name):
                            classes |= locals_.get(inner.value.id, set())
                        else:
                            classes |= _candidates(inner.value, name, known)
                    if classes:
                        found[name] = classes
                walk(child, name)
            else:
                walk(child, scope)

    walk(tree, module)
    return found


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
