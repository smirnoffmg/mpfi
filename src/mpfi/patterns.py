"""Call edges that tabular ML code declares as data instead of writing as calls.

Three shapes cover most of what a static call graph loses on such code:

  * a transform handed to ``DataFrame.pipe`` is called by the enclosing function,
    yet nothing in the source says so;
  * the steps of an sklearn pipeline live in a list of tuples and are dispatched
    inside ``fit``, each handing its output to the next;
  * a boosting library receives evaluation functions and callbacks as arguments.

Each shape is recovered here as the edge the interpreter would follow at runtime.
"""

import ast
from collections.abc import Callable
from dataclasses import dataclass

PIPE_METHODS = frozenset({"pipe"})

SEQUENTIAL_PIPELINES = frozenset(
    {"sklearn.pipeline.Pipeline", "sklearn.pipeline.make_pipeline"}
)
PARALLEL_PIPELINES = frozenset(
    {
        "sklearn.compose.ColumnTransformer",
        "sklearn.compose.make_column_transformer",
        "sklearn.pipeline.FeatureUnion",
        "sklearn.pipeline.make_union",
    }
)
# These take the steps as plain arguments; the classes take one list of them.
SPREAD_PIPELINES = frozenset(
    {
        "sklearn.pipeline.make_pipeline",
        "sklearn.pipeline.make_union",
        "sklearn.compose.make_column_transformer",
    }
)
FUNCTION_TRANSFORMERS = frozenset({"sklearn.preprocessing.FunctionTransformer"})

BOOSTING_TRAINERS = frozenset(
    {"lightgbm.train", "xgboost.train", "catboost.train", "lightgbm.cv", "xgboost.cv"}
)

STEPS_KEYWORDS = frozenset({"steps", "transformers", "transformer_list"})

# Given the qualified name of a class or function used as a pipeline step, the
# function of the analysed package that the table passes through, if any.
StepNode = Callable[[str], str | None]


@dataclass(frozen=True)
class _Stage:
    """Where a table enters a piece of pipeline, where it leaves, and between."""

    entries: frozenset[str] = frozenset()
    exits: frozenset[str] = frozenset()
    edges: frozenset[tuple[str, str]] = frozenset()

    def nodes(self) -> set[str]:
        found = set(self.entries | self.exits)
        for caller, callee in self.edges:
            found |= {caller, callee}
        return found


def ml_edges(tree: ast.Module, module: str, step_node: StepNode) -> dict[str, set[str]]:
    """Return caller -> callees for the ML patterns found in one module."""
    names = _resolve_names(tree, module)
    edges: dict[str, set[str]] = {}
    for scope, node in _calls(tree, module):
        for caller, callee in _edges_of_call(node, scope, names, step_node):
            edges.setdefault(caller, set()).add(callee)
    return edges


def _resolve_names(tree: ast.AST, module: str) -> dict[str, str]:
    """Map every name usable in the module to its fully qualified form."""
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.setdefault(node.name, f"{module}.{node.name}")
    return names


def _calls(tree: ast.AST, module: str) -> list[tuple[str, ast.Call]]:
    """Every call in the module, paired with the qualified name of its scope."""
    found: list[tuple[str, ast.Call]] = []

    def walk(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                walk(child, f"{scope}.{child.name}")
                continue
            if isinstance(child, ast.Call):
                found.append((scope, child))
            walk(child, scope)

    walk(tree, module)
    return found


def _edges_of_call(
    node: ast.Call, scope: str, names: dict[str, str], step_node: StepNode
) -> list[tuple[str, str]]:
    target = _qualify(node.func, names)

    if isinstance(node.func, ast.Attribute) and node.func.attr in PIPE_METHODS:
        return [(scope, callee) for callee in _callables(node.args[:1], names)]

    if target in SEQUENTIAL_PIPELINES | PARALLEL_PIPELINES:
        stage = _stage(node, names, step_node)
        # fit dispatches every step; each step then feeds the one after it.
        fit = [(f"{target}.fit", step) for step in sorted(stage.nodes())]
        return fit + sorted(stage.edges)

    if target in BOOSTING_TRAINERS:
        passed = [kw.value for kw in node.keywords] + node.args
        return [(target, callee) for callee in _callables(passed, names)]

    return []


def _stage(element: ast.expr, names: dict[str, str], step_node: StepNode) -> _Stage:
    """One step: a nested pipeline, a function wrapped as a transformer, or a class.

    A step from a library resolves to nothing and is left out of the chain: the
    steps on either side of it still hand the table to each other.
    """
    if isinstance(element, ast.Tuple):
        # (name, step), (name, step, columns) or (step, columns)
        element = next(
            (e for e in element.elts if not isinstance(e, ast.Constant)), element
        )
    if not isinstance(element, ast.Call):
        return _Stage()
    target = _qualify(element.func, names)
    if target in SEQUENTIAL_PIPELINES:
        return _in_sequence(
            [_stage(e, names, step_node) for e in _step_elements(element, target)]
        )
    if target in PARALLEL_PIPELINES:
        return _in_parallel(
            [_stage(e, names, step_node) for e in _step_elements(element, target)]
        )
    if target in FUNCTION_TRANSFORMERS:
        passed = element.args[:1] + [
            kw.value for kw in element.keywords if kw.arg == "func"
        ]
        target = next(iter(_callables(passed, names)), None)
    node = step_node(target) if target else None
    if node is None:
        return _Stage()
    return _Stage(frozenset({node}), frozenset({node}))


def _in_sequence(stages: list[_Stage]) -> _Stage:
    entries: frozenset[str] = frozenset()
    exits: frozenset[str] = frozenset()
    edges: set[tuple[str, str]] = set()
    for stage in stages:
        if not stage.entries:
            continue
        edges |= stage.edges
        edges |= {(done, nxt) for done in exits for nxt in stage.entries}
        entries = entries or stage.entries
        exits = stage.exits
    return _Stage(entries, exits, frozenset(edges))


def _in_parallel(stages: list[_Stage]) -> _Stage:
    return _Stage(
        frozenset().union(*(s.entries for s in stages)),
        frozenset().union(*(s.exits for s in stages)),
        frozenset().union(*(s.edges for s in stages)),
    )


def _step_elements(node: ast.Call, target: str) -> list[ast.expr]:
    if target in SPREAD_PIPELINES:
        return list(node.args)
    for keyword in node.keywords:
        if keyword.arg in STEPS_KEYWORDS:
            return _unpack(keyword.value)
    return _unpack(node.args[0]) if node.args else []


def _unpack(value: ast.expr) -> list[ast.expr]:
    if isinstance(value, ast.List | ast.Tuple):
        return list(value.elts)
    return []


def _callables(values: list[ast.expr], names: dict[str, str]) -> list[str]:
    """Functions handed to another function, directly or inside a list."""
    resolved = []
    for value in values:
        for candidate in _unpack(value) or [value]:
            if isinstance(candidate, ast.Name | ast.Attribute):
                qualified = _qualify(candidate, names)
                if qualified:
                    resolved.append(qualified)
    return resolved


def _qualify(node: ast.expr, names: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _qualify(node.value, names)
        return f"{base}.{node.attr}" if base else None
    return None
