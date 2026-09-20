"""Call edges that tabular ML code declares as data instead of writing as calls.

Three shapes cover most of what a static call graph loses on such code:

  * a transform handed to ``DataFrame.pipe`` is called by the enclosing function,
    yet nothing in the source says so;
  * the steps of an sklearn pipeline live in a list of tuples and are dispatched
    inside ``fit``;
  * a boosting library receives evaluation functions and callbacks as arguments.

Each shape is recovered here as the edge the interpreter would follow at runtime.
"""

import ast

PIPE_METHODS = frozenset({"pipe"})

# name of the constructor -> whether the last step is an estimator (fit) or
# another transform (fit_transform), which is what Pipeline.fit distinguishes.
SEQUENTIAL_PIPELINES = frozenset(
    {"sklearn.pipeline.Pipeline", "sklearn.pipeline.make_pipeline"}
)
PARALLEL_PIPELINES = frozenset(
    {"sklearn.compose.ColumnTransformer", "sklearn.pipeline.FeatureUnion"}
)

BOOSTING_TRAINERS = frozenset(
    {"lightgbm.train", "xgboost.train", "catboost.train", "lightgbm.cv", "xgboost.cv"}
)

STEPS_KEYWORDS = frozenset({"steps", "transformers", "transformer_list"})


def ml_edges(source: str, module: str) -> dict[str, set[str]]:
    """Return caller -> callees for the ML patterns found in one module."""
    tree = ast.parse(source)
    names = _resolve_names(tree, module)
    edges: dict[str, set[str]] = {}
    for scope, node in _calls(tree, module):
        for caller, callee in _edges_of_call(node, scope, names):
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
    node: ast.Call, scope: str, names: dict[str, str]
) -> list[tuple[str, str]]:
    target = _qualify(node.func, names)

    if isinstance(node.func, ast.Attribute) and node.func.attr in PIPE_METHODS:
        return [(scope, callee) for callee in _callables(node.args[:1], names)]

    if target in SEQUENTIAL_PIPELINES:
        return _pipeline_edges(node, names, sequential=True)

    if target in PARALLEL_PIPELINES:
        return _pipeline_edges(node, names, sequential=False)

    if target in BOOSTING_TRAINERS:
        passed = [kw.value for kw in node.keywords] + node.args
        return [(target, callee) for callee in _callables(passed, names)]

    return []


def _pipeline_edges(
    node: ast.Call, names: dict[str, str], sequential: bool
) -> list[tuple[str, str]]:
    """Model what `fit` does: transform every step, then fit the estimator."""
    owner = _qualify(node.func, names)
    if owner is None:
        return []
    # make_pipeline takes the steps as plain arguments; the classes take a list.
    steps = [_step_estimator(element, names) for element in _step_elements(node)]
    resolved = [step for step in steps if step]
    if not resolved:
        return []

    caller = f"{owner}.fit"
    if not sequential:
        return [(caller, f"{step}.fit_transform") for step in resolved]
    edges = [(caller, f"{step}.fit_transform") for step in resolved[:-1]]
    edges.append((caller, f"{resolved[-1]}.fit"))
    return edges


def _step_elements(node: ast.Call) -> list[ast.expr]:
    for keyword in node.keywords:
        if keyword.arg in STEPS_KEYWORDS:
            return _unpack(keyword.value)
    if node.args:
        return _unpack(node.args[0]) or list(node.args)
    return []


def _unpack(value: ast.expr) -> list[ast.expr]:
    if isinstance(value, ast.List | ast.Tuple):
        return list(value.elts)
    return []


def _step_estimator(element: ast.expr, names: dict[str, str]) -> str | None:
    """A step is either a bare estimator or a (name, estimator, ...) tuple."""
    if isinstance(element, ast.Tuple) and len(element.elts) >= 2:
        element = element.elts[1]
    if isinstance(element, ast.Call):
        return _qualify(element.func, names)
    return _qualify(element, names)


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
