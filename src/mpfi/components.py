"""MPFI — a static index of structural fragility in tabular ML pipelines.

Three components, each read off the source and the call graph:

  * FCI, how tightly the feature-engineering functions are wired to each other;
  * PDD, the longest chain of transformations data passes through;
  * SCC, the share of data boundaries that carry an explicit contract.

They are reported raw. Normalising them against a stratum and weighting them
into one index needs the sample, and belongs to the analysis rather than here.
"""

import ast
import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

from pycg_ml.generator import CallGraphGenerator
from pycg_ml.ml_patterns import ml_edges
from pycg_ml.utils import constants

# Calls that reshape a table rather than merely read from it.
TRANSFORM_CALLS = frozenset(
    {
        "assign",
        "merge",
        "join",
        "groupby",
        "agg",
        "apply",
        "applymap",
        "map",
        "pipe",
        "fillna",
        "dropna",
        "drop",
        "replace",
        "rename",
        "pivot",
        "pivot_table",
        "melt",
        "concat",
        "get_dummies",
        "cut",
        "qcut",
        "resample",
        "fit_transform",
        "transform",
        "inverse_transform",
        "encode",
        "scale",
    }
)

# Where a table enters or leaves the program.
IO_CALLS = frozenset(
    {
        "read_csv",
        "read_parquet",
        "read_json",
        "read_sql",
        "read_excel",
        "read_feather",
        "read_pickle",
        "read_table",
        "to_csv",
        "to_parquet",
        "to_json",
        "to_sql",
        "to_excel",
        "to_feather",
        "to_pickle",
    }
)

# Markers that say what a frame is allowed to contain.
CONTRACT_DECORATORS = frozenset(
    {"check_types", "check_input", "check_output", "validate_call"}
)
CONTRACT_HINTS = frozenset(
    {"DataFrame", "Series", "DataFrameModel", "SchemaModel", "BaseModel"}
)


@dataclass(frozen=True)
class Components:
    fci: float
    pdd: int
    scc: float
    feature_nodes: frozenset[str]
    boundaries: int


def components(package: Path) -> Components:
    """Measure one package; an empty or unparsable package scores zero."""
    modules = dict(_modules(package))
    graph = _call_graph(package)
    features = _feature_nodes(modules, graph)
    covered, total = _boundary_coverage(modules)
    return Components(
        fci=_coupling(graph, features),
        pdd=_depth(graph, features),
        scc=covered / total if total else 0.0,
        feature_nodes=frozenset(features),
        boundaries=total,
    )


def components_for_source(directory: Path, source: str) -> Components:
    """Measure a single module written out as a one-file package."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "main.py").write_text(source)
    return components(directory)


def _modules(package: Path) -> list[tuple[str, ast.Module]]:
    found = []
    for path in sorted(package.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except (OSError, SyntaxError, ValueError):
            continue
        found.append((_module_name(path, package), tree))
    return found


def _module_name(path: Path, package: Path) -> str:
    parts = path.relative_to(package).with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _call_graph(package: Path) -> dict[str, set[str]]:
    entries = sorted(str(p) for p in package.rglob("*.py"))
    if not entries:
        return {}
    generator = CallGraphGenerator(entries, str(package), -1, constants.CALL_GRAPH_OP)
    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            generator.analyze()
            graph: dict[str, set[str]] = generator.output()
            return graph
    except Exception:
        return {}


def _feature_nodes(
    modules: dict[str, ast.Module], graph: dict[str, set[str]]
) -> set[str]:
    """A function is feature engineering if it reshapes a table or is handed one."""
    nodes = set()
    for module, tree in modules.items():
        for scope, node in _functions(tree, module):
            if any(_is_transform(call) for call in ast.walk(node)):
                nodes.add(scope)
        for callees in ml_edges(ast.unparse(tree), module).values():
            nodes.update(c for c in callees if c in graph or c.startswith(f"{module}."))
    return nodes


def _functions(tree: ast.Module, module: str) -> list[tuple[str, ast.AST]]:
    found = []

    def walk(node, scope):
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


def _is_transform(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in TRANSFORM_CALLS
    )


def _coupling(graph: dict[str, set[str]], features: set[str]) -> float:
    """Edges that run between feature-engineering functions, per such function."""
    if not features:
        return 0.0
    internal = sum(
        len(callees & features) for node, callees in graph.items() if node in features
    )
    return internal / len(features)


def _depth(graph: dict[str, set[str]], features: set[str]) -> int:
    """Longest chain of feature-engineering functions, counted in functions."""
    if not features:
        return 0
    memo: dict[str, int] = {}

    def longest(node: str, seen: frozenset[str]) -> int:
        if node in memo and node not in seen:
            return memo[node]
        best = 1
        for callee in sorted(graph.get(node, ())):
            if callee in features and callee not in seen:
                best = max(best, 1 + longest(callee, seen | {callee}))
        memo[node] = best
        return best

    return max(longest(node, frozenset({node})) for node in features)


def _boundary_coverage(modules: dict[str, ast.Module]) -> tuple[int, int]:
    covered = total = 0
    for _, tree in modules.items():
        for scope, node in _functions(tree, ""):
            io_calls = [c for c in ast.walk(node) if _is_io(c)]
            if not io_calls:
                continue
            total += len(io_calls)
            if _has_contract(node):
                covered += len(io_calls)
    return covered, total


def _is_io(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    name = (
        node.func.attr
        if isinstance(node.func, ast.Attribute)
        else getattr(node.func, "id", "")
    )
    return name in IO_CALLS


def _has_contract(node: ast.AST) -> bool:
    """An annotation, a validating decorator or an assertion on the way out."""
    for decorator in getattr(node, "decorator_list", []):
        if _decorator_name(decorator) in CONTRACT_DECORATORS:
            return True
    if _mentions_contract(getattr(node, "returns", None)):
        return True
    signature = getattr(node, "args", None)
    if signature is not None:
        for arg in signature.args:
            if _mentions_contract(arg.annotation):
                return True
    return any(isinstance(child, ast.Assert) for child in ast.walk(node))


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    return getattr(node, "id", "")


def _mentions_contract(annotation: ast.AST | None) -> bool:
    if annotation is None:
        return False
    return any(
        (isinstance(n, ast.Name) and n.id in CONTRACT_HINTS)
        or (isinstance(n, ast.Attribute) and n.attr in CONTRACT_HINTS)
        for n in ast.walk(annotation)
    )


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Structural fragility of a tabular ML package"
    )
    parser.add_argument("package", help="root of the package to measure")
    args = parser.parse_args()

    result = components(Path(args.package))
    print(
        json.dumps(
            {
                "package": args.package,
                "fci": round(result.fci, 4),
                "pdd": result.pdd,
                "scc": round(result.scc, 4),
                "feature_nodes": len(result.feature_nodes),
                "boundaries": result.boundaries,
            }
        )
    )


if __name__ == "__main__":
    main()
