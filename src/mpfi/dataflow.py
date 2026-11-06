"""MPFI over data flow: FCI and PDD for flat notebook code.

The function graph of `components` sees feature engineering only where it is
wrapped in functions or pipeline steps. Notebooks write it as top-level
statements, so most of them score zero there. This module reads the same two
components off the data flow of the statements instead, the way provenance and
leakage analyses read notebooks (Namaki et al., KDD 2020; Yang et al., ASE 2022):
assignments in static single assignment form, a table of what pandas, NumPy and
scikit-learn calls do to a table, and a backward slice from the data the final
model is fitted on, so exploratory code and abandoned models drop out.

Two graphs come out of one pass:

  * operations: a node per statement that changes a table, an edge where one
    statement uses a table another defined; a statement derives features if it
    writes a column, filters rows or transforms a whole table;
  * columns: a node per version of a column, an edge from each column a new
    version is computed from; row filters select rows and derive no column.

On each, PDD is the largest number of deriving nodes on one path into the model
and FCI the edges between deriving nodes per deriving node, as in `components`.

The analysis is an approximation. User functions are opaque steps over their
table arguments, loops run once unless they iterate over literal column names,
both branches of a conditional are taken, and names are not tracked through
attributes, containers or aliases other than plain assignment.
"""

import ast
from dataclasses import dataclass, field, replace

# Read straight off a table without changing it.
ALIAS_ATTRIBUTES = frozenset({"values", "T", "array"})
ACCESSORS = frozenset({"loc", "iloc", "at", "iat", "str", "dt", "cat"})
META_ATTRIBUTES = frozenset(
    {"shape", "columns", "index", "dtypes", "dtype", "size", "ndim", "empty", "name"}
)

PASS_METHODS = frozenset(
    {
        "copy",
        "reset_index",
        "set_index",
        "sort_values",
        "sort_index",
        "reindex",
        "head",
        "tail",
        "to_numpy",
        "to_frame",
        "squeeze",
        "ravel",
        "reshape",
        "flatten",
        "tolist",
        "infer_objects",
    }
)
ROW_METHODS = frozenset(
    {"dropna", "drop_duplicates", "query", "nlargest", "nsmallest", "sample"}
)
JOIN_METHODS = frozenset({"merge", "join"})
# Reduce a table to numbers that parametrise a later step; not a feature.
AGGREGATE_METHODS = frozenset(
    {
        "mean",
        "median",
        "sum",
        "std",
        "var",
        "min",
        "max",
        "count",
        "nunique",
        "mode",
        "quantile",
        "value_counts",
        "corr",
        "describe",
        "unique",
        "idxmax",
        "idxmin",
        "any",
        "all",
        "isnull",
        "isna",
        "notnull",
        "notna",
        "info",
        "plot",
        "hist",
        "duplicated",
    }
)
GROUP_RESULTS = frozenset(
    AGGREGATE_METHODS | {"transform", "agg", "aggregate", "apply", "size", "first"}
)

PASS_FUNCTIONS = frozenset(
    {
        "DataFrame",
        "Series",
        "array",
        "asarray",
        "hstack",
        "vstack",
        "column_stack",
        "concatenate",
        "concat",
        "csr_matrix",
        "copy",
        "deepcopy",
    }
)
# Library containers built from features and a label: only the features count.
FEATURE_CONTAINERS = frozenset({"Dataset", "DMatrix", "Pool"})
SPLITTERS = frozenset({"train_test_split"})
BOOSTING_ROOTS = frozenset({"lightgbm", "xgboost", "catboost"})
CROSS_VALIDATION = frozenset({"cross_val_score", "cross_validate", "cross_val_predict"})
FIT_METHODS = frozenset({"fit", "fit_predict"})
TRANSFORM_METHODS = frozenset({"fit_transform", "transform"})
RESAMPLE_METHODS = frozenset({"fit_resample", "fit_sample"})

Key = str | list[str] | None

# Notebook functions call each other only a few levels deep; deeper chains are
# cut and read as an opaque step over their arguments.
MAX_INLINE_DEPTH = 4


@dataclass
class Frame:
    """The columns of one table version: named ones and the rest, unnamed."""

    columns: dict[str, int] = field(default_factory=dict)
    rest: int | None = None

    def nodes(self) -> set[int]:
        found = set(self.columns.values())
        if self.rest is not None:
            found.add(self.rest)
        return found


@dataclass(frozen=True)
class Value:
    """What an expression evaluates to, as far as data flow is concerned."""

    ops: frozenset[int] = frozenset()
    frame: Frame | None = None
    reads: frozenset[int] = frozenset()
    extra: frozenset[int] = frozenset()
    transformed: bool = False
    rows: bool = False
    pure: bool = False
    scalar: bool = False
    grouped: bool = False
    accessor: str | None = None
    items: tuple["Value", ...] | None = None

    @property
    def data(self) -> bool:
        return bool(self.frame is not None or self.ops or self.reads or self.items)

    def all_reads(self) -> set[int]:
        found = set(self.reads)
        if self.frame is not None:
            found |= self.frame.nodes()
        for item in self.items or ():
            found |= item.all_reads()
        return found


@dataclass(frozen=True)
class FlowComponents:
    op_fci: float
    op_pdd: int
    op_nodes: int
    col_fci: float
    col_pdd: int
    col_nodes: int
    col_fanin: float
    statements: int
    sliced_statements: int
    sink_line: int | None


@dataclass
class _Node:
    parents: frozenset[int]
    deriving: bool
    line: int


ZERO = FlowComponents(0.0, 0, 0, 0.0, 0, 0, 0.0, 0, 0, None)


def flow_components(source: str) -> FlowComponents:
    """Measure one script; no final model fit, or no parse, scores zero."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return ZERO
    flow = _Flow(tree)
    flow.block(tree.body)
    return flow.measure()


def _union(values: list[Value]) -> Value:
    frames = [v.frame for v in values if v.frame is not None]
    frame = None
    if frames:
        frame = Frame()
        for f in frames:
            for key, node in f.columns.items():
                frame.columns.setdefault(key, node)
            frame.rest = frame.rest if frame.rest is not None else f.rest
    return Value(
        ops=frozenset().union(*(v.ops for v in values)),
        frame=frame,
        reads=frozenset().union(*(v.reads for v in values if v.frame is None)),
        extra=frozenset().union(*(v.extra for v in values)),
    )


class _Flow:
    def __init__(self, tree: ast.Module) -> None:
        self.names = _imports(tree)
        self.functions = {
            n.name: n
            for n in tree.body
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        self.calling: list[str] = []
        self.returns: list[list[Value]] = []
        self.env: dict[str, Value] = {}
        self.consts: dict[str, str | list[str]] = {}
        self.ops: list[_Node] = []
        self.cols: list[_Node] = []
        self.sinks: list[tuple[int, Value]] = []
        self.statements = sum(isinstance(n, ast.stmt) for n in ast.walk(tree))
        self.line = 0

    # statements

    def block(self, body: list[ast.stmt]) -> None:
        for stmt in body:
            self.statement(stmt)

    def statement(self, stmt: ast.stmt) -> None:
        self.line = stmt.lineno
        match stmt:
            case ast.Assign(targets=targets, value=value):
                self.assignment(targets, value)
            case ast.AnnAssign(target=target, value=value) if value is not None:
                self.assignment([target], value)
            case ast.AugAssign(target=target, op=op, value=value):
                load = _as_load(target)
                self.assignment([target], ast.BinOp(load, op, value))
            case ast.Expr(value=ast.Call() as call):
                self.expression_call(call)
            case ast.Return(value=value) if self.returns:
                self.returns[-1].append(self.evaluate(value))
            case ast.For(target=target, iter=iterable, body=body, orelse=orelse):
                self.loop(target, iterable, body)
                self.block(orelse)
            case ast.While(body=body, orelse=orelse) | ast.If(body=body, orelse=orelse):
                self.block(body)
                self.block(orelse)
            case ast.With(body=body):
                self.block(body)
            case ast.Try(body=body, handlers=handlers, orelse=orelse, finalbody=final):
                self.block(body)
                for handler in handlers:
                    self.block(handler.body)
                self.block(orelse)
                self.block(final)

    def assignment(self, targets: list[ast.expr], value: ast.expr) -> None:
        literal = self.key(value)
        if (
            isinstance(literal, str | list)
            and len(targets) == 1
            and isinstance(targets[0], ast.Name)
        ):
            self.consts[targets[0].id] = literal
            self.env.pop(targets[0].id, None)
            return
        result = self.evaluate(value)
        for target in targets:
            self.bind(target, result)

    def expression_call(self, call: ast.Call) -> None:
        inplace = any(
            kw.arg == "inplace"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is True
            for kw in call.keywords
        )
        func = call.func
        if (
            inplace
            and isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
        ):
            self.bind(ast.Name(func.value.id, ast.Store()), self.evaluate(call))
            return
        self.evaluate(call)

    def loop(self, target: ast.expr, iterable: ast.expr, body: list[ast.stmt]) -> None:
        keys = self.key(iterable)
        if isinstance(target, ast.Name) and isinstance(keys, list) and keys:
            for key in keys:
                self.consts[target.id] = key
                self.block(body)
            self.consts.pop(target.id, None)
            return
        for name in _names(target):
            self.consts.pop(name, None)
            self.env.pop(name, None)
        self.block(body)

    def bind(self, target: ast.expr, value: Value) -> None:
        match target:
            case ast.Name(id=name):
                self.bind_name(name, value)
            case ast.Tuple(elts=elts) | ast.List(elts=elts):
                items = value.items
                if items is not None and len(items) == len(elts):
                    for element, item in zip(elts, items, strict=True):
                        self.bind(element, item)
                else:
                    for element in elts:
                        self.bind(element, value)
            case ast.Subscript(value=ast.Name(id=name), slice=key):
                self.bind_columns(name, key, None, value)
            case ast.Subscript(
                value=ast.Attribute(value=ast.Name(id=name), attr="loc" | "at"),
                slice=key,
            ):
                rows, columns = _split_loc(key)
                self.bind_columns(name, columns, rows, value)

    def bind_name(self, name: str, value: Value) -> None:
        self.consts.pop(name, None)
        if not value.data:
            self.env.pop(name, None)
            return
        if value.items is not None:
            value = _union(list(value.items))
        if value.frame is None:
            if value.scalar or value.pure:
                op = self.op(False, value.ops)
                self.env[name] = replace(value, ops=frozenset({op}))
                return
            node = self.column(value.all_reads() | value.extra, True)
            op = self.op(True, value.ops)
            self.env[name] = Value(
                ops=frozenset({op}), frame=Frame({name: node}), pure=True
            )
            return
        if value.transformed:
            frame = self.derive(value.frame, set(value.extra))
        else:
            frame = value.frame
        op = self.op(value.transformed or value.rows, value.ops)
        self.env[name] = Value(ops=frozenset({op}), frame=frame, pure=True)

    def bind_columns(
        self, name: str, key: ast.expr | None, rows: ast.expr | None, value: Value
    ) -> None:
        base = self.env.get(name)
        if base is None or base.frame is None:
            return
        frame = Frame(dict(base.frame.columns), base.frame.rest)
        mask = self.evaluate(rows) if rows is not None else Value()
        parents = value.all_reads() | set(value.extra) | mask.all_reads()
        keys = self.key(key)
        if isinstance(keys, str):
            keys = [keys]
        if isinstance(keys, list):
            reads = value.all_reads()
            copy = value.pure and len(reads) == 1 and rows is None
            for k in keys:
                frame.columns[k] = (
                    next(iter(reads)) if copy else self.column(parents, True)
                )
            deriving = not copy
        else:
            frame = self.derive(frame, parents - base.frame.nodes())
            deriving = True
        op = self.op(deriving, base.ops | value.ops | mask.ops)
        self.env[name] = Value(ops=frozenset({op}), frame=frame, pure=True)

    # expressions

    def evaluate(self, node: ast.expr | None) -> Value:
        match node:
            case None | ast.Constant() | ast.Lambda():
                return Value()
            case ast.Name(id=name):
                return self.env.get(name, Value())
            case ast.Attribute(value=inner, attr=attr):
                return self.attribute(self.evaluate(inner), attr)
            case ast.Subscript(value=inner, slice=key):
                return self.subscript(self.evaluate(inner), key)
            case ast.Call():
                return self.call(node)
            case ast.BinOp(left=left, right=right):
                return self.combine([self.evaluate(left), self.evaluate(right)])
            case ast.UnaryOp(operand=operand):
                return self.combine([self.evaluate(operand)])
            case ast.Compare(left=left, comparators=rest):
                return self.combine([self.evaluate(e) for e in [left, *rest]])
            case ast.BoolOp(values=operands):
                return self.combine([self.evaluate(e) for e in operands])
            case ast.IfExp(body=body, orelse=orelse):
                return _union([self.evaluate(body), self.evaluate(orelse)])
            case ast.Tuple(elts=elts) | ast.List(elts=elts):
                parts = [self.evaluate(e) for e in elts]
                if not any(v.data for v in parts):
                    return Value()
                return replace(_union(parts), items=tuple(parts), pure=True)
        return Value()

    def combine(self, values: list[Value]) -> Value:
        data = [v for v in values if v.data]
        if not data:
            return Value()
        table = next(
            (v for v in data if v.frame is not None and v.frame.rest is not None), None
        )
        ops = frozenset().union(*(v.ops for v in data))
        if table is not None:
            others = set().union(*(v.all_reads() for v in data if v is not table))
            return Value(
                ops=ops, frame=table.frame, extra=frozenset(others), transformed=True
            )
        reads = set().union(*(v.all_reads() | v.extra for v in data))
        return Value(ops=ops, reads=frozenset(reads))

    def attribute(self, base: Value, attr: str) -> Value:
        if base.frame is None:
            return base if base.data and attr in ALIAS_ATTRIBUTES else Value()
        if attr in META_ATTRIBUTES:
            return Value()
        if attr in ACCESSORS:
            return replace(base, accessor=attr)
        if attr in ALIAS_ATTRIBUTES:
            return base
        return self.columns_of(base, attr)

    def subscript(self, base: Value, key: ast.expr) -> Value:
        if not base.data:
            return Value()
        if base.frame is None:
            return replace(base, pure=False)
        accessor = base.accessor
        base = replace(base, accessor=None)
        if accessor in ("loc", "iloc", "at", "iat"):
            rows, columns = _split_loc(key)
            mask = self.evaluate(rows) if rows is not None else Value()
            picked = self.columns_of(base, columns) if columns is not None else base
            if accessor == "loc" and mask.data:
                return replace(
                    picked,
                    ops=picked.ops | mask.ops,
                    extra=picked.extra | frozenset(mask.all_reads()),
                    rows=True,
                    pure=False,
                )
            return picked
        if accessor in ("str", "dt", "cat"):
            return replace(base, transformed=True, pure=False)
        keys = self.key(key)
        if keys is None:
            mask = self.evaluate(key)
            if mask.data:
                return replace(
                    base,
                    ops=base.ops | mask.ops,
                    extra=base.extra | frozenset(mask.all_reads()),
                    rows=True,
                    pure=False,
                )
            return base
        return self.columns_of(base, key)

    def columns_of(self, base: Value, key: ast.expr | str) -> Value:
        keys = key if isinstance(key, str) else self.key(key)
        if keys is None or base.frame is None:
            return base
        names = [keys] if isinstance(keys, str) else list(keys)
        frame = Frame({k: self.lookup(base.frame, k) for k in names})
        return replace(base, frame=frame)

    def lookup(self, frame: Frame, key: str) -> int:
        if key not in frame.columns:
            parents = frozenset() if frame.rest is None else frozenset({frame.rest})
            frame.columns[key] = self.column(parents, False)
        return frame.columns[key]

    def call(self, node: ast.Call) -> Value:
        args = [self.evaluate(a) for a in node.args if not isinstance(a, ast.Starred)]
        keywords = {kw.arg: self.evaluate(kw.value) for kw in node.keywords if kw.arg}
        func = node.func
        if isinstance(func, ast.Attribute):
            receiver = self.evaluate(func.value)
            if receiver.data:
                return self.method(receiver, func.attr, node, args, keywords)
            if _qualified(func, self.names) is None:
                return self.object_method(func.attr, node, args, keywords)
        name = _qualified(func, self.names) or ""
        return self.function(name, node, args, keywords)

    def method(
        self,
        receiver: Value,
        name: str,
        node: ast.Call,
        args: list[Value],
        keywords: dict[str, Value],
    ) -> Value:
        passed = [v for v in [*args, *keywords.values()] if v.data]
        ops = receiver.ops.union(*(v.ops for v in passed))
        extra = frozenset(set().union(*(v.all_reads() for v in passed)))
        if receiver.grouped and name in GROUP_RESULTS:
            return replace(receiver, grouped=False)
        if receiver.frame is None:
            return Value(ops=ops, reads=frozenset(receiver.all_reads()), scalar=True)
        if receiver.accessor in ("str", "dt", "cat"):
            return replace(
                receiver, accessor=None, ops=ops, transformed=True, pure=False
            )
        if name == "groupby":
            keys = self.key(node.args[0]) if node.args else None
            by = (
                self.columns_of(receiver, node.args[0]).all_reads()
                if keys is not None
                else set()
            )
            return replace(
                receiver,
                extra=receiver.extra | frozenset(by),
                transformed=True,
                grouped=True,
                pure=False,
            )
        if name == "drop":
            return self.drop(receiver, node)
        if name in PASS_METHODS:
            return replace(receiver, ops=ops)
        if name in ROW_METHODS:
            return replace(
                receiver, ops=ops, extra=receiver.extra | extra, rows=True, pure=False
            )
        if name in JOIN_METHODS:
            joined = _union([receiver, *passed])
            return replace(joined, rows=True)
        if name in AGGREGATE_METHODS:
            return Value(ops=ops, reads=frozenset(receiver.all_reads()), scalar=True)
        return replace(
            receiver,
            ops=ops,
            extra=receiver.extra | extra,
            transformed=True,
            pure=False,
        )

    def drop(self, receiver: Value, node: ast.Call) -> Value:
        assert receiver.frame is not None
        keywords = {kw.arg: kw.value for kw in node.keywords}
        if "index" in keywords:
            return replace(receiver, rows=True, pure=False)
        target = keywords.get("columns") or keywords.get("labels")
        if target is None and node.args:
            target = node.args[0]
        keys = self.key(target) if target is not None else None
        if keys is None:
            return receiver
        dropped = {keys} if isinstance(keys, str) else set(keys)
        frame = Frame(
            {k: v for k, v in receiver.frame.columns.items() if k not in dropped},
            receiver.frame.rest,
        )
        return replace(receiver, frame=frame)

    def object_method(
        self, name: str, node: ast.Call, args: list[Value], keywords: dict[str, Value]
    ) -> Value:
        if name in FIT_METHODS and (len(node.args) >= 2 or "y" in keywords):
            features = keywords.get("X") or (args[0] if args else Value())
            self.sinks.append((self.line, features))
            return Value()
        if name in TRANSFORM_METHODS and args and args[0].data:
            first = args[0]
            if first.frame is None:
                return replace(first, pure=False)
            return replace(first, transformed=True, pure=False)
        if name in RESAMPLE_METHODS and len(args) >= 2:
            return Value(
                items=(replace(args[0], rows=True), replace(args[1], rows=True))
            )
        data = [v for v in [*args, *keywords.values()] if v.data]
        if not data:
            return Value()
        return self.combine(data) if len(data) > 1 else replace(_strip(data[0]))

    def function(
        self, name: str, node: ast.Call, args: list[Value], keywords: dict[str, Value]
    ) -> Value:
        if name in self.functions:
            return self.inline(self.functions[name], node, args, keywords)
        last = name.rsplit(".", 1)[-1]
        root = name.split(".", 1)[0]
        if last.startswith("read_"):
            return Value(frame=Frame(rest=self.column(frozenset(), False)), pure=True)
        if root in BOOSTING_ROOTS and last == "train":
            features = keywords.get("train_set") or keywords.get("dtrain")
            if features is None and len(args) >= 2:
                features = args[1]
            self.sinks.append((self.line, features or Value()))
            return Value()
        if last in CROSS_VALIDATION:
            features = keywords.get("X") or (args[1] if len(args) >= 2 else Value())
            self.sinks.append((self.line, features))
            return Value()
        if last in FEATURE_CONTAINERS:
            features = keywords.get("data") or (args[0] if args else Value())
            return features
        if last in SPLITTERS:
            arrays = [a for a in args if a.data]
            return Value(items=tuple(v for a in arrays for v in (a, a)))
        data = [v for v in [*args, *keywords.values()] if v.data]
        if not data:
            return Value()
        if last == "merge":
            return replace(_union(data), rows=True)
        if last in PASS_FUNCTIONS:
            return replace(_union(data), pure=True)
        return self.combine(data) if len(data) > 1 else _elementwise(data[0])

    def inline(
        self,
        definition: ast.FunctionDef | ast.AsyncFunctionDef,
        node: ast.Call,
        args: list[Value],
        keywords: dict[str, Value],
    ) -> Value:
        """Follow a call into the notebook's own function, once per call site.

        The body runs with its parameters bound to the arguments and the
        module's names visible; its own names do not leak out. A parameter the
        body rebinds is written back to the caller's variable, which is how a
        function that fills a frame in place and returns nothing is seen.
        """
        if definition.name in self.calling or len(self.calling) >= MAX_INLINE_DEPTH:
            data = [v for v in [*args, *keywords.values()] if v.data]
            return _elementwise(_union(data)) if data else Value()
        params = [a.arg for a in definition.args.posonlyargs + definition.args.args]
        bound: dict[str, Value] = dict(zip(params, args, strict=False))
        bound.update({k: v for k, v in keywords.items() if k in params})
        callers = {
            param: arg.id
            for param, arg in zip(params, node.args, strict=False)
            if isinstance(arg, ast.Name)
        }
        outer_env, outer_consts, line = self.env, self.consts, self.line
        self.env = {**outer_env, **{k: v for k, v in bound.items() if v.data}}
        for param in params:
            if not bound.get(param, Value()).data:
                self.env.pop(param, None)
        self.consts = dict(outer_consts)
        for param in params:
            self.consts.pop(param, None)
        passed = list(zip(params, node.args, strict=False)) + [
            (kw.arg, kw.value) for kw in node.keywords if kw.arg in params
        ]
        for param, expr in passed:
            with_outer = self.consts
            self.consts = outer_consts
            literal = self.key(expr)
            self.consts = with_outer
            if literal is not None:
                self.consts[param] = literal
        self.calling.append(definition.name)
        self.returns.append([])
        try:
            self.block(definition.body)
        finally:
            self.calling.pop()
            returned = self.returns.pop()
            inner = self.env
            self.env, self.consts, self.line = outer_env, outer_consts, line
        for param, caller in callers.items():
            after = inner.get(param)
            if after is not None and after is not bound.get(param):
                self.env[caller] = after
        if not returned:
            return Value()
        if len(returned) == 1:
            return returned[0]
        return _union(returned)

    # graph

    def column(self, parents: set[int] | frozenset[int], deriving: bool) -> int:
        self.cols.append(_Node(frozenset(parents), deriving, self.line))
        return len(self.cols) - 1

    def op(self, deriving: bool, reads: frozenset[int]) -> int:
        self.ops.append(_Node(reads, deriving, self.line))
        return len(self.ops) - 1

    def derive(self, frame: Frame, extra: set[int]) -> Frame:
        columns = {k: self.column({v} | extra, True) for k, v in frame.columns.items()}
        rest = None if frame.rest is None else self.column({frame.rest} | extra, True)
        return Frame(columns, rest)

    def key(self, node: ast.expr | None) -> Key:
        """Column names an expression spells out, or None if it spells none."""
        match node:
            case ast.Constant(value=str() as text):
                return text
            case ast.List(elts=elts) | ast.Tuple(elts=elts):
                keys = [self.key(e) for e in elts]
                texts = [k for k in keys if isinstance(k, str)]
                if texts and len(texts) == len(keys):
                    return texts
            case ast.Name(id=name) if name in self.consts:
                return self.consts[name]
            case ast.BinOp(left=left, op=ast.Add(), right=right):
                a, b = self.key(left), self.key(right)
                if isinstance(a, str) and isinstance(b, str):
                    return a + b
            case ast.Call(
                func=ast.Attribute(
                    value=ast.Constant(value=str() as template), attr="format"
                ),
                args=args,
                keywords=[],
            ):
                parts = [self.key(a) for a in args]
                texts = [p for p in parts if isinstance(p, str)]
                if len(texts) == len(parts):
                    try:
                        return template.format(*texts)
                    except (IndexError, KeyError, ValueError):
                        return None
            case ast.JoinedStr(values=values):
                pieces: list[str] = []
                for value in values:
                    if isinstance(value, ast.Constant):
                        pieces.append(str(value.value))
                    elif isinstance(value, ast.FormattedValue):
                        piece = self.key(value.value)
                        if not isinstance(piece, str):
                            return None
                        pieces.append(piece)
                return "".join(pieces)
        return None

    def measure(self) -> FlowComponents:
        if not self.sinks:
            return replace(ZERO, statements=self.statements)
        line, features = self.sinks[-1]
        op_slice = _ancestors(self.ops, set(features.ops))
        col_slice = _ancestors(self.cols, features.all_reads() | set(features.extra))
        op_fci, op_pdd, op_nodes = _shape(self.ops, op_slice)
        col_fci, col_pdd, col_nodes = _shape(self.cols, col_slice)
        deriving = [n for n in col_slice if self.cols[n].deriving]
        fanin = (
            sum(len(self.cols[n].parents) for n in deriving) / len(deriving)
            if deriving
            else 0.0
        )
        return FlowComponents(
            op_fci=op_fci,
            op_pdd=op_pdd,
            op_nodes=op_nodes,
            col_fci=col_fci,
            col_pdd=col_pdd,
            col_nodes=col_nodes,
            col_fanin=fanin,
            statements=self.statements,
            sliced_statements=len({self.ops[n].line for n in op_slice}),
            sink_line=line,
        )


def _ancestors(nodes: list[_Node], start: set[int]) -> set[int]:
    seen: set[int] = set()
    stack = list(start)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(nodes[n].parents)
    return seen


def _shape(nodes: list[_Node], within: set[int]) -> tuple[float, int, int]:
    """FCI, PDD and the number of deriving nodes of a slice of a DAG.

    Node ids grow with definition, so every parent precedes its child.
    """
    depth: dict[int, int] = {}
    nearest: dict[int, frozenset[int]] = {}
    edges = 0
    for n in sorted(within):
        node = nodes[n]
        parents = [p for p in node.parents if p in within]
        below = frozenset().union(
            *(frozenset({p}) if nodes[p].deriving else nearest[p] for p in parents)
        )
        nearest[n] = below
        depth[n] = int(node.deriving) + max((depth[p] for p in parents), default=0)
        if node.deriving:
            edges += len(below)
    count = sum(nodes[n].deriving for n in within)
    return (edges / count if count else 0.0), max(depth.values(), default=0), count


def _strip(value: Value) -> Value:
    return Value(ops=value.ops, reads=frozenset(value.all_reads()))


def _elementwise(value: Value) -> Value:
    if value.frame is not None:
        return replace(value, transformed=True, pure=False, items=None)
    return Value(ops=value.ops, reads=frozenset(value.all_reads() | value.extra))


def _split_loc(key: ast.expr) -> tuple[ast.expr | None, ast.expr | None]:
    if isinstance(key, ast.Tuple) and len(key.elts) == 2:
        rows, columns = key.elts
        return (None if isinstance(rows, ast.Slice) else rows), columns
    return (None if isinstance(key, ast.Slice) else key), None


def _names(target: ast.expr) -> list[str]:
    return [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]


def _as_load(target: ast.expr) -> ast.expr:
    copied = ast.parse(ast.unparse(target), mode="eval").body
    return copied


def _imports(tree: ast.Module) -> dict[str, str]:
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return names


def _qualified(node: ast.expr, names: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return names.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _qualified(node.value, names)
        if (
            base is None
            or isinstance(node.value, ast.Name)
            and node.value.id not in names
        ):
            return None
        return f"{base}.{node.attr}"
    return None
