# mpfi

MPFI is a static index of structural fragility in tabular ML pipelines. CES is a perturbation harness used to validate it. Research code for an MSc thesis at MIPT; definitions may change until the study protocol is pre-registered.

## MPFI

`mpfi` parses every module of the repository except tests and builds one graph over its functions. An edge `a → b` means one of three things:

- `a` calls `b`. `df.pipe(b)` inside `a` counts as a call. Names are resolved through imports, `self`, inheritance and held instances. There is no points-to analysis, so `f = transform; f(df)` is not resolved.
- The result of `a` is passed to `b`, as in `x = a(df); y = b(x)`, `b(a(df).fillna(0))` or `df.pipe(a).pipe(b)`. This is the data flow between functions.
- `a` and `b` are consecutive steps of an sklearn `Pipeline`. A step is a package function wrapped in `FunctionTransformer` or a package class, represented by its `transform`, `fit_transform` or `fit` method. The branches of a `ColumnTransformer` or `FeatureUnion` are not linked to each other. Each branch takes the edge from the step before and gives one to the step after.

A function is a feature-engineering node if it reshapes a table (`merge`, `groupby`, `fillna`, `fit_transform` and similar) or is used as a step: a pipeline step, a `.pipe` target, or a callback passed to `lightgbm.train` or `xgboost.train`. FCI and PDD are computed on the subgraph of these nodes:

- FCI (Feature Coupling Index): edges between feature-engineering nodes divided by the number of such nodes.
- PDD (Pipeline Dependency Depth): the number of functions on the longest path through feature-engineering nodes.

SCC (Schema Contract Coverage) does not use the graph. It counts data boundaries in the AST (I/O calls such as `read_csv` or `to_parquet` inside a function, and functions that take a table) and reports the share that carry a contract: a `DataFrame` or schema annotation, a validating decorator, or an `assert`.

The result does not depend on the hash seed.

```bash
uv sync
uv run mpfi path/to/repository
```

Output is one JSON line. For a small churn model where `load`, `clean`, `add_ratios` and `encode` are chained with `.pipe` and a `train` function fits the result:

```json
{"package": "churn_model", "fci": 0.5, "pdd": 3, "scc": 0.25, "feature_nodes": 4, "boundaries": 4}
```

`feature_nodes` counts feature-engineering functions. `boundaries` counts the data boundaries SCC is taken over.

## CES

`mpfi.ces` perturbs data columns (MSS) and pipeline stages (PPS) and measures prediction churn: the share of test objects whose prediction changed, minus the churn between model seeds. It needs pandas, NumPy and scikit-learn:

```bash
pip install -e ".[ces]"
```

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy src/
```

## License

MIT
