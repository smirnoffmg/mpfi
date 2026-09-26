# mpfi

MPFI is a static index of structural fragility in tabular ML pipelines. CES is a perturbation harness used to validate it. Research code for an MSc thesis at MIPT; definitions may change until the study protocol is pre-registered.

## MPFI

`mpfi` parses a repository and reports three components:

- FCI (Feature Coupling Index): calls and table hand-offs between feature-engineering functions, per function.
- PDD (Pipeline Dependency Depth): the longest chain of feature-engineering steps a table passes through.
- SCC (Schema Contract Coverage): the share of data boundaries with a schema contract.

The call graph is built from the AST without points-to analysis, so `f = transform; f(df)` is not resolved. Results are deterministic.

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
