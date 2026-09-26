# mpfi

MPFI is a static index of structural fragility in tabular ML pipelines.
CES is a perturbation harness used to validate it.

## Installation

```bash
uv sync
pip install -e ".[ces]"  # CES only
```

## Usage

```bash
uv run mpfi path/to/repository
```

See the [API reference](api.md) for the public API.
