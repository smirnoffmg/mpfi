# pypi-package-uv-template

A GitHub repository template for Python packages using [uv](https://docs.astral.sh/uv/), with ruff, mypy, pytest, pre-commit, MkDocs (API docs via mkdocstrings), GitHub Pages, and automatic PyPI release via trusted publishing.

**Use this template** — click "Use this template" above to create a new repo from this template.

### After creating your repo: replace the placeholder name

Run the Makefile with your **distribution name** (PyPI/repo, use hyphens) and **module name** (for `import` / `python -m`, use underscores):

```bash
make rename DIST_NAME=my-tool MODULE_NAME=my_tool
```

Then run `uv sync`, `uv run pytest`, and `uv run ruff check .` to confirm everything works.

**Manual alternative** — replace `pypi-package-uv-template` and `pypi_package_uv_template` everywhere:

| Location                                 | Change                                                                            |
| ---------------------------------------- | --------------------------------------------------------------------------------- |
| **Rename directory**                     | `src/pypi_package_uv_template/` → `src/<your_module>/` (e.g. `src/my_package/`)   |
| [pyproject.toml](pyproject.toml)         | `name = "..."` (PyPI name); `[tool.mypy]` → `packages = ["src/<your_module>"]`    |
| Python files in `src/`                   | Imports and any string that mentions the old name (e.g. the `print` in `main.py`) |
| [tests/test_main.py](tests/test_main.py) | `from <your_module> import ...` and the assertion text                            |
| [docs/api.md](docs/api.md)               | `::: <your_module>` (mkdocstrings directive)                                      |
| [docs/index.md](docs/index.md)           | Title and `python -m <your_module>` in the usage example                          |
| [mkdocs.yml](mkdocs.yml)                 | `site_name:` and `site_url:`                                                      |
| This README                              | Title (above) and the `python -m` command in Usage                                |

## Installation

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

Python 3.12+ is required (see [.python-version](.python-version)). uv will use it automatically.

## Usage

```bash
python -m pypi_package_uv_template
```

## Development

1. Install dependencies (including dev): `uv sync`
2. Install pre-commit hooks so checks run on commit: `pre-commit install`
3. Run checks manually:
   - `uv run ruff check .`
   - `uv run ruff format --check .`
   - `uv run mypy src/`
   - `uv run pytest`
4. Serve docs locally: `uv run mkdocs serve`
5. Add dependencies: `uv add <pkg>` or `uv add --group dev <pkg>`

To validate the whole repo: `pre-commit run --all-files`

## Documentation

API documentation is built with [MkDocs](https://www.mkdocs.org/) and [mkdocstrings](https://mkdocstrings.github.io/), and deployed to **GitHub Pages** on every push to `main`.

- **Local:** `uv run mkdocs serve`
- **Online:** After enabling Pages, docs will be at `https://<owner>.github.io/<repo>/`

To enable: In the repo **Settings → Pages**, set **Source** to **GitHub Actions** so Pages is served from the workflow.

## Releasing

Releases are published to **PyPI** automatically when you create a **GitHub Release**.

1. Configure **trusted publishing** for this repo on PyPI: go to your project on [pypi.org](https://pypi.org) → **Publishing** → **Add a new pending publisher**, and add the GitHub repo with workflow name `Publish to PyPI` and the appropriate ref (e.g. `release` for tags).
2. Create a new release (e.g. tag `v0.1.0`) and publish it; the [publish workflow](.github/workflows/publish.yml) will build and upload the package via OIDC (no API token needed).

## License

See [LICENSE](LICENSE) if present.
