# Rename template placeholders to your project.
#   DIST_NAME  = name on PyPI and in repo (hyphens ok), e.g. my-tool
#   MODULE_NAME = name for "import x" and "python -m x" (underscores), e.g. my_tool
# Usage: make rename DIST_NAME=my-tool MODULE_NAME=my_tool
# Then: uv sync && uv run pytest

DIST_NAME ?=
MODULE_NAME ?=

ifeq ($(DIST_NAME),)
$(error DIST_NAME is required. Example: make rename DIST_NAME=my-tool MODULE_NAME=my_tool)
endif
ifeq ($(MODULE_NAME),)
$(error MODULE_NAME is required. Example: make rename DIST_NAME=my-tool MODULE_NAME=my_tool)
endif

.PHONY: rename
rename:
	@test -d src/pypi_package_uv_template || (echo "Already renamed or wrong dir."; exit 1)
	@echo "Renaming to dist=$(DIST_NAME) module=$(MODULE_NAME)..."
	sed -i.bak 's/pypi-package-uv-template/$(DIST_NAME)/g' pyproject.toml README.md mkdocs.yml docs/index.md src/pypi_package_uv_template/__init__.py src/pypi_package_uv_template/main.py
	sed -i.bak 's/pypi_package_uv_template/$(MODULE_NAME)/g' pyproject.toml src/pypi_package_uv_template/__init__.py src/pypi_package_uv_template/main.py src/pypi_package_uv_template/__main__.py tests/test_main.py docs/api.md docs/index.md README.md
	@find . -name '*.bak' -delete
	mv src/pypi_package_uv_template src/$(MODULE_NAME)
	@echo "Done. Run: uv sync && uv run pytest"
