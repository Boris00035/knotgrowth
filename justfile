run:
    uv run python -m main

build:
    uv build
    uv pip install -e .

init:
    uv sync

br:
    uv build
    uv pip install -e .
    uv run python -m main
