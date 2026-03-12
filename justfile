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

blenderbuildw:
    uv build
    "C:\Users\Boris\scoop\apps\blender\current\5.0\python\bin\python.exe" -m pip install -e .