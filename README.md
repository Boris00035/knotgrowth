## Prerequisites
Install [UV](https://docs.astral.sh/uv/getting-started/installation/) and [just](https://github.com/casey/just) (On windows just is probably installed fastest using the package manager [scoop](https://scoop.sh/)).

## Getting started 
Run: `just init` and `just build`. Now each time you edit a python file that is not converted to C functions, you can run `just run`. If you do change a python file that converts to C functions, you should run `just build` again. The python files which are converted to C are specified in `pyproject.toml`  

