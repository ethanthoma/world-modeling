<h1 align="center">
    WorldFormer Model in PyTorch Functional
</h1>

> [!CAUTION]
> This is very WIP, implementation details may be wrong

This is an implementation of the [WorldFormer model](https://arxiv.org/pdf/2106.09608)

The codebase is made using PyTorch in a more functional style, similar to JAX

## Running

This project uses [uv](https://github.com/astral-sh/uv) as its package manager for Python.
It runs Python 3.12

The [flake.nix](./flake.nix) uses [uv2nix](https://github.com/adisbladis/uv2nix). This 
should be enough to setup your environment

The [scripts](./scripts/) dir contains helper scripts in bash to download the weights
and dataset

Dependency-wise, it is very sparse. The code only relies on PyTorch for almost 
all operations and Transformers for tokenization

## Performance

The model takes about 18GBs of RAM during training with batch size 16. No testing
has been performed (yet)
