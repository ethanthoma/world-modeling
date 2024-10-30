from typing import Any, Callable, TypeVar

import torch

T = TypeVar("T")


def nested_map(f: Callable[..., T], structure: Any, *args: Any) -> T:
    def _nested_map(
        f: Callable[..., T], structure_args: tuple[Any, ...], *args: Any
    ) -> T:
        if isinstance(structure_args[0], torch.Tensor):
            return f(*structure_args, *args)
        elif isinstance(structure_args[0], dict):
            keys = structure_args[0].keys()
            return {
                k: _nested_map(f, tuple(s[k] for s in structure_args), *args)
                for k in keys
            }
        elif isinstance(structure_args[0], list):
            return [
                _nested_map(f, tuple(s[i] for s in structure_args), *args)
                for i in range(len(structure_args[0]))
            ]
        return structure_args[0]

    if isinstance(structure, tuple):
        return _nested_map(f, structure, *args)
    return _nested_map(f, (structure,), *args)


def parameter_count(params: Any) -> int:
    count = 0

    def _parameter_count(t: torch.Tensor) -> torch.Tensor:
        nonlocal count
        count += t.numel()
        return t

    nested_map(_parameter_count, params)

    return count
