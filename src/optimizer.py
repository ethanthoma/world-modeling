from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, TypeVar

import torch


def SGD(params: Any, grads: Any, lr: float) -> Any:
    def f(params: torch.Tensor, grads: torch.Tensor, lr: float):
        return params.sub_(lr * grads)

    return nested_map(f, (params, grads), lr)


@dataclass
class Adam_State:
    exp_avgs: Dict[int, torch.Tensor] = field(default_factory=dict)
    exp_avg_sqs: Dict[int, torch.Tensor] = field(default_factory=dict)
    state_steps: Dict[int, torch.Tensor] = field(default_factory=dict)


def Adam(
    params: Any,
    grads: Any,
    state: Adam_State,
    lr: float = 0.001,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> Any:
    def f(
        params: torch.Tensor,
        grads: torch.Tensor,
        state: Adam_State,
        lr: float,
        beta: Tuple[float, float],
        eps: float,
    ):
        param_id = id(params)

        if param_id not in state.exp_avgs:
            state.exp_avgs[param_id] = torch.zeros_like(params)
            state.exp_avg_sqs[param_id] = torch.zeros_like(params)
            state.state_steps[param_id] = torch.tensor(0.0)

        exp_avg = state.exp_avgs[param_id]
        exp_avg_sq = state.exp_avg_sqs[param_id]
        step = state.state_steps[param_id]

        step += 1
        state.state_steps[param_id] = step

        beta1, beta2 = betas
        exp_avg.mul_(beta1).add_(grads, alpha=1 - beta1)

        exp_avg_sq.mul_(beta2).addcmul_(grads, grads, value=1 - beta2)

        bias_correction1 = 1 - beta1 ** step.item()
        bias_correction2 = 1 - beta2 ** step.item()
        exp_avg_corr = exp_avg / bias_correction1
        exp_avg_sq_corr = exp_avg_sq / bias_correction2

        denom = exp_avg_sq_corr.sqrt().add_(eps)
        step_size = lr
        params.sub_(step_size * (exp_avg_corr / denom))

        return params

    return nested_map(f, (params, grads), state, lr, betas, eps)


T = TypeVar("T")


def nested_map(f: Callable[..., T], structure_args: Tuple[Any, ...], *args: Any) -> T:
    if isinstance(structure_args[0], torch.Tensor):
        return f(*structure_args, *args)
    elif isinstance(structure_args[0], dict):
        keys = structure_args[0].keys()
        return {
            k: nested_map(f, tuple(s[k] for s in structure_args), *args) for k in keys
        }
    elif isinstance(structure_args[0], list):
        return [
            nested_map(f, tuple(s[i] for s in structure_args), *args)
            for i in range(len(structure_args[0]))
        ]
    return structure_args[0]
