from dataclasses import dataclass, field
from typing import Any

import torch

from util import nested_map


def SGD(params: Any, grads: Any, lr: float) -> Any:
    def f(params: torch.Tensor, grads: torch.Tensor, lr: float):
        return params.sub_(lr * grads)

    return nested_map(f, (params, grads), lr)


@dataclass
class Adam_State:
    exp_avgs: dict[int, torch.Tensor] = field(default_factory=dict)
    exp_avg_sqs: dict[int, torch.Tensor] = field(default_factory=dict)
    state_steps: dict[int, torch.Tensor] = field(default_factory=dict)


def Adam(
    params: Any,
    grads: Any,
    state: Adam_State,
    lr: float = 0.001,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> Any:
    def f(
        params: torch.Tensor,
        grads: torch.Tensor,
        state: Adam_State,
        lr: float,
        beta: tuple[float, float],
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
