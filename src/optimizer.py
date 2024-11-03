import dataclasses
import math
from typing import Any

import torch

from util import nested_map


def SGD(params: Any, grads: Any, lr: float) -> Any:
    def f(params: torch.Tensor, grads: torch.Tensor, lr: float):
        return params.sub(lr * grads)

    return nested_map(f, (params, grads), lr)


@dataclasses.dataclass
class Adam_State:
    exp_avgs: dict[int, torch.Tensor] = dataclasses.field(default_factory=dict)
    exp_avg_sqs: dict[int, torch.Tensor] = dataclasses.field(default_factory=dict)
    state_steps: dict[int, torch.Tensor] = dataclasses.field(default_factory=dict)
    seen_params: set = dataclasses.field(default_factory=set)

    def cleanup(self):
        current_params = set(self.seen_params)
        for param_id in list(self.exp_avgs.keys()):
            if param_id not in current_params:
                del self.exp_avgs[param_id]
                del self.exp_avg_sqs[param_id]
                del self.state_steps[param_id]
        self.seen_params.clear()


def Adam(
    params: Any,
    grads: Any,
    opt_state: Adam_State,
    lr: float = 0.001,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> Any:
    def f(
        params: torch.Tensor,
        grads: torch.Tensor,
        opt_state: Adam_State,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        tensor_path_from_route: str,
    ):
        param_id = tensor_path_from_route
        opt_state.seen_params.add(param_id)

        if param_id not in opt_state.exp_avgs:
            opt_state.exp_avgs[param_id] = torch.zeros_like(
                params, device=params.device
            )
            opt_state.exp_avg_sqs[param_id] = torch.zeros_like(
                params, device=params.device
            )
            opt_state.state_steps[param_id] = torch.tensor(0.0, device=params.device)

        exp_avg = opt_state.exp_avgs[param_id]
        exp_avg_sq = opt_state.exp_avg_sqs[param_id]
        step = opt_state.state_steps[param_id]

        step = step + 1
        opt_state.state_steps[param_id] = step

        beta1, beta2 = betas

        exp_avg = beta1 * exp_avg + (1 - beta1) * grads
        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grads * grads)

        opt_state.exp_avgs[param_id] = exp_avg
        opt_state.exp_avg_sqs[param_id] = exp_avg_sq

        step_val = step.item()
        bias_correction1 = 1 - beta1**step_val
        bias_correction2 = 1 - beta2**step_val

        exp_avg_corrected = exp_avg / bias_correction1
        exp_avg_sq_corrected = exp_avg_sq / bias_correction2

        denom = torch.sqrt(exp_avg_sq_corrected) + eps
        update = exp_avg_corrected / denom

        new_params = params - lr * update

        return new_params

    result = nested_map(f, (params, grads), opt_state, lr, betas, eps)

    opt_state.cleanup()

    return result
