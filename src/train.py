import sys
import time
from typing import Callable, Iterable, List, Tuple

import torch
import torch.func as func
import torch.nn.functional as F

from config import Transformer_Config
from weights import Transformer_Weights


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def loss(
    params: Any,
    x: Any,
    targets: torch.Tensor,
    predict: Callable[[Any, Any], torch.Tensor],
):
    logits = func.vmap(predict, in_dims=(None, 0))(params, x)
    return cross_entropy(logits, targets)


@torch.jit.script
def update(
    params: Any,
    x: Any,
    y: torch.Tensor,
    predict: Callable[[Any, Any], torch.Tensor],
    step_size: float = 0.01,
) -> List[Any]:
    grads = func.grad(loss)(params, x, y, predict)
    return [
        (w - step_size * dw, b - step_size * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]


def train(
    params: Any, training_generator: Iterable[Tuple[Any, Any]], num_epochs: int = 5
):
    for epoch in range(num_epochs):
        start_time = time.time()
        for x, y in training_generator:
            params = update(params, x, y)
        epoch_time = time.time() - start_time

        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))


def get_transformer_parameters(weights: Transformer_Weights) -> List[torch.Tensor]:
    params = []
    for layer in weights.layers:
        params.extend(
            [
                layer.rms_norm_weights_attention.weight,
                layer.attention_weights.q,
                layer.attention_weights.k,
                layer.attention_weights.v,
                layer.attention_weights.out,
                layer.rms_norm_weights_ffn.weight,
                layer.feed_forward_weights.weight_1,
                layer.feed_forward_weights.weight_2,
                layer.feed_forward_weights.weight_3,
            ]
        )
    params.extend([weights.norm.weight, weights.weight])
    return params


def train(
    params: Any,
    optimizer: torch.optim.Optimizer,
    data_generator: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
    num_epochs: int,
    model: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
):
    def spinner_generator():
        while True:
            for char in "|/-\\":
                yield char

    spinner = spinner_generator()

    for epoch in range(num_epochs):
        start_time = time.time()
        step = None

        for step, (inputs, targets) in enumerate(data_generator()):
            optimizer.zero_grad()

            loss = loss(params, inputs, targets, transformer)

            loss.backward()

            optimizer.step()

            sys.stdout.write(
                f"\rEpoch {epoch + 1}/{num_epochs} - Step {step} {next(spinner)}"
            )
            sys.stdout.flush()

        epoch_time = time.time() - start_time
        avg_loss = total_loss / steps

        sys.stdout.write("\r" + " " * (80) + "\r")
        sys.stdout.flush()

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Steps: {steps}, Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s"
        )
