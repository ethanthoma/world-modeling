from typing import List

import torch
import torch.nn.functional as F

from config import Transformer_Config
from weights import Transformer_Weights


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.mean(torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1))


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


def training_step(
    config: Transformer_Config,
    weights: Transformer_Weights,
    optimizer: torch.optim.Optimizer,
    input_data: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    optimizer.zero_grad()

    # Forward pass
    logits = transformer(config, weights, input_data)

    # Compute loss
    loss = cross_entropy_loss(logits, targets)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    return loss


def train(
    config: Transformer_Config,
    weights: Transformer_Weights,
    optimizer: torch.optim.Optimizer,
    data_generator: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
    num_steps: int,
):
    for step in range(num_steps):
        input_data, targets = data_generator()
        loss = training_step(config, weights, optimizer, input_data, targets)

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
