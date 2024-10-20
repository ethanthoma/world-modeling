import torch
import torch.func as func
import torch.jit as jrt

from config import Config, transformer_config_from_config
from model import transformer
from weights import init_transformer_weights

default_config = Config(
    dim=768,
    head_n=12,
    layers_n=12,
    vocab_size=30_522,
)


def main():
    config = transformer_config_from_config(default_config)
    weights = init_transformer_weights(config)

    batch_size = 32
    seq_len = 512

    x = torch.ones(
        batch_size,
        seq_len,
        config.dim,
    )

    batched_transformer = func.vmap(transformer, in_dims=(None, 0))

    y = batched_transformer(weights, x)
    assert y.shape == (batch_size, seq_len, config.vocab_size)

    print(y.shape)
