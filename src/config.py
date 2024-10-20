from typing import Callable, NamedTuple

import torch
import torch.nn.init as init


class Config(NamedTuple):
    dim: int
    head_n: int
    layers_n: int
    vocab_size: int


class Attention_Config(NamedTuple):
    head_n: int
    head_dim: int
    dim: int


class Feed_Forward_Config(NamedTuple):
    dim: int
    hidden_dim: int


class RMS_Norm_Config(NamedTuple):
    dim: int


class Layer_Config(NamedTuple):
    rms_norm_config_attention: RMS_Norm_Config
    attention_config: Attention_Config
    rms_norm_config_ffn: RMS_Norm_Config
    feed_forward_config: Feed_Forward_Config


class Transformer_Config(NamedTuple):
    dim: int
    head_n: int
    layers_n: int
    vocab_size: int
    layer_config: Layer_Config
    norm: RMS_Norm_Config


def transformer_config_from_config(config: Config) -> Transformer_Config:
    return Transformer_Config(
        dim=config.dim,
        head_n=config.head_n,
        layers_n=config.layers_n,
        vocab_size=config.vocab_size,
        layer_config=Layer_Config(
            rms_norm_config_attention=RMS_Norm_Config(
                dim=config.dim,
            ),
            attention_config=Attention_Config(
                dim=config.dim,
                head_n=config.head_n,
                head_dim=config.dim // config.head_n,
            ),
            rms_norm_config_ffn=RMS_Norm_Config(
                dim=config.dim,
            ),
            feed_forward_config=Feed_Forward_Config(
                dim=config.dim,
                hidden_dim=3 * config.dim,
            ),
        ),
        norm=RMS_Norm_Config(
            dim=config.dim,
        ),
    )
