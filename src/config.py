from typing import Callable, NamedTuple

import torch
import torch.nn.init as init


class BERT_Config(NamedTuple):
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    type_vocab_size: int
    vocab_size: int


BERT_BASE_CONFIG = BERT_Config(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    type_vocab_size=2,
    vocab_size=30522,
)
