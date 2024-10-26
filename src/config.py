from typing import Callable, NamedTuple


class BERT_Config(NamedTuple):
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int


BERT_CONFIG = BERT_Config(
    hidden_size=768,
    num_hidden_layers=6,
    num_attention_heads=6,
    intermediate_size=3072,
    vocab_size=30522,
)


class GPT2_Config(NamedTuple):
    num_hidden_layers: int
    num_attention_heads: int


GPT2_CONFIG = GPT2_Config(
    num_hidden_layers=6,
    num_attention_heads=6,
)


class Aggregator_Config(NamedTuple):
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int


AGGREGATOR_CONFIG = Aggregator_Config(
    hidden_size=768,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=4096,
)


class Worldformer_Config(NamedTuple):
    encoder_config: BERT_Config
    decoder_config: GPT2_Config
    aggregator_config: Aggregator_Config


WORLDFORMER_CONFIG = Worldformer_Config(
    encoder_config=BERT_CONFIG,
    decoder_config=GPT2_CONFIG,
    aggregator_config=AGGREGATOR_CONFIG,
)
