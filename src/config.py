from typing import Callable, NamedTuple


class BERT_Config(NamedTuple):
    special_tokens: dict
    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    input_length: int
    hidden_size: int


class GPT2_Config(NamedTuple):
    special_tokens: dict
    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    input_length: int
    hidden_size: int


class Aggregator_Config(NamedTuple):
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    input_length: int
    hidden_size: int


class Worldformer_Config(NamedTuple):
    encoder_config: BERT_Config
    decoder_config: GPT2_Config
    aggregator_config: Aggregator_Config


BERT_SPECIAL_TOKENS = {
    "additional_special_tokens": ["[ACT]", "[TRIPLE]", "[OBS]", "[GRAPH]"],
}

BERT_CONFIG = BERT_Config(
    special_tokens=BERT_SPECIAL_TOKENS,
    vocab_size=30532,
    num_hidden_layers=2,
    num_attention_heads=6,
    intermediate_size=3072,
    input_length=1024,
    hidden_size=768,
)

GPT2_SPECIAL_TOKENS = {
    "additional_special_tokens": ["[ACT]", "[TRIPLE]", "[GRAPH]"],
    "pad_token": "[PAD]",
}

GPT2_CONFIG = GPT2_Config(
    special_tokens=GPT2_SPECIAL_TOKENS,
    vocab_size=50261,
    num_hidden_layers=2,
    num_attention_heads=6,
    intermediate_size=3072,
    hidden_size=768,
    input_length=1024,
)


AGGREGATOR_CONFIG = Aggregator_Config(
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=4096,
    hidden_size=768,
    input_length=2048,
)


WORLDFORMER_CONFIG = Worldformer_Config(
    encoder_config=BERT_CONFIG,
    decoder_config=GPT2_CONFIG,
    aggregator_config=AGGREGATOR_CONFIG,
)
