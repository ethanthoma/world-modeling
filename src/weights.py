import copy
import pathlib
from typing import Any, TypedDict

import torch
import torch.nn.init as init

import config
import tokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention_Params(TypedDict):
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    output: torch.Tensor


class Feed_Forward_Params(TypedDict):
    intermediate: torch.Tensor
    output: torch.Tensor


class Layer_Params(TypedDict):
    attention: Attention_Params
    feed_forward: Feed_Forward_Params
    ln_1: torch.Tensor
    ln_2: torch.Tensor


class Transformer_Params(TypedDict):
    embeddings: torch.Tensor
    layernorm: torch.Tensor
    layers: list[Layer_Params]


class Aggregator_Params(TypedDict):
    layernorm: torch.Tensor
    layers: list[Layer_Params]


class Worldformer_Params(TypedDict):
    text_encoder: Transformer_Params
    graph_encoder: Transformer_Params
    aggregator: Aggregator_Params
    action_decoder: Transformer_Params
    graph_decoder: Transformer_Params


def load_bert_weights(
    config: config.BERT_Config,
    ckpt_path: pathlib.Path,
    prefix: str = "bert",
    device: torch.device = DEVICE,
) -> Transformer_Params:
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)

    new_tokens = tokenizer.count_new_tokens(tokenizer.BERT_TOKENIZER)

    state_dict[f"{prefix}.embeddings.word_embeddings.weight"] = increase_embed_size(
        state_dict[f"{prefix}.embeddings.word_embeddings.weight"], new_tokens, std=0.02
    )

    layers: list[Layer_Params] = []
    for i in range(config.num_hidden_layers):
        base = f"{prefix}.encoder.layer.{i}"

        qkv_weights = state_dict[f"{base}.attention.self.Wqkv.weight"]
        q, k, v = split_qkv_weights(qkv_weights)

        attention_params: Attention_Params = {
            "query": q,
            "key": k,
            "value": v,
            "output": state_dict[f"{base}.attention.output.dense.weight"],
        }

        feed_forward_params: Feed_Forward_Params = {
            "intermediate": state_dict[f"{base}.mlp.gated_layers.weight"].T,
            "output": state_dict[f"{base}.mlp.wo.weight"].T,
        }

        layers.append(
            {
                "attention": attention_params,
                "feed_forward": feed_forward_params,
                "ln_1": state_dict[f"{base}.attention.output.LayerNorm.weight"],
                "ln_2": state_dict[f"{base}.mlp.layernorm.weight"],
            }
        )

    return {
        "embeddings": state_dict[f"{prefix}.embeddings.word_embeddings.weight"],
        "layernorm": state_dict[f"{prefix}.embeddings.LayerNorm.weight"],
        "layers": layers,
    }


def load_gpt2_weights(
    config: config.GPT2_Config,
    ckpt_path: pathlib.Path,
    device: torch.device = DEVICE,
) -> Transformer_Params:
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)

    new_tokens = tokenizer.count_new_tokens(tokenizer.GPT2_TOKENIZER)

    state_dict["wte.weight"] = increase_embed_size(
        state_dict["wte.weight"], new_tokens, std=0.02
    )

    layers: list[Layer_Params] = []
    for i in range(config.num_hidden_layers):
        qkv_weights = state_dict[f"h.{i}.attn.c_attn.weight"]
        q, k, v = split_qkv_weights(qkv_weights)

        attention_params: Attention_Params = {
            "query": q,
            "key": k,
            "value": v,
            "output": state_dict[f"h.{i}.attn.c_proj.weight"],
        }

        feed_forward_params: Feed_Forward_Params = {
            "intermediate": state_dict[f"h.{i}.mlp.c_fc.weight"],
            "output": state_dict[f"h.{i}.mlp.c_proj.weight"],
        }

        layers.append(
            {
                "attention": attention_params,
                "feed_forward": feed_forward_params,
                "ln_1": state_dict[f"h.{i}.ln_1.weight"],
                "ln_2": state_dict[f"h.{i}.ln_2.weight"],
            }
        )

    return {
        "embeddings": state_dict["wte.weight"],
        "layernorm": state_dict["ln_f.weight"],
        "layers": layers,
    }


def increase_embed_size(
    old_embeddings: torch.Tensor,
    increase_vocab_size_by: int,
    std: float = 0.02,
) -> torch.Tensor:
    """
    Increase embedding matrix size for new tokens while preserving existing embeddings
    """
    new_embeddings = init.normal_(
        torch.empty(
            old_embeddings.shape[0] + increase_vocab_size_by,
            old_embeddings.shape[1],
            dtype=old_embeddings.dtype,
            device=old_embeddings.device,
        ),
        mean=0.0,
        std=std,
    )
    new_embeddings[: old_embeddings.shape[0]] = old_embeddings
    return new_embeddings


def split_qkv_weights(
    attn_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dim1, dim2 = attn_weight.shape
    split_dim = 1 if dim1 < dim2 else 0
    hidden_size = min(attn_weight.shape)
    return tuple(attn_weight.split(hidden_size, dim=split_dim))


def init_aggregator_weights(
    config: config.Aggregator_Config,
    device: torch.device = DEVICE,
) -> Aggregator_Params:
    def init_linear(out_features: int, in_features: int) -> torch.Tensor:
        weight = torch.empty(out_features, in_features, device=device)
        init.normal_(weight, mean=0.0, std=0.02)
        return weight

    layers: list[Layer_Params] = []
    for _ in range(config.num_hidden_layers):
        attention_params: Attention_Params = {
            "query": init_linear(config.hidden_size, config.hidden_size),
            "key": init_linear(config.hidden_size, config.hidden_size),
            "value": init_linear(config.hidden_size, config.hidden_size),
            "output": init_linear(config.hidden_size, config.hidden_size),
        }

        feed_forward_params: Feed_Forward_Params = {
            "intermediate": init_linear(config.hidden_size, config.intermediate_size),
            "output": init_linear(config.intermediate_size, config.hidden_size),
        }

        layers.append(
            {
                "attention": attention_params,
                "feed_forward": feed_forward_params,
                "ln_1": torch.ones(config.hidden_size, device=device),
                "ln_2": torch.ones(config.hidden_size, device=device),
            }
        )

    return {
        "layernorm": torch.ones(config.hidden_size, device=device),
        "layers": layers,
    }


def init_worldformer(
    config: config.Worldformer_Config,
    bert_path: pathlib.Path,
    gpt2_path: pathlib.Path,
    device: torch.device = DEVICE,
) -> Worldformer_Params:
    text_encoder = load_bert_weights(config.encoder_config, bert_path, device=device)
    graph_encoder = copy.deepcopy(text_encoder)

    action_decoder = load_gpt2_weights(config.decoder_config, gpt2_path, device=device)
    graph_decoder = copy.deepcopy(action_decoder)

    return {
        "text_encoder": text_encoder,
        "graph_encoder": graph_encoder,
        "aggregator": init_aggregator_weights(config.aggregator_config, device),
        "action_decoder": action_decoder,
        "graph_decoder": graph_decoder,
    }
