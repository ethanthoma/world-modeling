import torch
import transformers

import config
import preprocess

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_bert_tokenizer(config: config.BERT_Config) -> transformers.BertTokenizer:
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens(config.special_tokens)
    return tokenizer


def get_gpt2_tokenizer(config: config.GPT2_Config) -> transformers.GPT2Tokenizer:
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens(config.special_tokens)
    return tokenizer


def tokenize_input(
    config: config.BERT_Config,
    input: tuple[preprocess.Input, ...],
    device: torch.device = DEVICE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenizes input text and graph with BERT tokenizer"""
    bert_tokenizer = get_bert_tokenizer(config)

    textual_input, graph_input = zip(*input)

    textual_encoding = bert_tokenizer(
        textual_input,
        max_length=config.input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    graph_encoding = bert_tokenizer(
        graph_input,
        max_length=config.input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    _, textual_valid_mask = create_sequence_boundaries(
        textual_encoding.input_ids,
        bert_tokenizer.convert_tokens_to_ids("[ACT]"),
        bert_tokenizer.pad_token_id,
        device=device,
    )

    _, graph_valid_mask = create_sequence_boundaries(
        graph_encoding.input_ids,
        bert_tokenizer.convert_tokens_to_ids("[TRIPLE]"),
        bert_tokenizer.pad_token_id,
        device=device,
    )

    return (
        textual_encoding.input_ids.to(device),
        graph_encoding.input_ids.to(device),
        textual_valid_mask,
        graph_valid_mask,
    )


def tokenize_target(
    config: config.GPT2_Config,
    target: tuple[preprocess.Target, ...],
    device: torch.device = DEVICE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenizes target actions and graph differences with GPT2 tokenizer"""
    gpt2_tokenizer = get_gpt2_tokenizer(config)

    action_target, graph_target = zip(*target)

    action_encoding = gpt2_tokenizer(
        action_target,
        max_length=config.input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    graph_encoding = gpt2_tokenizer(
        graph_target,
        max_length=config.input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    action_boundaries, action_valid_mask = create_sequence_boundaries(
        action_encoding.input_ids,
        gpt2_tokenizer.convert_tokens_to_ids("[ACT]"),
        gpt2_tokenizer.pad_token_id,
        device=device,
    )

    graph_boundaries, graph_valid_mask = create_sequence_boundaries(
        graph_encoding.input_ids,
        gpt2_tokenizer.convert_tokens_to_ids("[TRIPLE]"),
        gpt2_tokenizer.pad_token_id,
        device=device,
    )

    return (
        action_encoding.input_ids.to(device),
        graph_encoding.input_ids.to(device),
        action_boundaries * action_valid_mask,
        graph_boundaries * graph_valid_mask,
    )


def gpt2_pad_token_id(config: config.GPT2_Config) -> int:
    return get_gpt2_tokenizer(config).pad_token_id


def create_sequence_boundaries(
    tokens: torch.Tensor,
    separator_token_id: int,
    pad_token_id: int,
    device: torch.device = DEVICE,
) -> tuple[torch.Tensor, torch.Tensor]:
    is_separator = (tokens == separator_token_id).float()
    is_padding = (tokens == pad_token_id).float()

    boundaries = torch.zeros_like(tokens, dtype=torch.float32)
    boundaries[:, 0] = 1.0
    boundaries[:, 1:] = is_separator[:, :-1]

    valid_mask = 1.0 - (is_padding + is_separator)

    return boundaries.to(device), valid_mask.to(device)
