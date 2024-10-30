import torch
import transformers

import preprocess

BERT_SPECIAL_TOKENS = {
    "additional_special_tokens": ["[ACT]", "[TRIPLE]", "[OBS]", "[GRAPH]"],
    "pad_token": "[PAD]",
    "sep_token": "[SEP]",
    "cls_token": "[CLS]",
}

GPT2_SPECIAL_TOKENS = {
    "additional_special_tokens": ["[ACT]", "[TRIPLE]", "[GRAPH]"],
    "pad_token": "[PAD]",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_bert_tokenizer() -> transformers.BertTokenizer:
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens(BERT_SPECIAL_TOKENS)
    return tokenizer


def get_gpt2_tokenizer() -> transformers.GPT2Tokenizer:
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    # GPT2 doesn't have pad_token by default
    tokenizer.add_special_tokens(GPT2_SPECIAL_TOKENS)
    return tokenizer


BERT_TOKENIZER = get_bert_tokenizer()
GPT2_TOKENIZER = get_gpt2_tokenizer()


def tokenize_input(
    input: tuple[preprocess.Input, ...],
    input_length: int = 1024,
    bert_tokenizer: transformers.BertTokenizer = BERT_TOKENIZER,
    device: torch.device = DEVICE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenizes input text and graph with BERT tokenizer"""
    textual_input, graph_input = zip(*input)

    textual_encoding = bert_tokenizer(
        textual_input,
        max_length=input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    graph_encoding = bert_tokenizer(
        graph_input,
        max_length=input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    _, textual_valid_mask = create_sequence_boundaries(
        textual_encoding.input_ids,
        bert_tokenizer.convert_tokens_to_ids("[ACT]"),
        device=device,
    )

    _, graph_valid_mask = create_sequence_boundaries(
        graph_encoding.input_ids,
        bert_tokenizer.convert_tokens_to_ids("[TRIPLE]"),
        device=device,
    )

    return (
        textual_encoding.input_ids.to(device),
        graph_encoding.input_ids.to(device),
        textual_valid_mask,
        graph_valid_mask,
    )


def tokenize_target(
    target: tuple[preprocess.Target, ...],
    input_length: int = 1024,
    gpt2_tokenizer: transformers.GPT2Tokenizer = GPT2_TOKENIZER,
    device: torch.device = DEVICE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenizes target actions and graph differences with GPT2 tokenizer"""
    action_target, graph_target = zip(*target)

    action_encoding = gpt2_tokenizer(
        action_target,
        max_length=input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    graph_encoding = gpt2_tokenizer(
        graph_target,
        max_length=input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    action_boundaries, action_valid_mask = create_sequence_boundaries(
        action_encoding.input_ids,
        gpt2_tokenizer.convert_tokens_to_ids("[ACT]"),
        device=device,
    )

    graph_boundaries, graph_valid_mask = create_sequence_boundaries(
        graph_encoding.input_ids,
        gpt2_tokenizer.convert_tokens_to_ids("[TRIPLE]"),
        device=device,
    )

    return (
        action_encoding.input_ids.to(device),
        graph_encoding.input_ids.to(device),
        action_boundaries * action_valid_mask,
        graph_boundaries * graph_valid_mask,
    )


def create_sequence_boundaries(
    tokens: torch.Tensor,
    separator_token_id: int,
    bos_token_id: int = None,
    eos_token_id: int = None,
    device: torch.device = DEVICE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates sequence boundaries and valid sequence mask for Set of Sequences
    Returns both sequence start positions and valid sequence mask
    """
    batch_size, seq_length = tokens.shape
    boundaries = torch.zeros_like(tokens, dtype=torch.float32)
    valid_mask = torch.ones_like(tokens, dtype=torch.float32)

    if bos_token_id is not None:
        boundaries = (tokens == bos_token_id).float()
    else:
        boundaries[:, 0] = 1
        is_separator = tokens == separator_token_id
        boundaries[:, 1:] = is_separator[:, :-1].float()

    if eos_token_id is not None:
        eos_positions = (tokens == eos_token_id).float().cumsum(dim=1)
        valid_mask = (eos_positions == 0).float()

    valid_mask *= (tokens != BERT_TOKENIZER.pad_token_id).float()
    valid_mask *= (tokens != GPT2_TOKENIZER.pad_token_id).float()

    return boundaries.to(device), valid_mask.to(device)


def count_new_tokens(tokenizer: transformers.PreTrainedTokenizer) -> int:
    """Calculate number of new tokens added to the tokenizer"""
    match tokenizer.__class__.__name__:
        case "GPT2Tokenizer":
            base = transformers.GPT2Tokenizer.from_pretrained("gpt2")
            new_tokens = len(set(tokenizer.additional_special_tokens))
            if tokenizer.pad_token and not base.pad_token:
                new_tokens += 1
            if tokenizer.bos_token and not base.bos_token:
                new_tokens += 1
            if tokenizer.eos_token and not base.eos_token:
                new_tokens += 1
            return new_tokens
        case "BertTokenizer":
            base = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
            return len(set(tokenizer.additional_special_tokens))
        case _:
            assert False
