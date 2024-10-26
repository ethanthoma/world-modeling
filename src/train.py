import functools
import itertools
import pathlib
import time
from collections.abc import Callable
from typing import Any, Dict, Iterable, List, Tuple, Union

import torch
import torch.func as func
import torch.jit as jit
import torch.nn.functional as F

import config
import data
import model
import preprocess
import spinner
import tokenizer
import weights

NUM_EPOCHS = 5
BATCH_SIZE = 1
BERT_PATH = pathlib.Path("./weights/bert.bin")
GPT2_PATH = pathlib.Path("./weights/gpt2.bin")
TRAIN_DATA_PATH = pathlib.Path("./data/jericho-world/train.json")
LR = 3e-4
PAD_TOKEN_ID = tokenizer.GPT2_TOKENIZER.eos_token_id


def train(num_epochs: int = NUM_EPOCHS) -> None:
    # ** Dataset **
    raw_data = data.data(TRAIN_DATA_PATH)

    preprocessed_data = map(preprocess.preprocess, raw_data)

    batched_data = itertools.starmap(
        zip, itertools.batched(preprocessed_data, BATCH_SIZE)
    )

    # ** Model **
    params = weights.init_worldformer(config.WORLDFORMER_CONFIG, BERT_PATH, GPT2_PATH)

    batched_predict = func.vmap(
        model.worldformer,
        in_dims=(None, *([0] * 8)),
        randomness="different",
    )

    # ** Training Loop **
    for epoch in range(num_epochs):
        start_time = time.time()
        step = None

        total_loss = 0

        with spinner.spinner() as w:
            w.write(f"Epoch: {epoch + 1}/{num_epochs} | Step: 0")

            for step, (input, target) in enumerate(batched_data):
                params, loss = update(params, batched_predict, input, target)

                total_loss += loss

                w.write(
                    f"Epoch: {epoch + 1}/{num_epochs} | Step: {step} | Loss: {loss}"
                )

                del loss

        epoch_time = time.time() - start_time
        avg_loss = total_loss / steps

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Total Steps: {steps} | Avg. Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s"
        )


def update(
    params: weights.Worldformer_Params,
    predict: Callable[[Any, Any], torch.Tensor],
    input: preprocess.Input,
    target: preprocess.Target,
) -> Tuple[weights.Worldformer_Params, torch.Tensor]:
    grads, loss_value = func.grad_and_value(loss, argnums=0)(
        params, predict, input, target
    )

    def update_params(p: dict, g: dict) -> dict:
        updated = {}
        for k, v in p.items():
            if isinstance(v, torch.Tensor):
                updated[k] = update_tensor_with_grads(v, g[k], LR)
                del g[k]
            elif isinstance(v, dict):
                updated[k] = update_params(v, g[k])
            elif isinstance(v, list):
                updated[k] = [
                    update_params(layer, g_layer) for layer, g_layer in zip(v, g[k])
                ]
            else:
                updated[k] = v
        return updated

    updated_params = update_params(params, grads)

    del grads

    return updated_params, loss_value.item()


@jit.script
def update_tensor_with_grads(
    params: torch.Tensor, grads: torch.Tensor, lr: float
) -> torch.Tensor:
    return params.sub_(lr * grads)


def loss(
    params: weights.Worldformer_Params,
    predict: Callable[[Any, Any], torch.Tensor],
    input: preprocess.Input,
    target: preprocess.Target,
) -> torch.Tensor:
    tokens_input = tokenizer.tokenize_input(input)
    tokens_target = tokenizer.tokenize_target(target)

    action_logits, graph_logits = predict(params, *tokens_input, *tokens_target)

    action_targets, graph_targets, action_boundaries, graph_boundaries = tokens_target

    action_logits = action_logits[..., :-1, :]
    graph_logits = graph_logits[..., :-1, :]

    action_targets = action_targets[..., 1:]
    graph_targets = graph_targets[..., 1:]
    action_boundaries = action_boundaries[..., 1:]
    graph_boundaries = graph_boundaries[..., 1:]

    action_loss = compute_sos_loss(
        action_logits,
        action_targets,
        action_boundaries,
        PAD_TOKEN_ID,
    )
    graph_loss = compute_sos_loss(
        graph_logits,
        graph_targets,
        graph_boundaries,
        PAD_TOKEN_ID,
    )

    total_loss = action_loss + graph_loss

    return total_loss.squeeze()


@jit.script
def compute_sos_loss(
    logits: torch.Tensor,  # [batch_size, seq_len, vocab_size]
    targets: torch.Tensor,  # [batch_size, seq_len]
    boundaries: torch.Tensor,  # [batch_size, seq_len]
    pad_token_id: int,
) -> torch.Tensor:
    batch_size = logits.size(0)
    seq_len = targets.size(1)
    device = logits.device

    attention_mask = torch.ne(targets, pad_token_id)
    attention_mask = attention_mask.to(dtype=torch.float32)

    total_loss = torch.zeros(1, device=device)
    total_sequences = 0

    for b in range(batch_size):
        # Get indices where boundaries == 1
        seq_starts = boundaries[b].nonzero().squeeze(-1)

        if seq_starts.size(0) == 0:
            continue

        # Process each sequence
        last_start_idx = -1
        for i in range(seq_starts.size(0)):
            start_idx = int(seq_starts[i].item())

            # If this is the last sequence, end at seq_len
            # Otherwise, end at the next start index
            if i < seq_starts.size(0) - 1:
                end_idx = int(seq_starts[i + 1].item())
            else:
                end_idx = seq_len

            # Skip invalid sequences
            if start_idx >= end_idx or start_idx <= last_start_idx:
                continue

            last_start_idx = start_idx

            seq_logits = logits[b, start_idx:end_idx]
            seq_targets = targets[b, start_idx:end_idx]
            seq_mask = attention_mask[b, start_idx:end_idx]

            if seq_mask.sum() == 0:
                continue

            seq_loss = F.cross_entropy(
                seq_logits.reshape(-1, seq_logits.size(-1)),
                seq_targets.reshape(-1),
                reduction="none",
                ignore_index=pad_token_id,
            )

            valid_tokens = seq_mask.sum()
            if valid_tokens > 0:
                seq_loss = (seq_loss * seq_mask).sum() / valid_tokens
                total_loss += seq_loss
                total_sequences += 1

    return total_loss / max(total_sequences, 1)
