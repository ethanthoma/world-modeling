import functools
import itertools
import logging
import pathlib
import time

import torch
import torch.func as func
import torch.jit as jit
import torch.nn.functional as F

import checkpoint
import config
import data
import model
import optimizer
import preprocess
import spinner
import tokenizer
import weights
from util import parameter_count

NUM_EPOCHS = 5
BATCH_SIZE = 16
BERT_PATH = pathlib.Path("./weights/bert.bin")
GPT2_PATH = pathlib.Path("./weights/gpt2.bin")
TRAIN_DATA_PATH = pathlib.Path("./data/jericho-world/train.json")
LR = 3e-4
PAD_TOKEN_ID = tokenizer.GPT2_TOKENIZER.pad_token_id
CHECKPOINT_DIR = pathlib.Path("./checkpoints")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(num_epochs: int = NUM_EPOCHS) -> None:
    # ** Dataset **
    logger.info("Creating dataset pipeline...")
    raw_data = data.data(TRAIN_DATA_PATH)

    preprocessed_data = map(preprocess.preprocess, raw_data)

    batched_data = itertools.starmap(
        zip, itertools.batched(preprocessed_data, BATCH_SIZE)
    )

    # ** Model **
    logger.info("Initializing model...")
    initial_params = weights.init_worldformer(
        config.WORLDFORMER_CONFIG, BERT_PATH, GPT2_PATH
    )
    initial_optimizer_state = optimizer.Adam_State()

    batched_predict = func.vmap(
        model.worldformer,
        in_dims=(None, *([0] * 8)),
        randomness="different",
    )

    # ** Checkpoint **
    logger.info("Loading latest checkpoint...")
    params, adam_state, start_epoch = checkpoint.resume_from_checkpoint(
        CHECKPOINT_DIR, initial_params, initial_optimizer_state
    )
    logger.info(f"Model has {parameter_count(params):_} parameters")

    # ** Loss **
    batched_loss_fn = lambda *args: torch.mean(
        func.vmap(loss_fn, in_dims=(None, *([0] * 6), None))(*args)
    )

    # ** Optimizer **
    opt = functools.partial(optimizer.Adam, state=adam_state)

    # ** Training Loop **
    logger.info("Starting training...")
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        step = None

        total_loss = 0

        with spinner.spinner() as w:
            w.write(f"Epoch: {epoch + 1}/{num_epochs} | Step: 0")

            for step, (input, target) in enumerate(batched_data):
                tokens_input = tokenizer.tokenize_input(input)
                tokens_target = tokenizer.tokenize_target(target)

                logits = batched_predict(params, *tokens_input, *tokens_target)

                grads, loss = func.grad_and_value(batched_loss_fn, argnums=0)(
                    params, *tokens_target, *logits, PAD_TOKEN_ID
                )

                params = opt(params=params, grads=grads, lr=LR)

                # ** Stats **
                elapsed_time = time.time() - start_time
                avg_step_time = elapsed_time / (step + 1)
                total_loss += loss

                w.write(
                    f"Epoch: {epoch + 1}/{num_epochs} | Step: {step + 1} | Avg. Step Time {avg_step_time:.2f}s | Loss: {loss:.4f}"
                )

        epoch_time = time.time() - start_time
        avg_loss = total_loss / (step + 1)

        checkpoint.save_checkpoint(CHECKPOINT_DIR, params, adam_state, epoch + 1)

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} | Total Steps: {step + 1} | Time: {epoch_time:.2f}s | Avg. Loss: {avg_loss:.4f}"
        )


@jit.script
def compute_sos_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    boundaries: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    shifted_logits = logits[:-1, :].contiguous()
    shifted_targets = target_ids[1:].contiguous()
    shifted_boundaries = boundaries[1:].contiguous()

    special_token_mask = shifted_targets != pad_token_id

    vocab_size = shifted_logits.size(-1)
    token_losses = F.cross_entropy(
        shifted_logits.view(-1, vocab_size),
        shifted_targets.view(-1),
        reduction="none",
        ignore_index=pad_token_id,
    ).view(shifted_targets.size())

    masked_losses = token_losses * special_token_mask.float()

    sequence_mask = shifted_boundaries > 0
    sequence_lengths = sequence_mask.sum(dim=0, keepdim=True).clamp(min=1)
    sequence_loss = masked_losses.sum(dim=0, keepdim=True) / sequence_lengths

    return sequence_loss.squeeze()


@jit.script
def loss_fn(
    params: weights.Worldformer_Params,
    action_targets: torch.Tensor,
    graph_targets: torch.Tensor,
    action_boundaries: torch.Tensor,
    graph_boundaries: torch.Tensor,
    action_logits: torch.Tensor,
    graph_logits: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    action_loss = compute_sos_loss(
        action_logits,
        action_targets,
        action_boundaries,
        pad_token_id,
    )

    graph_loss = compute_sos_loss(
        graph_logits,
        graph_targets,
        graph_boundaries,
        pad_token_id,
    )

    return action_loss + graph_loss
