import dataclasses
import functools
import gc
import itertools
import pathlib
import time
from typing import Any, Callable, Optional, TypedDict

import torch
import torch.func as func
import torch.nn.functional as F

import config
import input_pipeline
import model
import optimizer
import spinner
import util
import weights


class Train_Config(TypedDict):
    batch_size: int
    learning_rate: float
    grad_clip_max: float
    bert_config: config.BERT_Config
    gpt2_config: config.GPT2_Config
    worldformer_config: config.Worldformer_Config
    pad_token_id: int
    bert_path: pathlib.Path
    gpt2_path: pathlib.Path
    data_path: pathlib.Path
    checkpoint_dir: pathlib.Path
    device: str


@dataclasses.dataclass
class Early_Stopping:
    best_loss: Optional[float] = None
    time_at_best_loss: float = time.time()
    epochs_since_best_loss: int = 0
    epoch_limit: int = 5
    time_limit: float = 5 * 24 * 60 * 60
    min_delta: float = 0.01
    early_stop: bool = False

    def __call__(self, val_loss: Optional[float] = None) -> bool:
        def check_time() -> bool:
            return time.time() - self.time_at_best_loss >= self.time_limit

        if self.best_loss is None or val_loss + self.min_delta < self.best_loss:
            print("best loss is none or worse")
            self.best_loss = val_loss
            self.epochs_since_best_loss = 0
            self.time_at_best_loss = time.time()
        elif val_loss is None:
            print("val loss is none")
            self.early_stop |= check_time()
        else:
            print("best loss is better")
            self.epochs_since_best_loss += 1

            if self.epochs_since_best_loss >= self.epoch_limit or check_time():
                self.early_stop |= True

        return self.early_stop


@dataclasses.dataclass
class Train_State:
    apply_fn: Callable[
        [weights.Worldformer_Params, torch.Tensor, ...],
        tuple[torch.Tensor, torch.Tensor],
    ]
    params: weights.Worldformer_Params
    opt: Callable[
        [weights.Worldformer_Params, Any, optimizer.Adam_State, ...],
        weights.Worldformer_Params,
    ]
    opt_state: optimizer.Adam_State
    step: int = 0

    def apply_gradients(self, grads: Any):
        return dataclasses.replace(
            self,
            params=self.opt(params=self.params, grads=grads, opt_state=self.opt_state),
            step=self.step + 1,
        )

    def save_checkpoint(self, checkpoint_dir: pathlib.Path, epoch: int):
        torch.save(
            {
                "params": self.params,
                "opt_state": self.opt_state,
                "step": self.step,
                "epoch": epoch,
            },
            checkpoint_dir / f"checkpoint_epoch_{epoch}_{step}.pt",
        )


def loss_fn(
    params: weights.Worldformer_Params,
    x: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    y: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    apply_fn: Callable[
        [weights.Worldformer_Params, torch.Tensor, ...],
        tuple[torch.Tensor, torch.Tensor],
    ],
    pad_token_id: int,
    device: torch.device,
):
    with torch.autocast(device_type=device, dtype=torch.float32):
        action_logits, graph_logits = apply_fn(params, *x, *y)

        (
            action_targets,
            graph_targets,
            action_boundaries,
            graph_boundaries,
        ) = y

        batched_sos_loss = func.vmap(sos_loss, in_dims=(*([0] * 3), None))

        action_loss = batched_sos_loss(
            action_logits,
            action_targets,
            action_boundaries,
            pad_token_id,
        )

        graph_loss = batched_sos_loss(
            graph_logits,
            graph_targets,
            graph_boundaries,
            pad_token_id,
        )

        return torch.mean(action_loss + graph_loss)


def train(train_config: Train_Config) -> None:
    print("Training...")

    # ** Model **
    params = weights.init_worldformer(
        train_config["worldformer_config"],
        train_config["bert_path"],
        train_config["gpt2_path"],
    )
    params = util.nested_map(
        lambda t: t.requires_grad_() if t.is_floating_point else t, params
    )

    print(f"Model has {util.parameter_count(params):_} parameters")

    apply_fn = func.vmap(
        model.worldformer,
        in_dims=(None, *([0] * 8)),
        randomness="different",
    )

    # ** Optimizer **
    opt_state = optimizer.Adam_State()
    opt = functools.partial(optimizer.Adam, lr=train_config["learning_rate"])

    # ** Training Loop **
    state = Train_State(
        apply_fn=apply_fn,
        params=params,
        opt=opt,
        opt_state=opt_state,
    )
    early_stopping = Early_Stopping()

    # ** Training Helpers **
    time_fmt = lambda t: time.strftime("%H:%M:%S", time.gmtime(t))
    scaler = torch.amp.GradScaler(train_config["device"])
    grad_clip_norm_ = functools.partial(
        lambda grad, scaler: util.nested_map(
            lambda t: (t * (1 / scaler.get_scale())).clamp_(max=1.0)
            * scaler.get_scale(),
            grad,
        ),
        scaler=scaler,
    )

    scaled_loss_fn = lambda *args: scaler.scale(loss_fn(*args))

    # ** Dataset **
    train_set, valid_set = input_pipeline.input_pipeline(
        data_path=train_config["data_path"],
        batch_size=train_config["batch_size"],
        bert_config=train_config["bert_config"],
        gpt2_config=train_config["gpt2_config"],
    )

    print("Starting training loop...")
    for epoch in itertools.count():
        start_time, state.step, total_loss = time.time(), 0, 0

        with spinner.spinner():
            print(f"Epoch: {epoch + 1} | Step: 1...")

            for x, y in train_set():
                step_time = time.time()

                grads, loss = func.grad_and_value(scaled_loss_fn, argnums=(0))(
                    state.params,
                    x,
                    y,
                    state.apply_fn,
                    train_config["pad_token_id"],
                    train_config["device"],
                )

                total_loss += loss * (1 / scaler.get_scale())

                grad_clip_norm_(grads)

                state = state.apply_gradients(grads=grads)

                del grads, loss
                gc.collect()

                print(
                    f"Epoch: {epoch + 1} | "
                    f"Step: {state.step} | "
                    f"Time: [{time_fmt(time.time() - step_time)}:{time_fmt(time.time() - start_time)}] | "
                    f"Avg. Loss: [{(total_loss / state.step):.4f}]"
                )

                if early_stopping():
                    print("Timed out.  Stopping...")
                    state.save_checkpoint(train_config["checkpoint_dir"], epoch + 1)
                    print(f"Stored checkpoint to {train_config["checkpoint_dir"]}")
                    return

        with spinner.spinner():
            print(f"Epoch: {epoch + 1} | Validating...")

            val_loss = validate(train_config, state, valid_set)

        print(
            f"Epoch {epoch + 1} | "
            f"Step: {state.step} | "
            f"Time: [{time_fmt(time.time() - start_time)}] | "
            f"Avg. Loss: {(total_loss / (state.step)):.4f} | "
            f"Validation Score {val_loss:.4f}"
        )

        state.save_checkpoint(train_config["checkpoint_dir"], epoch + 1)
        print(f"Stored checkpoint to {train_config["checkpoint_dir"]}")

        if early_stopping(val_loss):
            print("Validation loss or timed out.  Stopping...")
            return


def sos_loss(
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


def validate(train_config: Train_Config, state: Train_State, valid_set):
    total_loss = 0
    total_steps = 0

    for x, y in valid_set():
        loss = loss_fn(
            state.params,
            x,
            y,
            state.apply_fn,
            train_config["pad_token_id"],
            train_config["device"],
        )
        total_loss += loss
        total_steps += 1

    return total_loss / total_steps
