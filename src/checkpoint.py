import pathlib
import time
from typing import NamedTuple, Optional

import torch

import optimizer
import weights


class Checkpoint(NamedTuple):
    params: weights.Worldformer_Params
    optimizer_state: optimizer.Adam_State
    epoch: int


torch.serialization.add_safe_globals([Checkpoint])
torch.serialization.add_safe_globals([weights.Worldformer_Params])
torch.serialization.add_safe_globals([optimizer.Adam_State])


def save_checkpoint(
    checkpoint_dir: pathlib.Path,
    params: weights.Worldformer_Params,
    optimizer_state: optimizer.Adam_State,
    epoch: int,
    keep_last_n: int = 3,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Checkpoint(params=params, optimizer_state=optimizer_state, epoch=epoch)

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)

    if keep_last_n > 0:
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        for old_checkpoint in checkpoints[keep_last_n:]:
            old_checkpoint.unlink()


def load_latest_checkpoint(
    checkpoint_dir: pathlib.Path,
) -> Optional[Checkpoint]:
    if not checkpoint_dir.exists():
        return None

    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_*.pt"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    if not checkpoints:
        return None

    return torch.load(checkpoints[0], weights_only=True)


def resume_from_checkpoint(
    checkpoint_dir: pathlib.Path,
    default_params: weights.Worldformer_Params,
    default_optimizer_state: optimizer.Adam_State,
) -> tuple[weights.Worldformer_Params, optimizer.Adam_State, int]:
    checkpoint = load_latest_checkpoint(checkpoint_dir)

    if checkpoint is None:
        return default_params, default_optimizer_state, 0

    return (
        checkpoint.params,
        checkpoint.optimizer_state,
        checkpoint.epoch,
    )
