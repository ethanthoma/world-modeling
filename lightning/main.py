import argparse

import lightning as L
from data import JerichoDataModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from model import WorldformerModel

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="subcommand")


def train(hparams):
    print("Training...")
    model = WorldformerModel()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints",
            filename="worldformer-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
            save_last=True,
        ),
    ]

    trainer = L.Trainer(
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=-1,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        precision=32,
        max_time={"days": 4},
        logger=True,
        num_sanity_val_steps=2,
        log_every_n_steps=100,
        sync_batchnorm=True,
        deterministic=True,
    )

    trainer.fit(model=model, train_dataloaders=JerichoDataModule())

    trainer.test(model, datamodule=datamodule)

    trainer.save_checkpoint("worldformer_final.ckpt")


if __name__ == "__main__":
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default=1)

    train_p = subparsers.add_parser("train")
    train_p.set_defaults(func=train)

    args = parser.parse_args()
    if args.subcommand is None:
        parser.print_help()
    else:
        args.func(args)
