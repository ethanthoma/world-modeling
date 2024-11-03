import pathlib

import torch

import config
import tokenizer
import train


def main():
    train_config = train.Train_Config(
        num_epochs=5,
        batch_size=1,
        learning_rate=3e-4,
        grad_clip_max=1.0,
        bert_config=config.BERT_CONFIG,
        gpt2_config=config.GPT2_CONFIG,
        worldformer_config=config.WORLDFORMER_CONFIG,
        pad_token_id=tokenizer.gpt2_pad_token_id(config.GPT2_CONFIG),
        bert_path=pathlib.Path("./weights/bert.bin"),
        gpt2_path=pathlib.Path("./weights/gpt2.bin"),
        data_path=pathlib.Path("./data/jericho-world/train.json"),
        checkpoint_dir=pathlib.Path("./checkpoints"),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    train.train(train_config)
