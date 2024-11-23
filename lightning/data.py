import functools
import itertools
import pathlib
from typing import Any, Iterator, Optional, TypedDict

import torch
import torch.utils.data
from stream import JerichoSample, StateDict, stream

import lightning.pytorch as L
from tokenizer import create_tokenizers


class Sample(TypedDict):
    """A sample from the Jericho dataset with encoder and decoder inputs/outputs."""

    # Encoder inputs
    text_input_ids: torch.Tensor
    text_attention_mask: torch.Tensor
    graph_input_ids: torch.Tensor
    graph_attention_mask: torch.Tensor

    # Decoder inputs/outputs
    action_input_ids: torch.Tensor
    action_attention_mask: torch.Tensor
    graph_decoder_input_ids: torch.Tensor
    graph_decoder_attention_mask: torch.Tensor

    # Labels for the decoders (shifted right by 1)
    action_labels: torch.Tensor
    graph_labels: torch.Tensor


class JerichoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        text_tokenizer: "TextTokenizer",
        triple_tokenizer: "TripleTokenizer",
        action_tokenizer: "ActionTokenizer",
        max_length: int = 1024,
        cache_dir: Optional[str] = None,
    ):
        self.data_path = pathlib.Path(data_path)
        self.text_tokenizer = text_tokenizer
        self.triple_tokenizer = triple_tokenizer
        self.action_tokenizer = action_tokenizer
        self.max_length = max_length

        # Load samples
        self.samples = list(stream(str(data_path)))

        # Create cache if specified
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            self.cache_file = self.cache_dir / f"{self.data_path.stem}_processed.pt"

            if self.cache_file.exists():
                self.processed_samples = torch.load(self.cache_file)
            else:
                self.processed_samples = self._preprocess_samples()
                torch.save(self.processed_samples, self.cache_file)
        else:
            self.cache_dir = None
            self.processed_samples = None

    def _preprocess_samples(self) -> list[JerichoSample]:
        """Preprocess all samples for faster loading."""
        processed = []
        for sample in self.samples:
            processed.append(self._process_sample(sample))
        return processed

    def _process_sample(self, sample: JerichoSample) -> Sample:
        """Process a single sample into model inputs."""
        # Encode text input (observation + valid actions)
        text_encoding = self.text_tokenizer(
            observation=sample["state"]["obs"],
            valid_actions=sample["state"]["valid_acts"],
            max_length=self.max_length,
        )

        # Encode current graph for encoder
        graph_encoding = self.triple_tokenizer(
            triples=sample["state"]["graph"],
            is_encoder=True,
            max_length=self.max_length,
        )

        # Encode next valid actions for action decoder
        action_encoding = self.action_tokenizer(
            valid_actions=sample["next_state"]["valid_acts"], max_length=self.max_length
        )

        # Encode graph differences for graph decoder
        graph_decoder_encoding = self.triple_tokenizer(
            triples=sample["graph_diff"], is_encoder=False, max_length=self.max_length
        )

        # Create shifted labels for the decoders
        action_labels = action_encoding["input_ids"][1:].clone()
        action_labels = torch.cat(
            [
                action_labels,
                torch.tensor([self.action_tokenizer.tokenizer.eos_token_id]),
            ]
        )

        graph_labels = graph_decoder_encoding["input_ids"][1:].clone()
        graph_labels = torch.cat(
            [
                graph_labels,
                torch.tensor([self.triple_tokenizer.gpt2_tokenizer.eos_token_id]),
            ]
        )

        return Sample(
            text_input_ids=text_encoding["input_ids"],
            text_attention_mask=text_encoding["attention_mask"],
            graph_input_ids=graph_encoding["input_ids"],
            graph_attention_mask=graph_encoding["attention_mask"],
            action_input_ids=action_encoding["input_ids"],
            action_attention_mask=action_encoding["attention_mask"],
            graph_decoder_input_ids=graph_decoder_encoding["input_ids"],
            graph_decoder_attention_mask=graph_decoder_encoding["attention_mask"],
            action_labels=action_labels,
            graph_labels=graph_labels,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        if self.processed_samples is not None:
            return self.processed_samples[idx]
        return self._process_sample(self.samples[idx])


class JerichoDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data/jericho-world") -> None:
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)

    def setup(self, stage: str) -> None:
        text_tokenizer, triple_tokenizer, action_tokenizer = create_tokenizers()

        if stage == "fit":
            full = JerichoDataset(
                data_path=self.data_dir / "train.json",
                text_tokenizer=text_tokenizer,
                triple_tokenizer=triple_tokenizer,
                action_tokenizer=action_tokenizer,
            )

            size_train, size_val = (
                int(len(full) * 0.9),
                len(full) - int((len(full) * 0.9)),
            )
            self.train, self.val = torch.utils.data.random_split(
                full,
                [size_train, size_val],
                generator=torch.Generator().manual_seed(2209),
            )

        if stage == "test" or stage == "predict":
            self.test = JerichoDataset(
                data_path=self.data_dir / "test.json",
                text_tokenizer=text_tokenizer,
                triple_tokenizer=triple_tokenizer,
                action_tokenizer=action_tokenizer,
            )
            self.predict = self.test

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.train, batch_size=16, num_workers=15)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.val, batch_size=16, num_workers=15)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.test, batch_size=16, num_workers=15)

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.predict, batch_size=16, num_workers=15)


if __name__ == "__main__":
    """Test the dataset implementation."""
    from rich.console import Console
    from rich.table import Table

    data_path = "./data/jericho-world/train.json"

    console = Console()

    console.print("\n[bold]Testing Dataset Implementation[/bold]")

    text_tokenizer, triple_tokenizer, action_tokenizer = create_tokenizers(
        data_path=data_path
    )

    # Create dataset
    dataset = JerichoDataset(
        data_path=data_path,
        text_tokenizer=text_tokenizer,
        triple_tokenizer=triple_tokenizer,
        action_tokenizer=action_tokenizer,
    )

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    # Get a batch
    batch = next(iter(dataloader))

    # Print batch information
    table = Table(title="Batch Information")
    table.add_column("Key", style="cyan")
    table.add_column("Shape", style="green")
    table.add_column("Type", style="yellow")

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            table.add_row(key, str(tuple(value.shape)), f"{value.dtype}")
        else:
            table.add_row(key, str(len(value)), str(type(value)))

    console.print(table)

    # Test sequence starts
    idx = 0
    console.print("\n[bold]Checking sequence formats:[/bold]")

    def print_sequence(name: str, decode) -> None:
        console.print(f"{name}: {decode(batch[name])[:100]}...")

    print_sequence("text_input_ids", text_tokenizer.decode)
    print_sequence(
        "graph_input_ids", functools.partial(triple_tokenizer.decode, is_encoder=True)
    )
    print_sequence("action_input_ids", action_tokenizer.decode)
    print_sequence(
        "graph_decoder_input_ids",
        functools.partial(triple_tokenizer.decode, is_encoder=False),
    )
