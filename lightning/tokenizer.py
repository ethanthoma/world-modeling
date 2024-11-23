import itertools
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import sentencepiece as spm
import torch
from transformers import (
    BertTokenizer,
    GPT2Tokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


class TextTokenizer:
    def __init__(self, bert_model_name: str = "bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # Add special tokens
        special_tokens = {"additional_special_tokens": ["[OBS]", "[ACT]"]}
        self.tokenizer.add_special_tokens(special_tokens)

    def __call__(
        self, observation: str, valid_actions: Dict[str, str], max_length: int = 1024
    ) -> Dict[str, torch.Tensor]:
        # Format text with special tokens
        formatted_text = f"[OBS] {observation}"

        # Add valid actions
        for action in valid_actions.values():
            formatted_text += f" [ACT] {action}"

        # Encode and pad
        encoding = self.tokenizer(
            formatted_text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

    def decode(self, input_ids: torch.Tensor) -> str:
        """Decode input IDs back into text."""
        # If input_ids is batched, take first example
        if input_ids.dim() > 1:
            input_ids = input_ids[0]

        return self.tokenizer.decode(input_ids, skip_special_tokens=False)


class TripleTokenizer:
    def __init__(
        self, bert_model_name: str = "bert-base-uncased", gpt2_model_name: str = "gpt2"
    ):
        # Initialize both tokenizers
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

        # Add special tokens to both
        special_tokens = {
            "additional_special_tokens": ["[GRAPH]", "[TRIPLE]", "[EOS]"],
            "pad_token": "[PAD]",
        }
        self.bert_tokenizer.add_special_tokens(special_tokens)
        self.gpt2_tokenizer.add_special_tokens(special_tokens)

        # Ensure GPT2 has necessary tokens
        if self.gpt2_tokenizer.pad_token is None:
            self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token

    def _format_triples(self, triples: List[List[str]], is_encoder: bool = True) -> str:
        """Format triples with appropriate special tokens."""
        formatted_text = "[GRAPH]"

        formatted_text += " [TRIPLE]".join(
            itertools.starmap(lambda s, r, o: f" {s}, {r}, {o}", triples)
        )

        if not is_encoder:
            formatted_text = formatted_text.strip() + self.gpt2_tokenizer.eos_token

        return formatted_text.strip()

    def __call__(
        self, triples: List[List[str]], is_encoder: bool, max_length: int = 1024
    ) -> Dict[str, torch.Tensor]:
        """Encode triples for the BERT-based encoder."""
        if is_encoder:
            formatted_text = self._format_triples(triples, is_encoder=True)

            encoding = self.bert_tokenizer(
                formatted_text,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            }
        else:
            formatted_text = self._format_triples(triples, is_encoder=False)

            encoding = self.gpt2_tokenizer(
                formatted_text,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            }

    def decode(self, input_ids: torch.Tensor, is_encoder: bool) -> str:
        """Decode encoder input IDs back into text."""
        if is_encoder:
            if input_ids.dim() > 1:
                input_ids = input_ids[0]

            return self.bert_tokenizer.decode(input_ids, skip_special_tokens=False)
        else:
            if input_ids.dim() > 1:
                input_ids = input_ids[0]

            return self.gpt2_tokenizer.decode(input_ids, skip_special_tokens=False)


class ActionTokenizer:
    """
    Whitespace tokenizer for action sequences based on GPT2.
    """

    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Add special tokens
        special_tokens = {
            "additional_special_tokens": ["[ACT]", "[EOS]"],
            "pad_token": "[PAD]",
        }
        self.tokenizer.add_special_tokens(special_tokens)

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(
        self, valid_actions: Dict[str, str], max_length: int = 1024
    ) -> Dict[str, torch.Tensor]:
        """Make the tokenizer callable for encoding actions."""
        formatted_text = ""
        for action in valid_actions.values():
            formatted_text += f"[ACT] {action} "

        formatted_text = formatted_text.strip() + self.tokenizer.eos_token

        # Encode and pad
        encoding = self.tokenizer(
            formatted_text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

    def decode(self, input_ids: torch.Tensor) -> str:
        """Decode action IDs back into text."""
        if input_ids.dim() > 1:
            input_ids = input_ids[0]

        return self.tokenizer.decode(input_ids, skip_special_tokens=False)


def create_tokenizers():
    text_tokenizer = TextTokenizer()
    triple_tokenizer = TripleTokenizer()
    action_tokenizer = ActionTokenizer()

    return text_tokenizer, triple_tokenizer, action_tokenizer


if __name__ == "__main__":
    create_tokenizers()
