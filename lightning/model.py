import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForMaskedLM,
    BertConfig,
    BertModel,
    GPT2Config,
    GPT2LMHeadModel,
)

import lightning.pytorch as L
from tokenizer import create_tokenizers


class CustomBertConfig(BertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_hidden_layers = 6
        self.num_attention_heads = 6
        self.hidden_size = 768
        self.intermediate_size = 3072
        self.max_position_embeddings = 1024
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.hidden_act = "gelu"


class CustomGPT2Config(GPT2Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_layer = 6
        self.n_head = 6
        self.n_embd = 768
        self.n_inner = 3072
        self.n_positions = 1024
        self.resid_pdrop = 0.1
        self.embd_pdrop = 0.1
        self.attn_pdrop = 0.1
        self.activation_function = "gelu"
        self.add_cross_attention = True


class AggregatorLayer(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=2, intermediate_size=4096):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_attention_heads, dropout=0.1, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        attention_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attention_output)
        feed_forward_output = self.feed_forward(x)
        x = self.layer_norm2(x + feed_forward_output)
        return x


class Aggregator(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        num_layers=2,
        num_attention_heads=2,
        intermediate_size=4096,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AggregatorLayer(hidden_size, num_attention_heads, intermediate_size)
                for _ in range(num_layers)
            ]
        )

    def forward(self, text_hidden_states, graph_hidden_states):
        # 2048 = 1024 + 1024
        combined = torch.cat([text_hidden_states, graph_hidden_states], dim=1)

        for layer in self.layers:
            combined = layer(combined)

        return combined


class WorldformerModel(L.LightningModule):
    def __init__(
        self,
        text_bert_name="bert-base-uncased",
        graph_bert_name="bert-base-uncased",
        learning_rate=3e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize tokenizers first to get vocab sizes
        self.text_tokenizer, self.triple_tokenizer, self.action_tokenizer = (
            create_tokenizers()
        )
        # Text Encoder (BERT-based)
        text_config = CustomBertConfig()
        # Set vocab size to match tokenizer's vocabulary
        text_config.vocab_size = len(self.text_tokenizer.tokenizer)
        self.text_encoder = BertModel(text_config)

        # Graph Encoder (BERT-based)
        graph_config = CustomBertConfig()
        # Set vocab size to match triple tokenizer's BERT vocabulary
        graph_config.vocab_size = len(self.triple_tokenizer.bert_tokenizer)
        self.graph_encoder = BertModel(graph_config)

        # Aggregator
        self.aggregator = Aggregator(
            hidden_size=768, num_layers=2, num_attention_heads=2, intermediate_size=4096
        )

        action_decoder_config = CustomGPT2Config()
        action_decoder_config.vocab_size = len(self.action_tokenizer.tokenizer)
        self.action_decoder = GPT2LMHeadModel(action_decoder_config)

        graph_decoder_config = CustomGPT2Config()
        graph_decoder_config.vocab_size = len(self.triple_tokenizer.gpt2_tokenizer)
        self.graph_decoder = GPT2LMHeadModel(graph_decoder_config)

        # Resize token embeddings for all models
        self.text_encoder.resize_token_embeddings(len(self.text_tokenizer.tokenizer))
        self.graph_encoder.resize_token_embeddings(
            len(self.triple_tokenizer.bert_tokenizer)
        )
        self.action_decoder.resize_token_embeddings(
            len(self.action_tokenizer.tokenizer)
        )
        self.graph_decoder.resize_token_embeddings(
            len(self.triple_tokenizer.gpt2_tokenizer)
        )

    def forward(
        self,
        text_input_ids,
        text_attention_mask,
        graph_input_ids,
        graph_attention_mask,
        action_input_ids=None,
        action_attention_mask=None,
        graph_decoder_input_ids=None,
        graph_decoder_attention_mask=None,
    ):
        # Encode text
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            return_dict=True,
        )
        text_hidden_states = text_outputs.last_hidden_state

        # Encode graph
        graph_outputs = self.graph_encoder(
            input_ids=graph_input_ids,
            attention_mask=graph_attention_mask,
            return_dict=True,
        )
        graph_hidden_states = graph_outputs.last_hidden_state

        # Aggregate encodings
        aggregated_states = self.aggregator(text_hidden_states, graph_hidden_states)

        # Decode actions and graph updates
        action_outputs = self.action_decoder(
            input_ids=action_input_ids,
            attention_mask=action_attention_mask,
            encoder_hidden_states=aggregated_states,
            return_dict=True,
        )

        graph_outputs = self.graph_decoder(
            input_ids=graph_decoder_input_ids,
            attention_mask=graph_decoder_attention_mask,
            encoder_hidden_states=aggregated_states,
            return_dict=True,
        )

        return action_outputs.logits, graph_outputs.logits

    def training_step(self, batch, batch_idx):
        action_logits, graph_logits = self(
            text_input_ids=batch["text_input_ids"],
            text_attention_mask=batch["text_attention_mask"],
            graph_input_ids=batch["graph_input_ids"],
            graph_attention_mask=batch["graph_attention_mask"],
            action_input_ids=batch["action_input_ids"],
            action_attention_mask=batch["action_attention_mask"],
            graph_decoder_input_ids=batch["graph_decoder_input_ids"],
            graph_decoder_attention_mask=batch["graph_decoder_attention_mask"],
        )

        # Calculate losses
        action_loss = F.cross_entropy(
            action_logits.view(-1, action_logits.size(-1)),
            batch["action_labels"].view(-1),
            ignore_index=-100,
        )

        graph_loss = F.cross_entropy(
            graph_logits.view(-1, graph_logits.size(-1)),
            batch["graph_labels"].view(-1),
            ignore_index=-100,
        )

        total_loss = action_loss + graph_loss

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("action_loss", action_loss, prog_bar=True)
        self.log("graph_loss", graph_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        action_logits, graph_logits = self(
            text_input_ids=batch["text_input_ids"],
            text_attention_mask=batch["text_attention_mask"],
            graph_input_ids=batch["graph_input_ids"],
            graph_attention_mask=batch["graph_attention_mask"],
            action_input_ids=batch["action_input_ids"],
            action_attention_mask=batch["action_attention_mask"],
            graph_decoder_input_ids=batch["graph_decoder_input_ids"],
            graph_decoder_attention_mask=batch["graph_decoder_attention_mask"],
        )

        action_loss = F.cross_entropy(
            action_logits.view(-1, action_logits.size(-1)),
            batch["action_labels"].view(-1),
            ignore_index=-100,
        )

        graph_loss = F.cross_entropy(
            graph_logits.view(-1, graph_logits.size(-1)),
            batch["graph_labels"].view(-1),
            ignore_index=-100,
        )

        total_loss = action_loss + graph_loss

        self.log("val_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("val_action_loss", action_loss, prog_bar=True, sync_dist=True)
        self.log("val_graph_loss", graph_loss, prog_bar=True, sync_dist=True)

        return total_loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def generate(
        self,
        text_input_ids,
        text_attention_mask,
        graph_input_ids,
        graph_attention_mask,
        max_length=1024,
        num_beams=15,
    ):
        # Encode inputs
        with torch.no_grad():
            text_hidden_states = self.text_encoder(
                input_ids=text_input_ids, attention_mask=text_attention_mask
            ).last_hidden_state

            graph_hidden_states = self.graph_encoder(
                input_ids=graph_input_ids, attention_mask=graph_attention_mask
            ).last_hidden_state

            aggregated_states = self.aggregator(text_hidden_states, graph_hidden_states)

            # Generate action sequence
            action_outputs = self.action_decoder.generate(
                encoder_hidden_states=aggregated_states,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

            # Generate graph updates
            graph_outputs = self.graph_decoder.generate(
                encoder_hidden_states=aggregated_states,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        return action_outputs, graph_outputs
