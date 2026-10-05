"""
model_def.py
Architecture definition for the SentimentScope transformer classifier.
Import this file wherever you need to rebuild the model and load trained weights.
"""
import math

import torch
import torch.nn as nn
from transformers import AutoTokenizer

# ----------------------------------------------------------------------------
# Constants / config (must match what the model was trained with)
# ----------------------------------------------------------------------------
MAX_LENGTH = 248
TOKENIZER_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

config = {
    "vocabulary_size": tokenizer.vocab_size,  # 30522 for bert-base-uncased
    "num_classes": 2,                         # 0 = negative, 1 = positive
    "d_embed": 248,
    "context_size": MAX_LENGTH,
    "layers_num": 8,
    "heads_num": 4,
    "head_size": 32,
    "dropout_rate": 0.1,
    "use_bias": True,
}


# ----------------------------------------------------------------------------
# Model classes
# ----------------------------------------------------------------------------
class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Q_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.K_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.V_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])

        self.dropout = nn.Dropout(config["dropout_rate"])

        casual_attention_mask = torch.tril(torch.ones(config["context_size"], config["context_size"]))
        self.register_buffer("casual_attention_mask", casual_attention_mask)

    def forward(self, input):
        batch_size, tokens_num, d_embed = input.shape
        Q = self.Q_weights(input)
        K = self.K_weights(input)
        V = self.V_weights(input)

        attention_scores = Q @ K.transpose(1, 2)
        attention_scores = attention_scores.masked_fill(
            self.casual_attention_mask[:tokens_num, :tokens_num] == 0,
            float("-inf"),
        )
        attention_scores = attention_scores / math.sqrt(K.shape[-1])
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        return attention_scores @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        heads_list = [AttentionHead(config) for _ in range(config["heads_num"])]
        self.heads = nn.ModuleList(heads_list)

        self.linear = nn.Linear(config["heads_num"] * config["head_size"], config["d_embed"])
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, input):
        heads_outputs = [head(input) for head in self.heads]
        x = torch.cat(heads_outputs, dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(config["d_embed"], 4 * config["d_embed"]),
            nn.GELU(),
            nn.Linear(4 * config["d_embed"], config["d_embed"]),
            nn.Dropout(config["dropout_rate"]),
        )

    def forward(self, input):
        return self.linear_layers(input)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.multi_head = MultiHeadAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config["d_embed"])

        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = nn.LayerNorm(config["d_embed"])

    def forward(self, input):
        x = input
        x = x + self.multi_head(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class DemoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_layer = nn.Embedding(config["vocabulary_size"], config["d_embed"])
        self.positional_embedding_layer = nn.Embedding(config["context_size"], config["d_embed"])

        blocks = [Block(config) for _ in range(config["layers_num"])]
        self.layers = nn.Sequential(*blocks)

        self.layer_norm = nn.LayerNorm(config["d_embed"])
        self.output = nn.Linear(config["d_embed"], config["num_classes"])

    def forward(self, token_ids):
        batch_size, tokens_num = token_ids.shape

        x = self.token_embedding_layer(token_ids)
        positions = torch.arange(tokens_num, device=token_ids.device)
        pos_embed = self.positional_embedding_layer(positions)
        x = x + pos_embed.unsqueeze(0)

        x = self.layers(x)
        x = self.layer_norm(x)

        x = torch.mean(x, dim=1)   # mean pooling over tokens
        logits = self.output(x)    # (B, num_classes)
        return logits


# ----------------------------------------------------------------------------
# Helper for loading trained weights
# ----------------------------------------------------------------------------
def load_model(weights_path="model.pth", device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DemoGPT(config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model