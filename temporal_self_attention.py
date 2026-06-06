import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads=8, causal_mask=False):
        super().__init__()
        assert model_dim % num_heads == 0

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.causal_mask = causal_mask
        self.head_dim = model_dim // num_heads

        self.query_projection = nn.Linear(model_dim, model_dim)
        self.key_projection = nn.Linear(model_dim, model_dim)
        self.value_projection = nn.Linear(model_dim, model_dim)
        self.output_projection = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        target_length = query.shape[-2]
        source_length = key.shape[-2]

        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)

        attention_scores = (query @ key) / (self.head_dim ** 0.5)

        if self.causal_mask:
            mask = torch.ones(
                target_length,
                source_length,
                dtype=torch.bool,
                device=query.device
            ).tril()
            attention_scores.masked_fill_(~mask, -torch.inf)

        attention_weights = torch.softmax(attention_scores, dim=-1)
        output = attention_weights @ value

        output = torch.cat(
            torch.split(output, batch_size, dim=0),
            dim=-1
        )

        output = self.output_projection(output)

        return output


class TemporalSelfAttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim,
        feed_forward_dim=2048,
        num_heads=8,
        dropout=0,
        causal_mask=False
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            causal_mask=causal_mask
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim)
        )

        self.attention_norm = nn.LayerNorm(model_dim)
        self.feed_forward_norm = nn.LayerNorm(model_dim)

        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(self, x, attention_dim=-2):
        x = x.transpose(attention_dim, -2)

        residual = x
        output = self.attention(x, x, x)
        output = self.attention_dropout(output)
        output = self.attention_norm(residual + output)

        residual = output
        output = self.feed_forward(output)
        output = self.feed_forward_dropout(output)
        output = self.feed_forward_norm(residual + output)

        output = output.transpose(attention_dim, -2)

        return output

