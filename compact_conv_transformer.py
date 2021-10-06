from torch.nn.modules import dropout
from torch.nn.modules.normalization import LayerNorm
from base_lit_mods import BaseClassifierLitMod
import torch
from torch import nn, matmul
from torch import functional as F

def sanity(x, expected_size, note=None):
    enabled = False
    if not enabled:
        return
    print(f'Expected size of {expected_size}, got size of {x.shape}')
    if note is not None:
        print('  ' + str(note))

def scaled_dot_product_attention(q, k, v):
    scale = q.size(-1) ** -0.5
    scores = nn.functional.softmax(q.bmm(k.transpose(1, 2)) * scale, dim=-1)
    return scores.bmm(v)


class AttentionHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def forward(self, x):
        return scaled_dot_product_attention(self.q(x), self.k(x), self.v(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(d_model) for _ in range(n_heads)]
        )
        self.linear = nn.Linear(n_heads * d_model, d_model)

    def forward(self, x):
        return self.linear(torch.cat([h(x) for h in self.heads], dim=-1))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.1):
        super().__init__()

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(n_heads, d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.dropout_1(self.attention(self.norm_1(x)))
        x = x + x2 # Residual connection
        x2 = self.dropout_2(self.feed_forward(self.norm_2(x)))
        x = x + x2
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_channels=3,
        d_model=128,
        patch_size=4,
        stride=2,
        heads = 2,
        n_layers = 2,
        n_classes = 10,
        ):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels, out_channels=16,
                kernel_size=patch_size, stride=stride,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=d_model,
                kernel_size=patch_size, stride=stride,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(start_dim=2),
        )
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff=d_model*2) for _ in range(n_layers)])
        self.norm = LayerNorm(d_model)
        self.attention_pool = nn.Linear(d_model, 1)
        self.classifier = nn.Sequential(
            #nn.Linear(d_model, d_model),
            #nn.ReLU(),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        sanity(x, ('bs', 'c', 'h', 'w'))
        x = self.embed(x).transpose(-2, -1)
        sanity(x, ('bs', 'n', 'd_model'))
        # Skip position embedding for now since this model doesn't strictly need it.
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        sanity(x, ('bs', 'n', 'd_model'))
        # Should be shape (batch_size, seq_len, d_model)
        softmax = nn.functional.softmax(self.attention_pool(x), dim=1).transpose(-1, -2)
        x = softmax.bmm(x).squeeze(-2)
        #x = matmul(nn.functional.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        sanity(x, ('bs', 'd_model'))
        # Should be shape (batch_size, d_model)
        x = self.classifier(x)
        sanity(x, ('bs', 'n_classes'))
        return x


class CCTLitMod(BaseClassifierLitMod):
    def __init__(self, n_channels):
        super().__init__(lr=5e-4, weight_decay=3e-2)
        self.model = TransformerEncoder(
            n_channels=n_channels,
            n_layers=2,
            heads=2,
            patch_size=3,
            stride=2,
            )