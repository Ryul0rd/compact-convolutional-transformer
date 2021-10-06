from torch.nn.modules import dropout
from torch.nn.modules.normalization import LayerNorm
from base_lit_mods import BaseClassifierLitMod
from torch import nn, matmul
from torch import functional as F

def sanity(x, expected_size, note=None):
    print(f'Expected size of {expected_size}, got size of {x.shape}')
    if note is not None:
        print('  ' + str(note))

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.h = heads
        self.d_model = d_model
        assert d_model % heads == 0, 'embedding_size must be evenly divisible by the number of heads'
        self.d_k = d_model // heads # Head dimension
        self.scale = self.d_k ** -0.5 # scale by 1/sqrt(d) to keep values small

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        #self.qkv = nn.Linear(embedding_size, 3 * embedding_size, bias=False)

    def attention(self, q, k, v):
        k = k.transpose(1, 2)
        scores = nn.functional.softmax(q.bmm(k) * self.scale, dim=-1)
        output = scores.bmm(v)

        return output

    def forward(self, x):
        batch_size = x.size(0)

        # perform linear operation and split into h heads
        
        k = self.k_linear(x).view(batch_size, -1, self.h, self.d_k)
        q = self.q_linear(x).view(batch_size, -1, self.h, self.d_k)
        v = self.v_linear(x).view(batch_size, -1, self.h, self.d_k)
        sanity(k, ('bs', 'n', 'h', 'd_k'))

        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        sanity(k, ('bs', 'h', 'sl', 'd_model'))

	    # calculate attention
        scores = self.attention(q, k, v)
        sanity(scores, ('bs', 'h', 'sl', 'd_model'))
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        sanity(concat, ('bs', 'n', 'd_model'))

        output = self.out(concat)
    
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff=256, dropout=0.1):
        super().__init__()

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(heads, d_model)
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
                in_channels=n_channels, out_channels=d_model,
                kernel_size=patch_size, stride=stride,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(start_dim=2),
        )
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff=d_model*2) for i in range(n_layers)])
        self.norm = LayerNorm(d_model)
        self.attention_pool = nn.Linear(d_model, 1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
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
        return x


class CCTLitMod(BaseClassifierLitMod):
    def __init__(self, n_channels):
        super().__init__(lr=5e-4, weight_decay=3e-2)
        self.model = TransformerEncoder(
            n_channels=n_channels,
            n_layers=2,
            #n_layers=6,
            heads=4,
            patch_size=6,
            stride=2,
            )