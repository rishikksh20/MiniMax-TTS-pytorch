import torch
from torch import nn
from core.attention import GQAttention
from torch.nn import functional as F
from core.modules import RMSNorm
from core.pos_encoding import rope_rotate


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x),  *args, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class GatedFeedForward(nn.Module):
    def __init__(self, idim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(idim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(idim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, idim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3(nn.Module):
    def __init__(self, dim, depth, n_heads, num_groups, head_dim, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, GQAttention(dim, n_heads = n_heads, num_groups=num_groups, head_dim = head_dim)),
                PreNorm(dim, GatedFeedForward(dim, mlp_dim))
            ]))
    def forward(self, x, cos, sin, mask=None):
        for attn, ff in self.layers:
            x = attn(x, cos, sin, mask) + x
            x = ff(x) + x
        return x


class Qwen3Model(nn.Module):
    def __init__(self, dim, depth, n_heads, num_groups, head_dim, mlp_dim, vocab_size, context_length,
                 dtype=torch.float32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim, dtype=dtype)
        self.qwen3 = Qwen3(dim, depth, n_heads, num_groups, head_dim, mlp_dim)

        self.final_norm = RMSNorm(dim, eps=1e-6)
        self.final_proj = nn.Linear(dim, vocab_size, bias=False, dtype=dtype)

        cos, sin = rope_rotate(head_dim, context_length)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.dtype = dtype


    def forward(self, x):
        emb = self.tok_emb(x)
        n = emb.shape[1]
        mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1)

        x = self.qwen3(emb, self.cos, self.sin, mask)

        x = self.final_norm(x)
        x = self.final_proj(x.to(self.dtype))
        return x



def test_qwen3_0_6B():
    QWEN3_CONFIG = {
        "vocab_size": 151_936,  # Vocabulary size
        "context_length": 40_960,  # Context length that was used to train the model
        "emb_dim": 1024,  # Embedding dimension
        "n_heads": 16,  # Number of attention heads
        "n_layers": 28,  # Number of layers
        "hidden_dim": 3072,  # Size of the intermediate dimension in FeedForward
        "head_dim": 128,  # Size of the heads in GQA
        "qk_norm": True,  # Whether to normalize queries and keys in GQA
        "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
        "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
        "dtype": torch.float32
    }
    model = Qwen3Model(dim=1024, depth=28, n_heads=16, num_groups=8, head_dim=128,
                       mlp_dim=3072, vocab_size=151936, context_length=40960)
    device = torch.device("cpu")
    out = model(torch.tensor([1, 2, 3]).unsqueeze(0)).to(device)

    print("Model output shape : ", out.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Account for weight tying
    total_params_normalized = total_params - model.tok_emb.weight.numel()
    print(f"\nTotal number of unique parameters: {total_params_normalized:,}")

    print("\nModel : \n", model)


if __name__ == "__main__":
    test_qwen3_0_6B()




