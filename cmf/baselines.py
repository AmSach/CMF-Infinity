from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import CMFConfig
from .model import DilatedContextEncoder


class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.embedding(input_ids) + self.position(positions).unsqueeze(0)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        states = self.encoder(x, mask=mask)
        logits = self.output(self.norm(states))
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
        if return_states:
            result["states"] = states
        return result


class TemporalConvLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        config = CMFConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=True,
            tie_embeddings=False,
        )
        self.encoder = DilatedContextEncoder(config)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        states = self.encoder(self.embedding(input_ids))
        logits = self.output(self.norm(states))
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
        if return_states:
            result["states"] = states
        return result


class TinyGPTLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                GPTBlock(
                    d_model=d_model,
                    nhead=nhead,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        _batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.embedding(input_ids) + self.position(positions).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        states = self.norm(x)
        logits = self.output(states)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
        if return_states:
            result["states"] = states
        return result


class GPTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        y = self.norm_1(x)
        qkv = self.qkv(y)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        x = x + self.resid_dropout(self.proj(attn))
        x = x + self.mlp(self.norm_2(x))
        return x
