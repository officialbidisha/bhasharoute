"""Approach C - multilingual Transformer router in PyTorch.

Architecture
    text -> XLM-R tokenizer (SentencePiece, 250k vocab) -> input_ids + attention_mask
         -> XLM-R encoder (12 layers, hidden 768)
         -> masked mean pooling over real (non-pad) tokens
         -> dropout -> two heads:
              scenario head : Linear(768 -> 16)  softmax over in-domain skills
              side-effect   : Linear(768 -> 1)   sigmoid, trained with its own loss

Two training regimes (kept clearly separate in reports):
    * frozen   - encoder weights fixed; only an MLP head is trained on cached pooled
                 features. A legitimate low-compute baseline, NOT fine-tuning.
    * finetune - all 12 transformer layers + heads are updated. The 192M-parameter
                 token-embedding matrix (69% of XLM-R's 278M params) stays frozen so the
                 AdamW state fits in 8 GB of unified memory; gradients still flow through
                 every attention / feed-forward layer.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

from bhasharoute.labels import SCENARIO_TO_ID


class RequestDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.texts = [r["text"] for r in rows]
        self.scenarios = [SCENARIO_TO_ID.get(r["scenario"], -100) for r in rows]  # -100 = ignored by CE
        self.side = [float(r["side_effect"]) for r in rows]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict:
        return {"text": self.texts[i], "scenario": self.scenarios[i], "side": self.side[i]}


class Collator:
    """Tokenises a batch with *dynamic* padding: each batch is padded only to its own
    longest sequence (MASSIVE requests are short, ~15 sub-words), and the attention mask
    marks real tokens (1) vs padding (0) so neither attention nor pooling sees pads.

    Lengths are rounded up to a multiple of 16. On Apple MPS every new input shape
    compiles and caches a new kernel graph; fully dynamic lengths produced dozens of
    shapes, the cache grew past 5 GB on an 8 GB machine and training thrashed swap.
    Bucketing to {16, 32, 48, 64} keeps the cache small at a negligible padding cost."""

    def __init__(self, tokenizer, max_len: int = 64, pad_multiple: int = 16):
        self.tok, self.max_len, self.pad_multiple = tokenizer, max_len, pad_multiple

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        enc = self.tok([b["text"] for b in batch], padding=True, truncation=True,
                       max_length=self.max_len, pad_to_multiple_of=self.pad_multiple,
                       return_tensors="pt")
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "scenario": torch.tensor([b["scenario"] for b in batch], dtype=torch.long),
            "side": torch.tensor([b["side"] for b in batch], dtype=torch.float),
        }


def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average token states where mask == 1. hidden [B, T, H], mask [B, T] -> [B, H]."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)


class Heads(nn.Module):
    """Scenario + side-effect heads. hidden_dim > 0 inserts a shared MLP layer (used in
    frozen mode, where the head must do more of the work)."""

    def __init__(self, in_dim: int, n_scenarios: int, hidden_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        if hidden_dim:
            self.trunk = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden_dim), nn.GELU())
            in_dim = hidden_dim
        else:
            self.trunk = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.scenario = nn.Linear(in_dim, n_scenarios)
        self.side_effect = nn.Linear(in_dim, 1)

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.dropout(self.trunk(pooled))
        return self.scenario(h), self.side_effect(h).squeeze(-1)


class RouterNet(nn.Module):
    def __init__(self, encoder_name: str, n_scenarios: int, mode: str, head_hidden: int = 0,
                 dropout: float = 0.1, revision: str | None = None):
        super().__init__()
        assert mode in ("frozen", "finetune")
        self.mode = mode
        self.encoder = AutoModel.from_pretrained(encoder_name, revision=revision)
        self.heads = Heads(self.encoder.config.hidden_size, n_scenarios, head_hidden, dropout)
        frozen = self.encoder.parameters() if mode == "frozen" else self.encoder.embeddings.parameters()
        for p in frozen:
            p.requires_grad = False

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return masked_mean_pool(out.last_hidden_state, attention_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.heads(self.encode(input_ids, attention_mask))

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Checkpoint only what training changed; frozen weights are reloaded from the
        pinned base-model revision recorded in config.json."""
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in keep}


def load_tokenizer(encoder_name: str, revision: str | None = None):
    return AutoTokenizer.from_pretrained(encoder_name, revision=revision)
