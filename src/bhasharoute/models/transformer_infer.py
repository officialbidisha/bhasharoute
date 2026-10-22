"""Load a trained approach-C checkpoint and produce Predictions for the harness / server."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bhasharoute.models.base import Predictions
from bhasharoute.models.encoder import Collator, RequestDataset, RouterNet, load_tokenizer


class TransformerClassifier:
    def __init__(self, artifact_dir: Path, device: str | torch.device = "cpu"):
        self.artifact_dir = artifact_dir
        self.cfg = json.loads((artifact_dir / "config.json").read_text())
        self.name = self.cfg["name"]
        self.version = f"{self.name}@{self.cfg['weights_sha256_12']}"
        self.device = torch.device(device)
        rev = self.cfg.get("revision")
        self.tok = load_tokenizer(self.cfg["encoder"], rev)
        self.model = RouterNet(self.cfg["encoder"], self.cfg["n_scenarios"], self.cfg["mode"],
                               self.cfg["head_hidden"], revision=rev)
        state = torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        frozen = {n for n, p in self.model.named_parameters() if not p.requires_grad}
        buffers = {n for n, _ in self.model.named_buffers()}
        assert not unexpected, unexpected
        assert set(missing) <= frozen | buffers, f"checkpoint is missing trained weights: {set(missing) - frozen}"
        self.model.to(self.device).eval()
        self.collate = Collator(self.tok, self.cfg["max_len"])

    @torch.no_grad()
    def logits(self, texts: list[str], batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
        ds = RequestDataset([{"text": t, "scenario": "", "side_effect": False} for t in texts])
        scen, side = [], []
        for b in DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=self.collate):
            s, e = self.model(b["input_ids"].to(self.device), b["attention_mask"].to(self.device))
            scen.append(s.float().cpu())
            side.append(e.float().cpu())
        return torch.cat(scen).numpy(), torch.cat(side).numpy()

    def predict(self, texts: list[str]) -> Predictions:
        scen, side = self.logits(texts)
        return Predictions(scen, scen.argmax(1), side, side > 0)
