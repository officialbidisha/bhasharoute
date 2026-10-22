"""Common prediction container so every approach is scored by the same harness."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Predictions:
    # [N, K] uncalibrated logits, or None for models without scores (rules)
    scenario_logits: np.ndarray | None
    # [N] argmax class id; -1 means the model produced no answer
    scenario_pred: np.ndarray
    # [N] uncalibrated side-effect logit, or None
    side_logit: np.ndarray | None
    # [N] hard side-effect decision (used for models without scores)
    side_pred: np.ndarray


class Classifier(Protocol):
    name: str

    def predict(self, texts: list[str]) -> Predictions: ...
