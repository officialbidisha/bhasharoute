"""Approach A - deterministic keyword router + lexical side-effect guard.

This is the kind of router a team ships before it has data: hand-written multilingual
keyword lists (English, Hindi, Bengali, and romanised forms). It is the floor every
learned model must beat, and its side-effect lexicon is reused at serving time as an
independent guard that ML can never override (see serving/gate.py).
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import yaml

from bhasharoute.config import CONFIG_DIR
from bhasharoute.labels import SCENARIO_TO_ID
from bhasharoute.models.base import Predictions
from bhasharoute.text import normalize


def _compile(phrases: list[str]) -> list[tuple[re.Pattern, int]]:
    out = []
    for p in phrases:
        p = normalize(p)
        if p.isascii():
            pat = re.compile(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])")
        else:
            # Indic: substring match - inflections attach directly to the stem (जगा -> जगाओ)
            pat = re.compile(re.escape(p))
        out.append((pat, len(p)))
    return out


@lru_cache(maxsize=1)
def _lexicon() -> tuple[dict[str, list], list]:
    with open(CONFIG_DIR / "rules.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    scen = {s: _compile(phrases) for s, phrases in cfg["scenarios"].items()}
    return scen, _compile(cfg["side_effect_triggers"])


def side_effect_trigger(text: str) -> bool:
    t = normalize(text)
    return any(p.search(t) for p, _ in _lexicon()[1])


class KeywordRouter:
    name = "A_rules"

    def route(self, text: str) -> str | None:
        t = normalize(text)
        best, best_score = None, 0
        for scenario, patterns in _lexicon()[0].items():
            score = sum(w for p, w in patterns if p.search(t))  # longer phrase = more specific
            if score > best_score:
                best, best_score = scenario, score
        return best

    def predict(self, texts: list[str]) -> Predictions:
        routed = [self.route(t) for t in texts]
        pred = np.array([SCENARIO_TO_ID.get(r, -1) if r else -1 for r in routed])
        side = np.array([side_effect_trigger(t) for t in texts])
        return Predictions(None, pred, None, side)
