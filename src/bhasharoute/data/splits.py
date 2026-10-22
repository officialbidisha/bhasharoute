"""Named training / tuning / evaluation views over data/processed/dataset.jsonl.

All models are fit on `train`, tuned (early stopping, temperature, thresholds) on `dev`,
and scored once on `test`. Held-out domains never appear in train or dev views.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from bhasharoute.config import PROCESSED_DIR, SEED

SYNTHETIC_VARIANTS = ("romanized", "codemix", "codemix_romanized")


def load_rows(path: Path = PROCESSED_DIR / "dataset.jsonl") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def training_rows(rows: list[dict], augment: bool, seed: int = SEED) -> list[dict]:
    """In-domain train rows. With `augment`, each (source id, language) also contributes ONE
    synthetic variant sampled uniformly - bounding compute while exposing the model to
    romanised and code-switched surface forms of the same request."""
    native = [r for r in rows if r["split"] == "train" and r["in_domain"] and r["variant"] == "native"]
    if not augment:
        return native
    pools: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["split"] == "train" and r["in_domain"] and r["variant"] in SYNTHETIC_VARIANTS:
            pools[(r["group_id"], r["lang"])].append(r)
    rng = random.Random(seed)
    sampled = [rng.choice(pool) for _, pool in sorted(pools.items())]
    return native + sampled


def dev_rows(rows: list[dict]) -> list[dict]:
    """Human-translated in-domain dev rows: early stopping, calibration, thresholds."""
    return [r for r in rows if r["split"] == "dev" and r["in_domain"] and r["variant"] == "native"]


def test_sets(rows: list[dict]) -> dict[str, list[dict]]:
    test = [r for r in rows if r["split"] == "test"]
    ind_native = [r for r in test if r["in_domain"] and r["variant"] == "native"]
    sets = {"native_all": ind_native}
    for lang in ("en", "hi", "bn"):
        sets[f"native_{lang}"] = [r for r in ind_native if r["lang"] == lang]
    sets["human_codemix"] = [r for r in ind_native if r["human_codemix"]]
    for v in SYNTHETIC_VARIANTS:
        sets[f"synthetic_{v}"] = [r for r in test if r["in_domain"] and r["variant"] == v]
    sets["ood_native"] = [r for r in test if not r["in_domain"] and r["variant"] == "native"]
    return sets
