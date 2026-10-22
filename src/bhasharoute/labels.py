"""Label spaces for the two independent tasks: domain routing and side-effect detection."""
from __future__ import annotations

from functools import lru_cache

import yaml

from bhasharoute.config import CONFIG_DIR, IN_DOMAIN_SCENARIOS

SCENARIO_TO_ID = {s: i for i, s in enumerate(IN_DOMAIN_SCENARIOS)}
ID_TO_SCENARIO = dict(enumerate(IN_DOMAIN_SCENARIOS))


@lru_cache(maxsize=1)
def side_effect_intents() -> frozenset[str]:
    with open(CONFIG_DIR / "side_effects.yaml", encoding="utf-8") as f:
        return frozenset(yaml.safe_load(f)["side_effect_intents"])


def is_side_effect(intent: str) -> bool:
    return intent in side_effect_intents()
