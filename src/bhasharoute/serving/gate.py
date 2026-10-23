"""Side-effect safety gate. The rule this file enforces: ML may ADD friction, never remove it.

Defence in depth, three independent layers:
  1. Lexical guard  - deterministic multilingual action-phrase triggers (models/rules.py).
  2. Model head     - the side-effect probability, thresholded for >=99% dev recall.
  3. Tool policy    - the authoritative check. Every tool declares whether it mutates
                      state; a mutating tool executes only with explicit user confirmation,
                      whatever the router predicted.

Layers 1-2 decide whether to ask early (better UX, fewer surprises). Layer 3 decides
whether anything actually happens, and it never reads a model output.
"""
from __future__ import annotations

from dataclasses import dataclass

from bhasharoute.models.rules import side_effect_trigger


@dataclass(frozen=True)
class GateDecision:
    requires_confirmation: bool
    reasons: tuple[str, ...]


def side_effect_gate(text: str, side_prob: float | None, threshold: float) -> GateDecision:
    reasons = []
    if side_effect_trigger(text):
        reasons.append("lexical_trigger")
    if side_prob is None:
        reasons.append("model_unavailable_fail_closed")
    elif side_prob >= threshold:
        reasons.append("model_side_effect")
    return GateDecision(bool(reasons), tuple(reasons))


@dataclass(frozen=True)
class Tool:
    name: str
    mutates: bool


def may_execute(tool: Tool, user_confirmed: bool) -> bool:
    """Authoritative execution check. Deliberately takes no model output as input."""
    return user_confirmed if tool.mutates else True
