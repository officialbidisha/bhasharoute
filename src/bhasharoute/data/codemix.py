"""Slot-aligned code-switching: a SYNTHETIC approximation of how Indian users mix English
entities into Hindi/Bengali requests ("kal subah seven am ka alarm laga do").

MASSIVE annotates slots in every language and the translations are parallel, so the
i-th `time` slot in the Hindi utterance and the i-th `time` slot in the English source
refer to the same entity. Swapping the Hindi value for the English one yields a
code-mixed request whose label is unchanged by construction - no model or LLM is
involved in labelling.
"""
from __future__ import annotations

from collections import defaultdict

from bhasharoute.data.massive import _SLOT, parse_slots


def codeswitch(target_annot: str, english_annot: str) -> str | None:
    """Return the target utterance with slot values replaced by their English
    counterparts, or None when no slot could be aligned (nothing to switch)."""
    english_by_type: dict[str, list[str]] = defaultdict(list)
    for slot in parse_slots(english_annot):
        english_by_type[slot.name].append(slot.value)

    used: dict[str, int] = defaultdict(int)
    switched = 0

    def _swap(match) -> str:
        nonlocal switched
        name, value = match.group(1), match.group(2)
        k = used[name]
        used[name] += 1
        candidates = english_by_type.get(name, [])
        if k < len(candidates) and candidates[k].strip() and candidates[k] != value:
            switched += 1
            return candidates[k]
        return value

    text = _SLOT.sub(_swap, target_annot)
    return text if switched else None
