"""Unicode-aware text normalisation and script detection shared by every model."""
from __future__ import annotations

import re
import unicodedata

_DIGITS = str.maketrans("०१२३४५६७८९০১২৩৪৫৬৭৮৯", "01234567890123456789")
_SPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[।॥!?.,;:\"'()\[\]{}]")  # includes danda / double danda


def normalize(text: str) -> str:
    """NFC, unify Indic digits, casefold Latin, strip punctuation, collapse whitespace.

    Used for de-duplication keys and by the classical models. The transformer receives
    the NFC text only - its own tokenizer handles case and punctuation.
    """
    t = unicodedata.normalize("NFC", text).translate(_DIGITS).casefold()
    t = _PUNCT.sub(" ", t)
    return _SPACE.sub(" ", t).strip()


def script_of(ch: str) -> str | None:
    cp = ord(ch)
    if 0x0900 <= cp <= 0x097F:
        return "Deva"
    if 0x0980 <= cp <= 0x09FF:
        return "Beng"
    if ch.isascii() and ch.isalpha():
        return "Latn"
    return None


def detect_script(text: str) -> str:
    """Dominant-script label; 'mixed' when a second script covers >=20% of letters."""
    counts: dict[str, int] = {}
    for ch in text:
        s = script_of(ch)
        if s:
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        return "none"
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    if len(ranked) > 1 and ranked[1][1] / total >= 0.2:
        return "mixed"
    return ranked[0][0]
