"""Rule-based romanisation of Hindi / Bengali into casual "Hinglish" / "Banglish".

Real users in India often type Hindi and Bengali in Latin script. MASSIVE only ships
native-script text, so romanised variants here are SYNTHETIC: OPTITRANS
transliteration followed by heuristics that move it toward how people actually type:

* word-final schwa deletion  (kala -> kal, mausama -> mausam)
* anusvara / chandrabindu -> n (pAMcha -> panch)
* long-vowel and retroflex case distinctions dropped (A -> a, T -> t)
* Bengali: ব -> b (Bengali has no separate v), nukta removed (য় -> y)

Known limitations (documented in the datasheet): medial schwa is kept (Bengali
"kemana" stays "kemana", real users type "kemon"); spelling is deterministic,
whereas real users vary ("mujhe" / "muje"). These slices are a robustness probe, not a
substitute for native-speaker data.
"""
from __future__ import annotations

import re
import unicodedata

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

_SRC = {"hi": sanscript.DEVANAGARI, "bn": sanscript.BENGALI}

# Consonant letters: if a word ends in one of these (no vowel sign, no virama), the
# inherent vowel is normally silent in speech.
_DEVA_CONS = set(range(0x0915, 0x093A)) | set(range(0x0958, 0x0960))
_BENG_CONS = set(range(0x0995, 0x09BA)) | {0x09CE, 0x09DC, 0x09DD, 0x09DF}
_NUKTA = {0x093C, 0x09BC}
_VIRAMA = {0x094D, 0x09CD}
_RA = {0x0930, 0x09B0}
_KEEPS_VOWEL_HI = {0x0930, 0x092F, 0x0935}  # र / य / व as the second member (मित्र, सत्य)

_REPLACEMENTS = [
    (".N", "n"), ("~N", "n"), ("~n", "n"), ("M", "n"), (".h", ""), ("H", "h"),
    ("R^i", "ri"), ("RRi", "ri"), ("^", ""), ("़", ""), ("়", ""),
]
_COLLAPSE = [("aa", "a"), ("ii", "i"), ("uu", "u")]


def _strip_nukta(word: str) -> str:
    # ड़/ढ़/ज़/য় etc. -> base letter; casual typing rarely distinguishes them.
    decomposed = unicodedata.normalize("NFD", word)
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if ord(c) not in _NUKTA))


def _drops_final_schwa(word: str, lang: str) -> bool:
    """True when the inherent vowel of a word-final bare consonant is usually silent.

    Simple consonant ending: silent (kal, mausam). Reph conjunct (र्म, র্ম): silent, which
    covers loanwords like alarm/form. Other conjuncts: Bengali keeps the vowel (জন্য ->
    janya), Hindi keeps it only for र/य/व second members (मित्र -> mitra, but स्टेशन्स ...).
    """
    cons = _DEVA_CONS if lang == "hi" else _BENG_CONS
    if not word or ord(word[-1]) not in cons:
        return False
    if len(word) < 3 or ord(word[-2]) not in _VIRAMA:
        return True
    if ord(word[-3]) in _RA:
        return True
    return lang == "hi" and ord(word[-1]) not in _KEEPS_VOWEL_HI


def _clean(roman: str, lang: str) -> str:
    for a, b in _REPLACEMENTS:
        roman = roman.replace(a, b)
    roman = roman.lower()
    for a, b in _COLLAPSE:
        roman = roman.replace(a, b)
    if lang == "bn":
        roman = roman.replace("v", "b")
    return re.sub(r"[^a-z0-9']", "", roman)


def romanize_word(word: str, lang: str) -> str:
    if not any(0x0900 <= ord(c) <= 0x09FF for c in word):
        return word  # already Latin (code-mixed English token) or a number
    word = _strip_nukta(word)
    if lang == "bn":
        word = word.replace("অ্যা", "আ")  # অ্যা spells an "æ"/"a" sound
    roman = _clean(transliterate(word, _SRC[lang], sanscript.OPTITRANS), lang)
    if _drops_final_schwa(word, lang) and len(roman) > 2 and roman.endswith("a"):
        roman = roman[:-1]
    return roman


def romanize(text: str, lang: str) -> str:
    if lang not in _SRC:
        raise ValueError(f"romanize supports hi/bn, got {lang!r}")
    text = text.replace("।", " ").replace("॥", " ")
    return " ".join(romanize_word(w, lang) for w in text.split())
