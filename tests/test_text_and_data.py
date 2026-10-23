import re

import pytest

from bhasharoute.config import PROCESSED_DIR
from bhasharoute.data.codemix import codeswitch
from bhasharoute.data.massive import human_judged_codemix, parse_slots, strip_annotation
from bhasharoute.data.translit import romanize
from bhasharoute.text import detect_script, normalize


def test_normalize_unifies_digits_case_and_punctuation():
    assert normalize("Wake me at ৭ AM!") == "wake me at 7 am"
    assert normalize("मुझे  ५ बजे   जगा दो।") == "मुझे 5 बजे जगा दो"


def test_regex_word_tokenization_breaks_indic_but_whitespace_does_not():
    # The pitfall the TF-IDF tokenizer avoids: \w misses Devanagari vowel signs.
    assert re.findall(r"\w+", "मुझे") != ["मुझे"]
    assert normalize("मुझे जगा दो").split() == ["मुझे", "जगा", "दो"]


@pytest.mark.parametrize("text,script", [
    ("set an alarm", "Latn"), ("अलार्म लगाओ", "Deva"), ("অ্যালার্ম সেট করো", "Beng"),
    ("मेरा email check करो", "mixed"), ("123", "none"),
])
def test_detect_script(text, script):
    assert detect_script(text) == script


@pytest.mark.parametrize("src,lang,expected", [
    ("मुझे इस सप्ताह सुबह पांच बजे जगा दो", "hi", "mujhe is saptah subah panch baje jaga do"),
    ("अलार्म लगाओ", "hi", "alarm lagao"),
    ("मित्र", "hi", "mitra"),            # conjunct-final keeps its vowel
    ("পড়ার জন্য আলো", "bn", "padar janya alo"),
    ("অ্যালার্ম সেট করুন", "bn", "alarm set karun"),
    ("olly चुप रहो", "hi", "olly chup raho"),  # Latin tokens pass through
])
def test_romanize(src, lang, expected):
    assert romanize(src, lang) == expected


def test_slot_codeswitch_swaps_aligned_values_only():
    hi = "मुझे [date : शुक्रवार] को सुबह [time : नौ बजे] जगा दो"
    en = "wake me up at [time : nine am] on [date : friday]"
    assert codeswitch(hi, en) == "मुझे friday को सुबह nine am जगा दो"
    assert codeswitch("बंद करो", "stop") is None


def test_massive_helpers():
    assert parse_slots("[time : five am] alarm")[0].name == "time"
    assert strip_annotation("wake me at [time : five am]") == "wake me at five am"
    j = lambda li: {"language_identification": li}
    assert human_judged_codemix({"judgments": [j("target|english"), j("target|english"), j("target")]})
    assert not human_judged_codemix({"judgments": [j("target|english"), j("target"), j("target")]})


@pytest.mark.skipif(not (PROCESSED_DIR / "dataset.jsonl").exists(), reason="run bhasharoute.data.build first")
def test_built_dataset_has_no_leakage():
    from bhasharoute.data.splits import load_rows
    rows = load_rows()
    group_split, by_split = {}, {"train": set(), "dev": set(), "test": set()}
    for r in rows:
        assert group_split.setdefault(r["group_id"], r["split"]) == r["split"], "group crosses splits"
        by_split[r["split"]].add(normalize(r["text"]))
    assert not by_split["train"] & by_split["test"]
    assert not by_split["train"] & by_split["dev"]
    assert not by_split["dev"] & by_split["test"]
    assert all(not r["in_domain"] for r in rows if r["scenario"] in ("cooking", "takeaway"))
