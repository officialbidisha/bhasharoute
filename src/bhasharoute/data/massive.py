"""Readers for the MASSIVE 1.1 release (Amazon, CC BY 4.0).

Every MASSIVE utterance id is one English source request that professional translators
localised into each language; the train/dev/test partition is assigned per id, so a
request and all of its translations always share a split.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_SLOT = re.compile(r"\[(\w+) : (.*?)\]")


@dataclass(frozen=True)
class Slot:
    name: str
    value: str


def load_locale(raw_dir: Path, locale: str) -> list[dict]:
    with open(raw_dir / f"{locale}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def parse_slots(annot_utt: str) -> list[Slot]:
    return [Slot(m.group(1), m.group(2)) for m in _SLOT.finditer(annot_utt)]


def strip_annotation(annot_utt: str) -> str:
    return _SLOT.sub(lambda m: m.group(2), annot_utt)


def human_judged_codemix(row: dict) -> bool:
    """Majority of MASSIVE's native-speaker judges marked the utterance as mixing
    the target language with English (language_identification == "target|english...")."""
    judgments = row.get("judgments") or []
    if not judgments:
        return False
    mixed = sum(1 for j in judgments if "english" in j.get("language_identification", "")
                and "target" in j.get("language_identification", ""))
    return mixed * 2 > len(judgments)
