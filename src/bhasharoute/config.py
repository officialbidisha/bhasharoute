"""Project-wide constants: paths, label space, held-out domains, seeds."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CHALLENGE_PATH = ROOT / "data" / "challenge" / "challenge.jsonl"
CONFIG_DIR = ROOT / "configs"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

LOCALES = {"en": "en-US", "hi": "hi-IN", "bn": "bn-BD"}

# Whole domains withheld from training and threshold tuning. Their *test* rows are the
# out-of-distribution set used to measure abstention ("I can't help with that yet").
# Food was chosen because it overlaps lexically with in-domain skills (qa, lists,
# recommendation, transport booking) - a realistic, hard OOD case rather than an easy one.
HOLDOUT_SCENARIOS = ("cooking", "takeaway")

ALL_SCENARIOS = (
    "alarm", "audio", "calendar", "cooking", "datetime", "email", "general", "iot", "lists",
    "music", "news", "play", "qa", "recommendation", "social", "takeaway", "transport", "weather",
)
IN_DOMAIN_SCENARIOS = tuple(s for s in ALL_SCENARIOS if s not in HOLDOUT_SCENARIOS)

SEED = 13
