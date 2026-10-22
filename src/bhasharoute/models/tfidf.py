"""Approach B - TF-IDF + logistic regression, the classical NLP baseline.

Features: character n-grams (2-5, word-boundary aware) + word uni/bi-grams over the
normalised text. Character n-grams are what make this baseline strong across scripts:
they need no tokenizer and partially transfer between spelling variants.

Word tokenisation deliberately uses whitespace splitting after `normalize`, NOT the
regex `\\w+`: Python's `\\w` does not match Devanagari/Bengali vowel signs (Unicode
category Mc/Mn), so `\\w+` silently shreds "मुझे" into "म" + "झ".

    uv run python -m bhasharoute.models.tfidf [--augment]
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

from bhasharoute.config import ARTIFACTS_DIR, SEED
from bhasharoute.data.splits import dev_rows, load_rows, training_rows
from bhasharoute.labels import SCENARIO_TO_ID
from bhasharoute.models.base import Predictions
from bhasharoute.text import normalize

C_GRID = (1.0, 3.0, 10.0, 30.0)


def _whitespace_tokens(text: str) -> list[str]:
    return text.split()


def make_features() -> FeatureUnion:
    return FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                                 preprocessor=normalize, max_features=300_000)),
        ("word", TfidfVectorizer(tokenizer=_whitespace_tokens, token_pattern=None, ngram_range=(1, 2),
                                 min_df=2, sublinear_tf=True, preprocessor=normalize)),
    ])


class TfidfClassifier:
    def __init__(self, name: str, features: FeatureUnion, scenario_lr: LogisticRegression,
                 side_lr: LogisticRegression):
        self.name = name
        self.features, self.scenario_lr, self.side_lr = features, scenario_lr, side_lr

    def predict(self, texts: list[str]) -> Predictions:
        x = self.features.transform(texts)
        logits = self.scenario_lr.decision_function(x)  # multinomial logits (softmax -> proba)
        side = self.side_lr.decision_function(x)        # binary logit
        return Predictions(logits, logits.argmax(1), side, side > 0)

    def save(self, out_dir) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_dir / "model.joblib")

    @staticmethod
    def load(out_dir) -> "TfidfClassifier":
        return joblib.load(out_dir / "model.joblib")


def train(augment: bool) -> None:
    rows = load_rows()
    tr, dv = training_rows(rows, augment), dev_rows(rows)
    t0 = time.time()
    feats = make_features()
    x_tr = feats.fit_transform([r["text"] for r in tr])
    x_dv = feats.transform([r["text"] for r in dv])
    y_tr = np.array([SCENARIO_TO_ID[r["scenario"]] for r in tr])
    y_dv = np.array([SCENARIO_TO_ID[r["scenario"]] for r in dv])
    s_tr = np.array([r["side_effect"] for r in tr])
    s_dv = np.array([r["side_effect"] for r in dv])

    log = {"n_train": len(tr), "n_dev": len(dv), "n_features": x_tr.shape[1], "c_search": {}}
    best = None
    for c in C_GRID:  # model selection on dev only
        lr = LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=SEED)
        lr.fit(x_tr, y_tr)
        f1 = f1_score(y_dv, lr.predict(x_dv), average="macro")
        log["c_search"][c] = round(f1, 4)
        if best is None or f1 > best[0]:
            best = (f1, c, lr)
    side_best = None
    for c in C_GRID:
        lr = LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=SEED)
        lr.fit(x_tr, s_tr)
        auc = roc_auc_score(s_dv, lr.decision_function(x_dv))
        if side_best is None or auc > side_best[0]:
            side_best = (auc, c, lr)
    name = "B_tfidf_aug" if augment else "B_tfidf"
    model = TfidfClassifier(name, feats, best[2], side_best[2])
    model.save(ARTIFACTS_DIR / name)
    log.update({"best_C": best[1], "dev_macro_f1": round(best[0], 4), "side_best_C": side_best[1],
                "side_dev_auroc": round(side_best[0], 4), "train_seconds": round(time.time() - t0, 1)})
    (ARTIFACTS_DIR / name / "train_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--augment", action="store_true")
    train(ap.parse_args().augment)
