"""Evaluate every available model on identical data and write reports/.

    uv run python -m bhasharoute.eval.harness

Protocol (no test-set tuning):
  * dev (human-translated, in-domain) -> temperature T, abstention threshold tau (95% dev
    coverage), side-effect threshold (>=99% dev recall). Saved to artifacts/<model>/calibration.json.
  * test -> scored once with those frozen settings.
  * OOD = test rows of the held-out domains (cooking, takeaway); never seen in train/dev.
"""
from __future__ import annotations

import argparse
import gc
import json
import random

import numpy as np
import torch

from bhasharoute.config import ARTIFACTS_DIR, CHALLENGE_PATH, REPORTS_DIR, SEED
from bhasharoute.data.splits import dev_rows, load_rows, test_sets
from bhasharoute.eval import latency as lat
from bhasharoute.eval.metrics import (binary_report, classification_report, confusion,
                                      expected_calibration_error, fit_temperature, group_consistency,
                                      ood_detection, reliability_curve, selective, sigmoid, softmax,
                                      threshold_for_recall)
from bhasharoute.labels import ID_TO_SCENARIO, SCENARIO_TO_ID
from bhasharoute.models.rules import KeywordRouter, side_effect_trigger
from bhasharoute.serving.gate import side_effect_gate

MODEL_NAMES = ["A_rules", "B_tfidf", "B_tfidf_aug", "C_xlmr_frozen", "C_xlmr_finetune", "C_xlmr_finetune_aug"]
LABELS = list(SCENARIO_TO_ID)
TARGET_SIDE_RECALL = 0.99


def load_model(name: str, device: str):
    if name == "A_rules":
        return KeywordRouter()
    d = ARTIFACTS_DIR / name
    if not (d / ("model.joblib" if name.startswith("B_") else "config.json")).exists():
        return None
    if name.startswith("B_"):
        from bhasharoute.models.tfidf import TfidfClassifier
        return TfidfClassifier.load(d)
    from bhasharoute.models.transformer_infer import TransformerClassifier
    return TransformerClassifier(d, device)


def _y(rows):
    return np.array([SCENARIO_TO_ID.get(r["scenario"], -1) for r in rows])


def _s(rows):
    return np.array([bool(r["side_effect"]) for r in rows])


def _texts(rows):
    return [r["text"] for r in rows]


class Scored:
    """Applies a model's dev-fitted calibration to raw predictions."""

    def __init__(self, preds, calib: dict | None):
        self.p, self.c = preds, calib
        if calib:
            self.prob = softmax(preds.scenario_logits, calib["temperature"])
            self.conf = self.prob.max(1)
            self.side_prob = sigmoid(preds.side_logit, calib["side_temperature"])
            self.side_pred = self.side_prob >= calib["side_threshold"]
        else:
            self.prob = self.conf = self.side_prob = None
            self.side_pred = preds.side_pred.astype(bool)
        self.pred = preds.scenario_pred

    def gate(self, texts: list[str]) -> np.ndarray:
        if self.c is None:  # rules: the gate is the lexical guard alone
            return np.array([side_effect_trigger(t) for t in texts])
        return np.array([side_effect_gate(t, float(sp), self.c["side_threshold"]).requires_confirmation
                         for t, sp in zip(texts, self.side_prob)])


def calibrate(name: str, model, dev: list[dict]) -> dict | None:
    p = model.predict(_texts(dev))
    if p.scenario_logits is None:
        return None
    y, s = _y(dev), _s(dev)
    t = fit_temperature(p.scenario_logits, y)
    te = fit_temperature(p.side_logit, s.astype(float), binary=True)
    conf = softmax(p.scenario_logits, t).max(1)
    calib = {"temperature": t, "side_temperature": te,
             "abstain_threshold": float(np.quantile(conf, 0.05)),
             "side_threshold": threshold_for_recall(s, sigmoid(p.side_logit, te), TARGET_SIDE_RECALL),
             "fitted_on": f"dev native in-domain (n={len(dev)})",
             "policy": "tau keeps 95% of dev requests; side threshold gives >=99% dev recall"}
    if name != "A_rules":
        (ARTIFACTS_DIR / name / "calibration.json").write_text(json.dumps(calib, indent=2))
    return calib


def evaluate(name: str, model, dev, sets, challenge) -> tuple[dict, dict]:
    calib = calibrate(name, model, dev)
    scored = {k: Scored(model.predict(_texts(rows)), calib) for k, rows in sets.items()}
    out: dict = {"name": name, "calibration": calib, "sets": {}}

    for k, rows in sets.items():
        if k == "ood_native":
            continue
        sc, y, s = scored[k], _y(rows), _s(rows)
        rep = classification_report(y, sc.pred, LABELS)
        if k != "native_all":
            rep.pop("per_class")
        if calib:
            correct = sc.pred == y
            rep["ece_uncalibrated"] = expected_calibration_error(softmax(sc.p.scenario_logits).max(1), correct)
            rep["ece_calibrated"] = expected_calibration_error(sc.conf, correct)
            rep["selective_at_tau"] = selective(sc.conf, correct, calib["abstain_threshold"])
        rep["side_effect_model"] = binary_report(s, sc.side_pred, sc.side_prob)
        rep["side_effect_gate"] = binary_report(s, sc.gate(_texts(rows)))
        out["sets"][k] = rep

    ood_rows, sc_ood = sets["ood_native"], scored["ood_native"]
    ood: dict = {"n": len(ood_rows)}
    if calib:
        ood.update(ood_detection(scored["native_all"].conf, sc_ood.conf))
        ood["abstain_rate_at_tau"] = float((sc_ood.conf < calib["abstain_threshold"]).mean())
        routed = [ID_TO_SCENARIO[int(i)] for i in sc_ood.pred[sc_ood.conf >= calib["abstain_threshold"]]]
        ood["false_accept_routed_to"] = dict(sorted(((r, routed.count(r)) for r in set(routed)),
                                                    key=lambda kv: -kv[1])[:5])
    else:
        ood["abstain_rate_at_tau"] = float((sc_ood.pred == -1).mean())
    se_mask = _s(ood_rows)
    ood["unseen_domain_side_effect_gate_recall"] = float(sc_ood.gate(_texts(ood_rows))[se_mask].mean())
    ood["unseen_domain_side_effect_model_recall"] = float(sc_ood.side_pred[se_mask].mean())
    out["ood"] = ood

    nat = sets["native_all"]
    out["consistency_en_hi_bn"] = group_consistency([r["group_id"] for r in nat], scored["native_all"].pred)
    pairs_rows = [r for r in nat if r["lang"] != "en"] + sets["synthetic_romanized"]
    pairs_pred = np.r_[scored["native_all"].pred[[r["lang"] != "en" for r in nat]],
                       scored["synthetic_romanized"].pred]
    out["consistency_native_vs_romanized"] = group_consistency(
        [f"{r['group_id']}:{r['lang']}" for r in pairs_rows], pairs_pred)

    out["challenge"] = score_challenge(model, calib, challenge)
    extras = {"native_all": scored["native_all"], "ood": sc_ood}
    return out, extras


def score_challenge(model, calib, rows: list[dict]) -> dict:
    sc = Scored(model.predict(_texts(rows)), calib)
    abstained = (sc.pred == -1) if calib is None else (sc.conf < calib["abstain_threshold"])
    gate = sc.gate(_texts(rows))
    per_cat: dict[str, list[bool]] = {}
    items = []
    for i, r in enumerate(rows):
        exp = r["expected_scenario"]
        got = None if abstained[i] else ID_TO_SCENARIO.get(int(sc.pred[i]))
        ok = (got is None) if exp is None else (got == exp)
        per_cat.setdefault(r["category"], []).append(ok)
        items.append({"id": r["id"], "text": r["text"], "expected": exp, "got": got,
                      "confidence": None if sc.conf is None else round(float(sc.conf[i]), 3),
                      "expected_side_effect": r["expected_side_effect"], "gated": bool(gate[i])})
    needs = [it for it in items if it["expected_side_effect"] is True]
    false_conf = [it for it in items if it["expected_side_effect"] is False and it["gated"]]
    return {"routing_accuracy": float(np.mean([v for vs in per_cat.values() for v in vs])),
            "per_category": {k: f"{sum(v)}/{len(v)}" for k, v in per_cat.items()},
            "side_effects_gated": f"{sum(it['gated'] for it in needs)}/{len(needs)}",
            "ungated_side_effects": [it["text"] for it in needs if not it["gated"]],
            "extra_confirmations_on_read_only": f"{len(false_conf)}/{sum(1 for it in items if it['expected_side_effect'] is False)}",
            "items": items}


def measure_efficiency(name: str, model, texts: list[str], nat_rows: list[dict]) -> dict:
    torch.set_num_threads(4)
    if name == "A_rules":
        return {"latency_cpu": lat.per_request_latency(lambda t: model.route(t), texts)}
    if name.startswith("B_"):
        size = (ARTIFACTS_DIR / name / "model.joblib").stat().st_size / 1e6
        return {"size_mb": size, "latency_cpu": lat.per_request_latency(lambda t: model.predict([t]), texts)}
    from bhasharoute.models.transformer_infer import TransformerClassifier
    cpu = TransformerClassifier(ARTIFACTS_DIR / name, "cpu")
    res = {"params_total_m": round(sum(p.numel() for p in cpu.model.parameters()) / 1e6, 1),
           "size_fp32_mb": round(lat.state_dict_mb(cpu.model), 1),
           "latency_cpu_fp32": lat.per_request_latency(lambda t: cpu.logits([t], 1), texts)}
    q = lat.dynamic_int8(cpu.model)
    if q is not None:
        cpu.model = q.eval()
        res["size_int8_mb"] = round(lat.state_dict_mb(q), 1)
        res["latency_cpu_int8"] = lat.per_request_latency(lambda t: cpu.logits([t], 1), texts)
        logits, _ = cpu.logits(_texts(nat_rows), 64)
        y = _y(nat_rows)
        from sklearn.metrics import f1_score
        res["int8_native_all"] = {"accuracy": float((logits.argmax(1) == y).mean()),
                                  "macro_f1": float(f1_score(y, logits.argmax(1), average="macro"))}
    del cpu
    gc.collect()
    return res


def save_figures(name: str, extras: dict, rows: list[dict], fig_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    sc, y = extras["native_all"], _y(rows)
    cm = confusion(y, sc.pred, len(LABELS)).astype(float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(LABELS)), LABELS, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(f"{name}: row-normalised confusion (test, native en/hi/bn)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(fig_dir / f"confusion_{name}.png", dpi=130)
    plt.close(fig)
    np.savetxt(fig_dir / f"confusion_{name}.csv", cm, fmt="%d", delimiter=",", header=",".join(LABELS))

    if sc.c:
        correct = sc.pred == y
        fig, ax = plt.subplots(figsize=(5, 5))
        for conf, lbl in [(softmax(sc.p.scenario_logits).max(1), "before (T=1)"),
                          (sc.conf, f"after (T={sc.c['temperature']:.2f})")]:
            m, a, _ = reliability_curve(conf, correct)
            ax.plot(m, a, marker="o", label=lbl)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_title(f"{name}: reliability (test)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"reliability_{name}.png", dpi=130)
        plt.close(fig)


def error_sample(extras: dict, rows: list[dict], k: int = 6) -> dict:
    sc, y = extras["native_all"], _y(rows)
    rng = random.Random(SEED)
    out = {}
    for lang in ("en", "hi", "bn"):
        idx = [i for i, r in enumerate(rows) if r["lang"] == lang and sc.pred[i] != y[i]]
        out[lang] = [{"text": rows[i]["text"], "gold": rows[i]["scenario"],
                      "pred": ID_TO_SCENARIO.get(int(sc.pred[i])),
                      "confidence": None if sc.conf is None else round(float(sc.conf[i]), 3)}
                     for i in rng.sample(idx, min(k, len(idx)))]
    cm = confusion(y, sc.pred, len(LABELS))
    np.fill_diagonal(cm, 0)
    flat = sorted(((int(cm[i, j]), LABELS[i], LABELS[j]) for i in range(len(LABELS))
                   for j in range(len(LABELS)) if cm[i, j]), reverse=True)[:8]
    out["top_confusions"] = [{"gold": g, "pred": p, "count": c} for c, g, p in flat]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODEL_NAMES))
    ap.add_argument("--skip-efficiency", action="store_true")
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    rows = load_rows()
    dev, sets = dev_rows(rows), test_sets(rows)
    challenge = [json.loads(line) for line in open(CHALLENGE_PATH, encoding="utf-8")]
    lat_texts = random.Random(SEED).sample(_texts(sets["native_all"]), 200)

    REPORTS_DIR.mkdir(exist_ok=True)
    metrics_path = REPORTS_DIR / "metrics.json"
    results = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    results["_data"] = {k: len(v) for k, v in sets.items()} | {"dev": len(dev)}

    for name in args.models.split(","):
        model = load_model(name, device)
        if model is None:
            print(f"skip {name}: no artifacts")
            continue
        print(f"== {name}", flush=True)
        res, extras = evaluate(name, model, dev, sets, challenge)
        res["errors"] = error_sample(extras, sets["native_all"])
        save_figures(name, extras, sets["native_all"], REPORTS_DIR / "figures")
        if not args.skip_efficiency:
            res["efficiency"] = measure_efficiency(name, model, lat_texts, sets["native_all"])
        results[name] = res
        metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        na = res["sets"]["native_all"]
        print(f"   native acc {na['accuracy']:.4f} macro-F1 {na['macro_f1']:.4f}", flush=True)
        del model, extras
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    from bhasharoute.eval.report import update_readme, write_report
    write_report(results)
    update_readme(results)


if __name__ == "__main__":
    main()
