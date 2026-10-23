"""Metrics: classification, calibration (ECE + temperature scaling), OOD detection,
selective prediction, side-effect detection, cross-surface-form consistency."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def sigmoid(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x / temperature))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, binary: bool = False) -> float:
    """Temperature scaling (Guo et al., 2017): one scalar T > 0 minimising dev NLL.
    Optimised in log-space so T stays positive. Does not change the argmax."""
    z = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64 if binary else torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    loss_fn = torch.nn.BCEWithLogitsLoss() if binary else torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().item())


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """ECE: |accuracy - confidence| per equal-width confidence bin, weighted by bin size."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence > lo) & (confidence <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - confidence[m].mean())
    return float(ece)


def reliability_curve(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15):
    edges = np.linspace(0, 1, n_bins + 1)
    mids, accs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence > lo) & (confidence <= hi)
        if m.any():
            mids.append(confidence[m].mean())
            accs.append(correct[m].mean())
            counts.append(int(m.sum()))
    return np.array(mids), np.array(accs), np.array(counts)


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    """y_pred == -1 (no answer) counts as an error for accuracy and recall."""
    k = list(range(len(labels)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=k, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=k, average="macro", zero_division=0)),
        "no_answer_rate": float((y_pred == -1).mean()),
        "per_class": {labels[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                                  "support": int(s[i])} for i in k},
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n)))


def ood_detection(conf_in: np.ndarray, conf_out: np.ndarray) -> dict:
    """In-domain is the positive class; score = max softmax probability."""
    y = np.r_[np.ones(len(conf_in)), np.zeros(len(conf_out))]
    s = np.r_[conf_in, conf_out]
    tau95 = np.quantile(conf_in, 0.05)  # keeps 95% of in-domain requests
    return {"auroc": float(roc_auc_score(y, s)), "fpr_at_95tpr": float((conf_out >= tau95).mean())}


def selective(conf: np.ndarray, correct: np.ndarray, tau: float) -> dict:
    answered = conf >= tau
    return {"coverage": float(answered.mean()),
            "selective_accuracy": float(correct[answered].mean()) if answered.any() else float("nan"),
            "risk_if_answering_all": float(1 - correct.mean())}


def binary_report(y: np.ndarray, pred: np.ndarray, score: np.ndarray | None = None) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    out = {"precision": float(p), "recall": float(r), "f1": float(f), "n_pos": int(y.sum()), "n": int(len(y))}
    if score is not None and 0 < y.sum() < len(y):
        out["auroc"] = float(roc_auc_score(y, score))
    return out


def threshold_for_recall(y: np.ndarray, score: np.ndarray, target: float = 0.99) -> float:
    """Largest threshold whose recall on (dev) data is >= target: safety-first operating point."""
    pos = np.sort(score[y.astype(bool)])
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[k]) if len(pos) else 0.5


def group_consistency(group_ids: list[str], preds: np.ndarray) -> float:
    """Fraction of source requests (with >=2 surface forms present) whose forms all get the
    same prediction - e.g. the English, Hindi and Bengali versions of one request."""
    by_group: dict[str, set] = defaultdict(set)
    sizes: dict[str, int] = defaultdict(int)
    for g, p in zip(group_ids, preds):
        by_group[g].add(int(p))
        sizes[g] += 1
    eligible = [g for g in by_group if sizes[g] >= 2]
    return float(np.mean([len(by_group[g]) == 1 for g in eligible])) if eligible else float("nan")
