import numpy as np

from bhasharoute.eval.metrics import (binary_report, expected_calibration_error, fit_temperature,
                                      group_consistency, ood_detection, softmax, threshold_for_recall)


def test_ece_zero_when_confidence_matches_accuracy():
    conf = np.full(1000, 0.8)
    correct = np.r_[np.ones(800), np.zeros(200)].astype(bool)
    assert expected_calibration_error(conf, correct) < 1e-9


def test_ece_detects_overconfidence():
    conf = np.full(1000, 0.95)
    correct = np.r_[np.ones(600), np.zeros(400)].astype(bool)
    assert abs(expected_calibration_error(conf, correct) - 0.35) < 1e-6


def test_temperature_scaling_recovers_known_temperature():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(4000, 5)) * 2
    labels = np.array([rng.choice(5, p=p) for p in softmax(true_logits)])
    t = fit_temperature(true_logits * 3.0, labels)  # model 3x over-confident
    assert 2.6 < t < 3.4
    assert np.array_equal(softmax(true_logits * 3, t).argmax(1), true_logits.argmax(1))  # argmax unchanged


def test_ood_detection_perfect_separation():
    res = ood_detection(np.linspace(0.6, 1.0, 100), np.linspace(0.1, 0.5, 100))
    assert res["auroc"] == 1.0 and res["fpr_at_95tpr"] == 0.0


def test_threshold_for_recall_meets_target():
    rng = np.random.default_rng(1)
    y = rng.random(5000) < 0.3
    score = np.where(y, rng.normal(1, 1, 5000), rng.normal(-1, 1, 5000))
    thr = threshold_for_recall(y, score, 0.99)
    assert binary_report(y, score >= thr)["recall"] >= 0.99


def test_group_consistency():
    assert group_consistency(["a", "a", "b", "b", "c"], np.array([1, 1, 2, 3, 4])) == 0.5
