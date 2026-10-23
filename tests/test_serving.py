"""Router, safety gate and API tests with a stub backend (no model download needed)."""
import itertools

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bhasharoute.labels import SCENARIO_TO_ID
from bhasharoute.models.base import Predictions
from bhasharoute.serving.api import create_app
from bhasharoute.serving.gate import Tool, may_execute, side_effect_gate
from bhasharoute.serving.router import Router

CAL = {"temperature": 1.0, "side_temperature": 1.0, "abstain_threshold": 0.5, "side_threshold": 0.3}


class StubBackend:
    def __init__(self, scenario: str, margin: float, side_logit: float, fail: bool = False):
        self.k, self.margin, self.side, self.fail = SCENARIO_TO_ID[scenario], margin, side_logit, fail

    def predict(self, texts):
        if self.fail:
            raise RuntimeError("inference exploded")
        logits = np.zeros((len(texts), len(SCENARIO_TO_ID)))
        logits[:, self.k] = self.margin
        side = np.full(len(texts), self.side)
        return Predictions(logits, logits.argmax(1), side, side > 0)


def test_confident_prediction_routes():
    d = Router(StubBackend("weather", 10.0, -5.0), CAL, "stub@1").route("aaj mausam kaisa hai")
    assert d.scenario == "weather" and not d.abstained and d.model_version == "stub@1"
    assert not d.requires_confirmation


def test_low_confidence_abstains():
    d = Router(StubBackend("weather", 0.5, -5.0), CAL, "stub@1").route("hmm okay")
    assert d.abstained and d.scenario is None


def test_inference_failure_falls_back_to_rules_and_fails_closed():
    d = Router(StubBackend("weather", 10.0, -5.0, fail=True), CAL, "stub@1").route("set an alarm for 7")
    assert d.fallback_used and d.model_version == "rules-fallback"
    assert d.scenario == "alarm"
    assert d.requires_confirmation and "model_unavailable_fail_closed" in d.gate_reasons


def test_lexical_trigger_requires_confirmation_even_if_model_says_no():
    d = Router(StubBackend("email", 10.0, -20.0), CAL, "stub@1").route("send an email to mom")
    assert d.requires_confirmation and d.gate_reasons == ["lexical_trigger"]


@pytest.mark.parametrize("text,prob", list(itertools.product(
    ["send the report", "delete my alarm", "बत्ती बंद करो", "লাইট বন্ধ করো", "what's the weather"],
    [0.0, 0.01, 0.29, 0.31, 0.99, None])))
def test_ml_can_only_add_friction(text, prob):
    """Monotonicity: lowering the model's side-effect probability never removes a
    confirmation the lexical guard requires; a missing model always requires one."""
    gate = side_effect_gate(text, prob, threshold=0.3)
    lexical_only = side_effect_gate(text, 0.0, threshold=0.3)
    if lexical_only.requires_confirmation or prob is None:
        assert gate.requires_confirmation
    if prob is not None and prob >= 0.3:
        assert gate.requires_confirmation


def test_tool_policy_ignores_model_output():
    assert may_execute(Tool("send_email", mutates=True), user_confirmed=False) is False
    assert may_execute(Tool("send_email", mutates=True), user_confirmed=True) is True
    assert may_execute(Tool("get_weather", mutates=False), user_confirmed=False) is True


def test_api_contract_and_metrics():
    router = Router(StubBackend("alarm", 10.0, 5.0), CAL, "stub@1")
    client = TestClient(create_app(router))
    r = client.post("/route", json={"text": "kal subah 6 baje alarm laga do"})
    assert r.status_code == 200
    body = r.json()
    assert body["scenario"] == "alarm" and body["requires_confirmation"] is True
    assert set(body) >= {"confidence", "abstained", "side_effect_prob", "model_version", "latency_ms"}
    assert client.get("/healthz").json() == {"status": "ok", "model_version": "stub@1"}
    assert "bhasharoute_requests_total 1" in client.get("/metrics").text
    assert client.post("/route", json={"text": ""}).status_code == 422
