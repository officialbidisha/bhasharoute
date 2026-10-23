"""Production routing interface: model inference + calibration + abstention + side-effect
gate + rule fallback, with model-version tracking, structured logs and counters.

    router = Router.from_artifacts("C_xlmr_finetune_aug")
    decision = router.route("kal subah 6 baje ka alarm laga do")
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass

from bhasharoute.config import ARTIFACTS_DIR
from bhasharoute.eval.metrics import sigmoid, softmax
from bhasharoute.labels import ID_TO_SCENARIO
from bhasharoute.models.rules import KeywordRouter
from bhasharoute.serving.gate import side_effect_gate
from bhasharoute.text import detect_script

log = logging.getLogger("bhasharoute")


@dataclass
class RouteDecision:
    scenario: str | None           # None = abstain ("I can't help with that yet" / clarify)
    confidence: float | None       # calibrated probability of `scenario`
    abstained: bool
    side_effect_prob: float | None
    requires_confirmation: bool
    gate_reasons: list[str]
    model_version: str
    fallback_used: bool
    latency_ms: float


class Router:
    def __init__(self, backend, calibration: dict, version: str, fallback: KeywordRouter | None = None):
        self.backend = backend
        self.cal = calibration
        self.version = version
        self.fallback = fallback or KeywordRouter()
        self.counters: Counter = Counter()
        self.latencies_ms: list[float] = []

    @classmethod
    def from_artifacts(cls, name: str, device: str = "cpu") -> "Router":
        d = ARTIFACTS_DIR / name
        calibration = json.loads((d / "calibration.json").read_text())
        if name.startswith("B_"):
            from bhasharoute.models.tfidf import TfidfClassifier
            backend = TfidfClassifier.load(d)
            digest = hashlib.sha256((d / "model.joblib").read_bytes()).hexdigest()[:12]
            version = f"{name}@{digest}"
        else:
            from bhasharoute.models.transformer_infer import TransformerClassifier
            backend = TransformerClassifier(d, device)
            version = backend.version
        return cls(backend, calibration, version)

    def route(self, text: str) -> RouteDecision:
        t0 = time.perf_counter()
        fallback_used = False
        try:
            p = self.backend.predict([text])
            probs = softmax(p.scenario_logits, self.cal["temperature"])[0]
            k = int(probs.argmax())
            confidence = float(probs[k])
            abstained = confidence < self.cal["abstain_threshold"]
            scenario = None if abstained else ID_TO_SCENARIO[k]
            side_prob = float(sigmoid(p.side_logit, self.cal["side_temperature"])[0])
        except Exception:  # inference failure must not take the assistant down
            log.exception("model inference failed; using rule fallback")
            fallback_used = True
            scenario = self.fallback.route(text)
            confidence, abstained, side_prob = None, scenario is None, None  # gate fails closed
        gate = side_effect_gate(text, side_prob, self.cal["side_threshold"])
        decision = RouteDecision(scenario, confidence, abstained, side_prob, gate.requires_confirmation,
                                 list(gate.reasons), "rules-fallback" if fallback_used else self.version,
                                 fallback_used, round((time.perf_counter() - t0) * 1000, 2))
        self._record(text, decision)
        return decision

    def _record(self, text: str, d: RouteDecision) -> None:
        self.counters["requests_total"] += 1
        self.counters["abstained_total"] += d.abstained
        self.counters["confirmation_required_total"] += d.requires_confirmation
        self.counters["fallback_total"] += d.fallback_used
        self.latencies_ms = (self.latencies_ms + [d.latency_ms])[-1000:]
        # Structured log without the raw request (privacy): hash, length and script only.
        log.info(json.dumps({"event": "route", "text_sha256_8": hashlib.sha256(text.encode()).hexdigest()[:8],
                             "chars": len(text), "script": detect_script(text), **asdict(d)}))

    def prometheus(self) -> str:
        lines = [f"bhasharoute_{k} {v}" for k, v in sorted(self.counters.items())]
        if self.latencies_ms:
            s = sorted(self.latencies_ms)
            lines += [f'bhasharoute_latency_ms{{quantile="0.5"}} {s[len(s) // 2]}',
                      f'bhasharoute_latency_ms{{quantile="0.95"}} {s[max(0, int(len(s) * 0.95) - 1)]}']
        lines.append(f'bhasharoute_model_info{{version="{self.version}"}} 1')
        return "\n".join(lines) + "\n"
