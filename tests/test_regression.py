"""Behavioural regression tests against trained artifacts (skipped when artifacts are absent).

These pin down behaviour a model update must not break: clear requests in every surface
form route correctly, and side-effecting requests are always gated.
"""
import pytest

from bhasharoute.config import ARTIFACTS_DIR

MODEL = "C_xlmr_finetune_aug"
pytestmark = pytest.mark.artifacts

CASES = [
    ("set an alarm for 6 am tomorrow", "alarm", True),
    ("सुबह छह बजे का अलार्म लगा दो", "alarm", True),
    ("kal subah 6 baje ka alarm laga do", "alarm", True),
    ("কাল সকাল ছটায় অ্যালার্ম দাও", "alarm", True),
    ("what's the weather like today", "weather", False),
    ("aaj mausam kaisa hai", "weather", False),
    ("আজ আবহাওয়া কেমন", "weather", False),
    ("turn off the bedroom lights", "iot", True),
    ("lights band kar do", "iot", True),
    ("send an email to my boss saying I'm late", "email", True),
    ("play some music", "play", False),
]


@pytest.fixture(scope="module")
def router():
    if not (ARTIFACTS_DIR / MODEL / "calibration.json").exists():
        pytest.skip(f"{MODEL} not trained/evaluated")
    from bhasharoute.serving.router import Router
    return Router.from_artifacts(MODEL, "cpu")


@pytest.mark.parametrize("text,scenario,side_effect", CASES)
def test_golden_routes(router, text, scenario, side_effect):
    d = router.route(text)
    assert d.scenario == scenario, d
    if side_effect:
        assert d.requires_confirmation, d
