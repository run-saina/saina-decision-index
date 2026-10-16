import pytest
from decision_index.engines.base import Unsupported, validate

import saina_decision_index.engine as engine_module
from saina.errors import InputTooLong

REVISION = "b85231b8ca2f4f66cfd2e177d59635515474c736"


class FakeScorer:
    """Favours the first option; records what the model would have been shown."""

    max_length = 8192
    codes = ["A", "B", "C", "D"]

    def __init__(self):
        self.calls = []

    def count_text(self, text):
        return len(text.split())

    def encode(self, context, question, choices, mode):
        self.calls.append((context, question, list(choices)))
        if "too long" in context:
            raise InputTooLong("Input exceeds model context window")
        return [1, 2, 3]

    def forward(self, ids, size, mode):
        return [0.6] + [0.4 / (size - 1)] * (size - 1)


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(engine_module, "load_scorer", lambda *a, **k: FakeScorer())
    return engine_module.SainaEngine(revision=REVISION, device="cpu", repo_commit="abc")


def test_requires_pinned_revision(monkeypatch):
    monkeypatch.setattr(engine_module, "load_scorer", lambda *a, **k: FakeScorer())
    with pytest.raises(ValueError, match="immutable"):
        engine_module.SainaEngine(revision="main")


def test_answers_pass_kit_validation(engine):
    questions = {
        "intent": {"type": "choice", "instructions": "Which intent?",
                   "criteria": {"refund": "Wants money back", "track": "Where is my order"}},
        "urgent": {"type": "noul", "instructions": "Is it urgent?",
                   "criteria": {"true": "Needs action today", "false": "Can wait"}},
    }
    response, raw = engine({"message": "I want my money back"}, questions)
    validate(questions, response)
    assert response["answers"]["intent"]["choice"] == "refund"
    assert response["answers"]["urgent"]["noul"] == pytest.approx(0.6)
    assert response["model"] == "run-saina/saina-helm-0.8b@" + REVISION
    # Option keys and descriptions both reach the model; nothing benchmark-specific is added.
    assert engine.scorer.calls[0][2] == ["refund: Wants money back", "track: Where is my order"]
    assert engine.provenance["repo_commit"] == "abc"
    assert engine.provenance["gold_access"] is False


def test_capacity_limits_are_unsupported(engine):
    too_many = {"q": {"type": "choice", "instructions": "Pick", "criteria": {k: None for k in "ABCDE"}}}
    with pytest.raises(Unsupported, match="More than 4 options"):
        engine("state", too_many)
    with pytest.raises(Unsupported):
        engine("too long", {"q": {"type": "noul", "instructions": "Yes?"}})
    with pytest.raises(Unsupported, match="question type"):
        engine("state", {"q": {"type": "rank", "instructions": "Order"}})


def test_conv_kernel_needs_a_pinned_commit():
    with pytest.raises(ValueError, match="40-char commit"):
        engine_module.use_conv_kernel("main")
